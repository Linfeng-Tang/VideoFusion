import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .Transformer_util import Co_TransformerBlock, TransformerBlock, Cross_TransformerBlock
from einops import rearrange
import math
import numbers

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # Ensure n_feat is divisible by 4 for PixelUnshuffle
        assert n_feat % 4 == 0, "n_feat must be divisible by 4 for PixelUnshuffle."
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # Ensure n_feat is divisible by 2
        assert n_feat % 2 == 0, "n_feat must be divisible by 2."
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return self.body(x)

    
# 3D卷积模块
class Conv3DModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        super(Conv3DModule, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.module(x)


class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=2):
        super().__init__()
        self.blocks = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, x):
        return self.blocks(x)


# 融合模块   
# dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group
class FusionModule(nn.Module):
    def __init__(self, 
                 in_channels=192,
                 num_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 embed_dim=64,
                 group=4):
        
        super(FusionModule, self).__init__()
        
        self.cross_transformer = Cross_TransformerBlock(
            dim=in_channels, 
            num_heads=num_heads, 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            embed_dim=embed_dim, 
            group=group)
        
    def forward(self, ir, vi):
        # Ensure ir and vi have the shape (b*t, c, h, w)
        assert ir.shape == vi.shape, "ir and vi must have the same shape."

        # Compute Q, K, V
        Q = ir + vi
        # Compute cross-attention for IR and VI
        fusion_ir = self.cross_transformer(Q, ir)  # Cross-attention with IR
        fusion_vi = self.cross_transformer(Q, vi)  # Cross-attention with VI
        # Fuse the two outputs
        fusion = fusion_ir + fusion_vi
        return fusion



# Define the CoAttentionModule class
class CoAttentionModule(nn.Module):
    def __init__(self, 
                 in_channels=192,
                 num_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 embed_dim=64,
                 group=4):
        
        super(CoAttentionModule, self).__init__()
        self.co_transformer = Co_TransformerBlock(
            dim=in_channels, 
            num_heads=num_heads, 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            embed_dim=embed_dim, 
            group=group)

    def forward(self, x):
        """
        x shape: b t c h w  (batch_size, time, channels, height, width)
        """
        b, t, c, h, w = x.shape
        # Initialize a list to store the updated frames
        updated_frames = []
        for i in range(t):
            curr_frame = x[:, i, :, :, :]  # shape: b c h w
            if i == 0:
               # First frame: calculate attention between F0 and F1
                prev_frame = curr_frame
                next_frame = x[:, i + 1, :, :, :]
            elif i == t - 1:
                # Last frame: calculate attention between Fn-1 and Fn (Fn uses itself as the next frame)
                prev_frame = x[:, i - 1, :, :, :]
                next_frame = curr_frame
            else:
                # For all other frames: use both the previous and next frame
                prev_frame = x[:, i - 1, :, :, :]
                next_frame = x[:, i + 1, :, :, :]

            # Ensure the shape of prev_frame, curr_frame, next_frame match expectations (b, c, h, w)
            assert prev_frame.shape == curr_frame.shape == next_frame.shape, \
                f"Shape mismatch: {prev_frame.shape}, {curr_frame.shape}, {next_frame.shape}"
                
            updated_frame = self.co_transformer(curr_frame, prev_frame, next_frame)

            # Append the updated frame to the list
            updated_frames.append(updated_frame)

        # Stack updated frames along the time dimension
        updated_x = torch.stack(updated_frames, dim=1)  # shape: b t c h w
        return updated_x


class StackedCoAttention(nn.Module):
    def __init__(self,
                in_channels=192,
                num_layers=4,
                num_heads=4,
                ffn_expansion_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                embed_dim=64,
                group=4):
        super(StackedCoAttention, self).__init__()
        self.layers = nn.ModuleList([CoAttentionModule(
                                    in_channels=in_channels,
                                    num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias,
                                    LayerNorm_type=LayerNorm_type,
                                    embed_dim=embed_dim,
                                    group=group) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Ensure that x maintains the same shape after each layer
        return x

    
class CrossModalityDiffEnhance(nn.Module):
    def __init__(self, 
                 in_channels=192,
                 num_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 embed_dim=64,
                 group=4):
        super(CrossModalityDiffEnhance, self).__init__()
        self.cross_transformer = Cross_TransformerBlock(
            dim=in_channels, 
            num_heads=num_heads, 
            ffn_expansion_factor=ffn_expansion_factor, 
            bias=bias, 
            LayerNorm_type=LayerNorm_type, 
            embed_dim=embed_dim, 
            group=group)

    def forward(self, x, suppl):
        """
        x: Visible video input, shape: (b*t, c, h, w)
        suppl: Infrared video input, shape: (b*t, c, h, w)
        """
        # Ensure x and suppl have the same shape
        assert x.shape == suppl.shape, f"Shape mismatch: x shape {x.shape} and suppl shape {suppl.shape} must be the same."

        diff = suppl - x  # (b*t, c, h, w)
        enhanced_x = self.cross_transformer(x, diff)
        return enhanced_x

      

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return out

class CrossModalAggregation(nn.Module):
    def __init__(self, in_channels):
        super(CrossModalAggregation, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, 2, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ChannelAttention = ChannelAttention(in_channels)
        self.SpatialAttention = SpatialAttention()

    def forward(self, x_main, x_supple):
        # Concatenate the main and supplementary modalities
        x = torch.cat([x_main, x_supple], dim=1)
        x = self.conv(x)

        # Split the output into two parts: one for each modality
        w_main, w_supple = x[:, 0:1, :, :], x[:, 1:2, :, :]
        w_main = self.sigmoid(w_main)
        w_supple = self.sigmoid(w_supple)

        # Pooling the weights
        w_main = self.pool(w_main)
        w_supple = self.pool(w_supple)

        # Apply the weights to the input features
        g_inp1 = w_main * x_main
        g_inp2 = w_supple * x_supple

        # Apply Channel and Spatial Attention to the main modality
        x_main_CA = self.ChannelAttention(g_inp1) * g_inp1
        x_main_CA_SA = self.SpatialAttention(x_main_CA) * x_main_CA

        # Fuse the main and supplementary modalities
        fuse = x_main_CA_SA + g_inp2
        return fuse

    
# 分解模块
class VideoDecomposition(nn.Module):
    def __init__(self, in_channels):
        super(VideoDecomposition, self).__init__()
        self.channel_attention_ir = ChannelAttention(in_channels)
        self.channel_attention_vi = ChannelAttention(in_channels)
        self.spatial_attention_ir = SpatialAttention()
        self.spatial_attention_vi = SpatialAttention()
        
    def forward(self, x):
        # F shape: (bt) c h w
        # 使用通道注意力进行特征分解
        x_ir = self.channel_attention_ir(x) * x  # 使用通道注意力进行特征筛选
        x_ir = self.spatial_attention_ir(x_ir) * x_ir  # 使用空间注意力进行特征筛选
        
        x_vi = self.channel_attention_vi(x) * x  # 使用通道注意力进行特征筛选
        x_vi = self.spatial_attention_vi(x_vi) * x_vi  # 使用空间注意力进行特征筛选
        return x_ir, x_vi

    

class FusionEnhance(nn.Module):
    def __init__(self, 
            in_channels=64, 
            num_blocks=2,
            group=4,
            num_heads=2,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            embed_dim=64):
        super(FusionEnhance, self).__init__()
        # 使用 3D 卷积来恢复视频
        self.transformer = nn.Sequential(*[TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type,embed_dim=embed_dim,group=group,) for _ in range(num_blocks)])
        
    def forward(self, x):
        x = self.transformer(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, 
            in_channels=64, 
            out_channels=64,
            num_blocks=2,
            group=4,
            num_heads=2,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            embed_dim=64):
        super(Decoder3D, self).__init__()
        # 使用 3D 卷积来恢复视频
        self.decoder = nn.Sequential(*[TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group,) for _ in range(num_blocks)])

        self.conv = nn.Sequential(
           nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=True),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def forward(self, x):
        x = self.decoder(x)
        x = self.conv(x)
        return x
    
@ARCH_REGISTRY.register()     
class VideoFusion_Trans(nn.Module):
    def __init__(self, 
                in_channels=3, 
                num_features=64,
                out_channels=3,
                num_blocks = [2, 2, 4], 
                num_decoder_blocks = 2,
                num_heads = [2, 4, 8],
                ffn_expansion_factor = 2.66,
                bias = False,
                LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                embed_dim = 64,
                groups=[4, 8, 16],
                ):
        super(VideoFusion_Trans, self).__init__()

        # 3D convolution layers for infrared and visible inputs
        self.conv3d_ir_1 = Conv3DModule(in_channels, num_features)
        self.conv3d_vi_1 = Conv3DModule(in_channels, num_features)

        # Fusion and enhancement layers
        self.fusion_1 = self._create_fusion_layer(num_features, num_heads[0], ffn_expansion_factor, groups[0], embed_dim, LayerNorm_type, bias)
        self.fusion_enhance_1 = self._create_fusion_enhance_layer(num_features, num_blocks[0], ffn_expansion_factor, groups[0], num_heads[0], embed_dim, LayerNorm_type, bias)
        
        # Downsample and convolution enhancements
        self.down_ir_1 = Downsample(num_features)
        self.down_vi_1 = Downsample(num_features)

        self.conv3d_ir_2 = Conv3DModule(num_features*2, num_features*2)
        self.conv3d_vi_2 = Conv3DModule(num_features*2, num_features*2)
        
        self.ir_conv_enhance_2 = ResidualBlocks2D(num_feat=num_features*2, num_block=4)
        self.vi_conv_enhance_2 = ResidualBlocks2D(num_feat=num_features*2, num_block=4)
        
        self.cross_modal_enhance_ir_2 = CrossModalityDiffEnhance(num_features*2, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type, groups[1])
        self.cross_modal_enhance_vi_2 = CrossModalityDiffEnhance(num_features*2, num_heads[1], ffn_expansion_factor, bias, LayerNorm_type, groups[1])
        
        # Cross-modal aggregation
        self.CMA_ir_2 = CrossModalAggregation(num_features*2)
        self.CMA_vi_2 = CrossModalAggregation(num_features*2)
        
        self.fusion_2 = self._create_fusion_layer(num_features*2, num_heads[1], ffn_expansion_factor, groups[1], embed_dim, LayerNorm_type, bias)
        self.fusion_enhance_2 = self._create_fusion_enhance_layer(num_features*2, num_blocks[1], ffn_expansion_factor, groups[1], num_heads[1], embed_dim, LayerNorm_type, bias)
        
        # Downsample to next scale
        self.down_ir_2 = Downsample(num_features*2)
        self.down_vi_2 = Downsample(num_features*2)

        self.conv3d_ir_3 = Conv3DModule(num_features*4, num_features*4)
        self.conv3d_vi_3 = Conv3DModule(num_features*4, num_features*4)
        self.ir_conv_enhance_3 = ResidualBlocks2D(num_feat=num_features*4, num_block=4)
        self.vi_conv_enhance_3 = ResidualBlocks2D(num_feat=num_features*4, num_block=4)

        # Cross-modal attention and enhancement
        # self.co_attention = StackedCoAttention(num_features*4, num_layers=num_blocks[-1], num_heads=num_heads[-1], embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias, group=groups[-1])
        self.cross_modal_enhance_ir_3 = CrossModalityDiffEnhance(num_features*4, num_heads[-1], ffn_expansion_factor, bias, LayerNorm_type, groups[-1])
        self.cross_modal_enhance_vi_3 = CrossModalityDiffEnhance(num_features*4, num_heads[-1], ffn_expansion_factor, bias, LayerNorm_type, groups[-1])
        
        # Cross-modal aggregation
        self.CMA_ir_3 = CrossModalAggregation(num_features*4)
        self.CMA_vi_3 = CrossModalAggregation(num_features*4)
        
        # Final fusion and upsampling
        self.fusion_3 = self._create_fusion_layer(num_features*4, num_heads[2], ffn_expansion_factor, groups[2], embed_dim, LayerNorm_type, bias)
        self.co_attention_3 = StackedCoAttention(num_features*4, num_layers=num_blocks[2], ffn_expansion_factor=ffn_expansion_factor, num_heads=num_heads[2], embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias, group=groups[1])

        self.up2 = Upsample(num_features*4)        
        self.co_attention_2 = StackedCoAttention(num_features*2, num_layers=num_blocks[1], ffn_expansion_factor=ffn_expansion_factor, num_heads=num_heads[1], embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias, group=groups[1])
        
        self.up1 = Upsample(num_features*2)
        self.co_attention_1 = StackedCoAttention(num_features*1, num_layers=num_blocks[0], ffn_expansion_factor=ffn_expansion_factor, num_heads=num_heads[0], embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias, group=groups[0])
        
        # Decomposition and decoders
        self.decomposition = VideoDecomposition(num_features)
        self.decoder_ir = Decoder3D(num_features, out_channels, num_blocks=num_decoder_blocks, group=groups[0], num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias)
        self.decoder_vi = Decoder3D(num_features, out_channels, num_blocks=num_decoder_blocks, group=groups[0], num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias)
        self.decoder_fusion = Decoder3D(num_features, out_channels, num_blocks=num_decoder_blocks, group=groups[0], num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type, bias=bias)
        self.Tanh = nn.Tanh()

    def _create_fusion_layer(self, in_channels, num_heads, ffn_expansion_factor, group, embed_dim, LayerNorm_type, bias):
        return FusionModule(in_channels=in_channels,
                            num_heads=num_heads,
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            embed_dim=embed_dim,
                            group=group)

    def _create_fusion_enhance_layer(self, in_channels, num_blocks, ffn_expansion_factor, group, num_heads, embed_dim, LayerNorm_type, bias):
        return FusionEnhance(in_channels=in_channels,
                             num_blocks=num_blocks,
                             ffn_expansion_factor=ffn_expansion_factor,
                             group=group,
                             num_heads=num_heads,
                             bias=bias,
                             LayerNorm_type=LayerNorm_type,
                             embed_dim=embed_dim)

    def forward(self, x_ir, x_vi):
        b, t, c, h, w = x_ir.shape
        x_ir_T = rearrange(x_ir, 'b t c h w -> b c t h w')
        x_vi_T = rearrange(x_vi, 'b t c h w -> b c t h w')

        # First scale: 3D convolutions and fusion
        f_ir_1 = self.conv3d_ir_1(x_ir_T)
        f_vi_1 = self.conv3d_vi_1(x_vi_T)
        f_ir_1 = rearrange(f_ir_1, 'b c t h w -> (b t) c h w')
        f_vi_1 = rearrange(f_vi_1, 'b c t h w -> (b t) c h w')

        f_fusion_1 = self.fusion_1(f_ir_1, f_vi_1)
        f_fusion_1 = self.fusion_enhance_1(f_fusion_1)

        # Downsample and enhance
        f_ir_2 = self.down_ir_1(f_ir_1)
        f_vi_2 = self.down_vi_1(f_vi_1)
        
        # 3D convolutions for the second scale
        f_ir_2 = rearrange(f_ir_2, '(b t) c h w -> b c t h w', b=b)
        f_vi_2 = rearrange(f_vi_2, '(b t) c h w -> b c t h w', b=b)
        f_ir_2 = self.conv3d_ir_2(f_ir_2)
        f_vi_2 = self.conv3d_vi_2(f_vi_2)
        f_ir_2 = rearrange(f_ir_2, 'b c t h w -> (b t) c h w')
        f_vi_2 = rearrange(f_vi_2, 'b c t h w -> (b t) c h w')
        
        f_ir_2 = self.ir_conv_enhance_2(f_ir_2)
        f_vi_2 = self.vi_conv_enhance_2(f_vi_2)

        # Cross-modal enhancement
        f_ir_2_enhance = self.cross_modal_enhance_ir_2(f_ir_2, f_vi_2)
        f_vi_2_enhance = self.cross_modal_enhance_vi_2(f_vi_2, f_ir_2)

        # Cross-modal aggregation
        f_ir_2_enhance = self.CMA_ir_2(f_ir_2_enhance, f_ir_2)
        f_vi_2_enhance = self.CMA_vi_2(f_vi_2_enhance, f_vi_2)
        
        # Fusion and enhancement at second scale
        f_fusion_2 = self.fusion_2(f_ir_2_enhance, f_vi_2_enhance)
        f_fusion_2 = self.fusion_enhance_2(f_fusion_2)

        # Downsample to third scale
        f_ir_3 = self.down_ir_2(f_ir_2_enhance)
        f_vi_3 = self.down_vi_2(f_vi_2_enhance)
        
        # 3D convolutions for the second scale
        f_ir_3 = rearrange(f_ir_3, '(b t) c h w -> b c t h w', b=b)
        f_vi_3 = rearrange(f_vi_3, '(b t) c h w -> b c t h w', b=b)
        f_ir_3 = self.conv3d_ir_3(f_ir_3)
        f_vi_3 = self.conv3d_vi_3(f_vi_3)
        f_ir_3 = rearrange(f_ir_3, 'b c t h w -> (b t) c h w')
        f_vi_3 = rearrange(f_vi_3, 'b c t h w -> (b t) c h w')
        
        f_ir_3 = self.ir_conv_enhance_3(f_ir_3)
        f_vi_3 = self.vi_conv_enhance_3(f_vi_3)

        # Cross-modal enhancement
        f_ir_3_enhance = self.cross_modal_enhance_ir_3(f_ir_3, f_vi_3)
        f_vi_3_enhance = self.cross_modal_enhance_vi_3(f_vi_3, f_ir_3)

        # Cross-modal aggregation
        f_ir_3_enhance = self.CMA_ir_3(f_ir_3_enhance, f_ir_3)
        f_vi_3_enhance = self.CMA_vi_3(f_vi_3_enhance, f_vi_3)

        # Fusion at the third scale
        f_fusion_3 = self.fusion_3(f_ir_3_enhance, f_vi_3_enhance)

        # Inter-frame attention
        f_fusion_3 = rearrange(f_fusion_3, '(b t) c h w -> b t c h w', b=b, t=t)
        f_fusion_3 = self.co_attention_3(f_fusion_3)

        # Upsample and reconstruct
        f_fusion_3 = rearrange(f_fusion_3, 'b t c h w -> (b t) c h w')
        f_fusion_2 = self.up2(f_fusion_3) + f_fusion_2
        f_fusion_2 = rearrange(f_fusion_2, '(b t) c h w -> b t c h w', b=b, t=t)
        f_fusion_2 = self.co_attention_2(f_fusion_2)
        f_fusion_2 = rearrange(f_fusion_2, 'b t c h w -> (b t) c h w')

        f_fusion_1 = self.up1(f_fusion_2) + f_fusion_1
        f_fusion_1 = rearrange(f_fusion_1, '(b t) c h w -> b t c h w', b=b, t=t)
        f_fusion_1 = self.co_attention_1(f_fusion_1)
        f_fusion_1 = rearrange(f_fusion_1, 'b t c h w -> (b t) c h w')

        # Decompose and decode
        f_ir_rec, f_vi_rec = self.decomposition(f_fusion_1)

        ir_video = self.decoder_ir(f_ir_rec)
        vi_video = self.decoder_vi(f_vi_rec)
        fusion_video = self.decoder_fusion(f_fusion_1)

        # Post-processing with Tanh and reconstruction
        # ir_video = rearrange(ir_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous() + x_ir
        # vi_video = rearrange(vi_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous() + x_vi
        # fusion_video = rearrange(fusion_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous() + x_vi
        
        ir_video = rearrange(ir_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous()
        vi_video = rearrange(vi_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous()
        fusion_video = rearrange(fusion_video, '(b t) c h w -> b t c h w', b=b, t=t).contiguous()

        fusion_video = (self.Tanh(fusion_video) + 1) / 2
        ir_video = (self.Tanh(ir_video) + 1) / 2
        vi_video = (self.Tanh(vi_video) + 1) / 2

        return {"fusion": fusion_video, "ir": ir_video, "vi": vi_video}

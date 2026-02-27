import math
import torch
from torch import autograd as autograd
from math import exp
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torch.fft
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


def RGB2YCrCb(rgb_image, with_CbCr=False):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :param with_CbCr: boolean flag to determine if Cb and Cr channels should be returned
    :return: Y, CbCr (if with_CbCr is True), otherwise Y, Cb, Cr
    """
    R, G, B = rgb_image[:, 0:1, ::], rgb_image[:, 1:2, ::], rgb_image[:, 2:3, ::]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128/255.0

    Y, Cb, Cr = Y.clamp(0.0, 1.0), Cb.clamp(0.0, 1.0), Cr.clamp(0.0, 1.0)
    
    if with_CbCr:
        return Y, torch.cat([Cb, Cr], dim=1)
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    R = Y + 1.402 * (Cr - 128/255.0)
    G = Y - 0.344136 * (Cb - 128/255.0) - 0.714136 * (Cr - 128/255.0)
    B = Y + 1.772 * (Cb - 128/255.0)
    return torch.cat([R, G, B], dim=1).clamp(0, 1.0)

@LOSS_REGISTRY.register()
class TemporalConsistency_loss(nn.Module):
    def __init__(self):
        super(TemporalConsistency_loss, self).__init__()

    def forward(self, V_f, V_ir):
        """
        计算时间一致性损失（L1 Loss）

        Args:
            V_f (torch.Tensor): 融合视频，形状为 [B, T, C, H, W]
            V_ir (torch.Tensor): 红外视频，形状为 [B, T, C, H, W]

        Returns:
            torch.Tensor: 时间一致性损失值
        """
        # 确保输入形状一致
        assert V_f.shape == V_ir.shape, "The shapes of V_f and V_ir must be the same."
        
        # 计算时间差分
        diff_f = V_f[:, 1:] - V_f[:, :-1]  # [B, T-1, C, H, W]
        diff_ir = V_ir[:, 1:] - V_ir[:, :-1]  # [B, T-1, C, H, W]

        # 使用 L1 损失计算时间一致性
        loss = F.l1_loss(diff_f, diff_ir)

        return loss

    

@LOSS_REGISTRY.register()
class Fidelity_loss(nn.Module):
    def __init__(self):
        super(Fidelity_loss, self).__init__()
        print('Using Fidelity_loss() as loss function ~')
        self.grad_operator = Gradient()
        self.loss_func = nn.L1Loss(reduction='mean')
        
    def forward(self, img_rec, img_ref, type='ir'):
        Y_rec, _, _ = RGB2YCrCb(img_rec)
        Y_ref, _, _ = RGB2YCrCb(img_ref)
        grad_rec_x, grad_rec_y = self.grad_operator(Y_rec)
        grad_ref_x, grad_ref_y = self.grad_operator(Y_ref)
        loss_intensity = 10 * self.loss_func(img_rec, img_ref)
        loss_grad = 10 * self.loss_func(grad_rec_x, grad_ref_x) + 10 * self.loss_func(grad_rec_y, grad_ref_y)
        loss_fidelity = loss_intensity + loss_grad
        
        return {
            f'{type}_loss_intensity': loss_intensity,
            f'{type}_loss_grad': loss_grad,
            f'{type}_loss_fid': loss_fidelity
        }
    
@LOSS_REGISTRY.register()
class Fusion_loss(nn.Module):
    def __init__(self):
        super(Fusion_loss, self).__init__()        
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_max = L_Intensity_Max()
        self.loss_func_consist = L_Intensity_Consist()
        self.loss_func_color = L_Color()
        # self.loss_func_fft = FrequencyLoss()

    # ir_compose=1.2 grad_weight=5
    def forward(self, img_f, img_ir, img_vi, int_weight=15, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=0.8, color_weight=100, grad_weight=1, fft_weight=0.05, regular=False):
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi)
        Y_ir, _, _ = RGB2YCrCb(img_ir)
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f)

        loss_ssim = ssim_weight * (self.loss_func_ssim(img_vi, img_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_max = int_weight * self.loss_func_max(Y_f, Y_vi, Y_ir)
        loss_consist = consist_weight * self.loss_func_consist(Y_f, Y_vi, Y_ir, ir_compose)
        loss_color = color_weight * self.loss_func_color(Cb_f, Cr_f, Cb_vi, Cr_vi)
        # loss_color = color_weight * self.loss_func_color(img_f, img_vi)
        loss_grad = grad_weight * self.loss_func_grad(Y_f, Y_vi, Y_ir)
        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_grad
         
        return {
            'loss_intensity_max': loss_max,
            'loss_color': loss_color,
            'loss_grad': loss_grad,
            'loss_intensity_consist': loss_consist,
            'loss_ssim': loss_ssim,
            # 'loss_fft': loss_fft,
            'loss_fusion': total_loss
        }

  
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
    
    def forward(self, img_f, img_vi, img_ir):
        # 计算图像的频域表示
        img_f_fft = torch.fft.fft2(img_f)
        img_vi_fft = torch.fft.fft2(img_vi)
        # img_ir_fft = torch.fft.fft2(img_ir)
        
        # 计算幅度差异
        magnitude_diff_vi = torch.abs(img_f_fft) - torch.abs(img_vi_fft)
        freq_loss_vi = torch.mean(torch.abs(magnitude_diff_vi))  # 可见光频域损失
        
        # magnitude_diff_ir = torch.abs(img_f_fft) - torch.abs(img_ir_fft)
        # freq_loss_ir = torch.mean(torch.abs(magnitude_diff_ir))  # 红外频域损失
        
        # 返回总频域损失
        return freq_loss_vi
    
    
class L_Color_Cos(nn.Module):
    def __init__(self):
        super(L_Color_Cos, self).__init__()
    def forward(self, img_f, img_vi):
        img_ref = F.normalize(img_vi, p=2, dim=1)
        ref_p = F.normalize(img_f, p=2, dim=1)
        loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
        return loss_cos 

class L_Intensity_Max(nn.Module):
    def __init__(self):
        super(L_Intensity_Max, self).__init__()

    def forward(self, img_f, img_vi, img_ir):
        img_max = torch.max(img_vi, img_ir)
        return F.l1_loss(img_max, img_f)

class L_Color(nn.Module):
    def __init__(self):
        super(L_Color, self).__init__()

    def forward(self, Cb_f, Cr_f, Cb_vi, Cr_vi):
        return F.l1_loss(Cb_f, Cb_vi) + F.l1_loss(Cr_f, Cr_vi)

class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, img_f, img_vi, img_ir, ir_compose, consist_mode="l1"):
        loss_func = F.mse_loss if consist_mode == "l2" else F.l1_loss
        return loss_func(img_vi, img_f) + ir_compose * loss_func(img_ir, img_f)

class L_Gradient_Max(nn.Module):
    def __init__(self):
        super(L_Gradient_Max, self).__init__()
        self.grad_operator = Gradient()

    def forward(self, img_f, img_vi, img_ir):
        # 计算融合图像的梯度
        grad_f_x, grad_f_y = self.grad_operator(img_f)
        ## 计算源图像的梯度
        grad_vi_x, grad_vi_y = self.grad_operator(img_vi)
        grad_ir_x, grad_ir_y = self.grad_operator(img_ir)
        # 计算梯度幅值最大值一致性损失
        loss = F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x)) + F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y))
        return loss


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(img, self.sobel_x)
        grad_y = F.conv2d(img, self.sobel_y)
        return torch.abs(grad_x), torch.abs(grad_y)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    # print(img1.device, window.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


# Classes to re-use window
class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device).type(img1.dtype)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
    
@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)
@LOSS_REGISTRY.register()
class BrightnessConsistency_loss(nn.Module):
    def __init__(self):
        super(BrightnessConsistency_loss, self).__init__()
        
    def forward(self, x):
        """
        输入 x: 形状为 (batch, time, channel, height, width) 的视频序列
        返回: 相邻帧间的亮度变化损失
        """
        # 转换为灰度图
        # 使用ITU-R BT.601标准的权重
        r, g, b = x.split(1, dim=2)
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # (B,T,1,H,W)
        
        # 计算每帧的平均亮度
        global_brightness = gray.mean(dim=(-2,-1))  # (B,T,1)
        
        # 计算相邻帧之间的亮度差异
        brightness_diff = global_brightness[:, 1:] - global_brightness[:, :-1]  # (B,T-1,1)
        
        # 计算L1损失
        loss = torch.abs(brightness_diff).mean()        
        return loss
    
@LOSS_REGISTRY.register()
class LowFrequencyConsistency_loss(torch.nn.Module):
    """
    用于约束相邻帧在低频分量上差异的损失函数。
    """
    def __init__(self, keep_ratio=0.1):
        super(LowFrequencyConsistency_loss, self).__init__()
        self.keep_ratio = keep_ratio

    def extract_low_frequency(self, frame_tensor, keep_ratio=0.1):
        """
        提取帧的低频分量。
        :param frame_tensor: 输入帧张量 (C, H, W)
        :param keep_ratio: 保留的低频比例
        :return: 低频分量张量 (C, H, W)
        """
        C, H, W = frame_tensor.shape
        fft_frame = torch.fft.fft2(frame_tensor)  # 计算FFT
        fft_frame = torch.fft.fftshift(fft_frame)  # 将低频移动到中心

        # 创建低频掩码
        center_h, center_w = H // 2, W // 2
        keep_h, keep_w = int(H * keep_ratio // 2), int(W * keep_ratio // 2)
        mask = torch.zeros((H, W), dtype=torch.bool, device=frame_tensor.device)
        mask[center_h - keep_h:center_h + keep_h, center_w - keep_w:center_w + keep_w] = True

        # 将掩码扩展到与 (C, H, W) 匹配
        mask = mask.unsqueeze(0).expand(C, -1, -1)

        # 只保留低频分量
        low_freq_fft = torch.zeros_like(fft_frame)
        low_freq_fft[mask] = fft_frame[mask]
        return low_freq_fft
    
    def forward(self, frames):
        """
        计算低频一致性损失。
        :param frames: 视频帧序列 (B, T, C, H, W)，
                       B: batch size, T: 时间步数, C: 通道数, H: 高度, W: 宽度
        :return: 平均低频一致性损失 (标量)
        """
        B, T, C, H, W = frames.shape
        loss = 0.0
        for b in range(B):
            for t in range(T - 1):
                # 提取相邻帧的低频分量
                low_freq_1 = self.extract_low_frequency(frames[b, t, ::], self.keep_ratio)
                low_freq_2 = self.extract_low_frequency(frames[b, t + 1, ::], self.keep_ratio)
                
                # 使用 F.mse_loss 计算损失
                loss += F.mse_loss(torch.view_as_real(low_freq_1), torch.view_as_real(low_freq_2))
        
        # 对 batch 和时间轴上的损失进行归一化
        return loss / (B * (T - 1))

@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class HEM(nn.Module):
    def __init__(self, loss_weight=2.0):
        super(HEM, self).__init__()
        self.hard_thre_p = 0.5
        self.random_thre_p = 0.1
        self.L1_loss = nn.L1Loss()
        self.device = 'cuda'
        self.loss_weight = loss_weight

    def hard_mining_mask(self, x, y):
        with torch.no_grad():
            b, c, h, w = x.size()

            hard_mask = np.zeros(shape=(b, 1, h, w))
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
            res_numpy = res.cpu().numpy()
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)]
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind].item()
                hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

            random_thre_ind = int(self.random_thre_p * w * h)
            random_mask = np.zeros(shape=(b, 1 * h * w))
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                np.random.shuffle(random_mask[i])
            random_mask = np.reshape(random_mask, (b, 1, h, w))

            mask = hard_mask + random_mask
            mask = (mask > 0.).astype(np.float32)

            mask = torch.Tensor(mask).to(self.device)

        return mask

    def forward(self, x, y):
        mask = self.hard_mining_mask(x.detach(), y.detach()).detach()
        hem_loss = self.L1_loss(x * mask, y * mask)
        return self.loss_weight * hem_loss


@LOSS_REGISTRY.register()
class FFT(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFT, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        # pred_fft_concat = torch.cat([pred_fft.real, pred_fft.imag], dim=1)
        # target_fft_concat = torch.cat([target_fft.real, target_fft.imag], dim=1)
        # print(pred_fft.shape, target_fft.shape)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=weight[:, :, :-1, :])
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=weight[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operaton: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. ' f'Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        loss = 0
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_D
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class HistogramConsistency_loss(nn.Module):
    def __init__(self, eps=1e-6, bins=256):
        super(HistogramConsistency_loss, self).__init__()
        self.eps = eps  # 控制正则项的权重
        self.bins = bins  # 直方图的bin数（默认256）
        
    def quantize(self, x):
        """
        将 [0, 1] 范围的连续值映射到 [0, 255] 范围的整数，用于计算直方图。
        :param x: 输入数据，范围 [0, 1]
        :return: 量化后的整数数据
        """
        return torch.clamp((x * 255).round().long(), min=0, max=255)

    def forward(self, x):
        """
        计算相邻两帧之间的直方图差异，返回差异的均值。
        
        :param x: 输入数据，形状为 (b, t, c, h, w)，
                  b - 批量大小，t - 时间帧数，c - 通道数（如RGB），
                  h - 图像高度，w - 图像宽度。
        :return: 所有相邻帧之间直方图差异的均值
        """
        b, t, c, h, w = x.shape
        hist_diff = []

        # 对于每一对相邻帧，计算直方图差异
        for i in range(1, t):
            frame_t_1 = x[:, i-1]  # 第t-1帧
            frame_t = x[:, i]      # 第t帧

            # 量化每个帧的数据
            frame_t_1_quantized = self.quantize(frame_t_1)
            frame_t_quantized = self.quantize(frame_t)

            # 计算每个通道的直方图差异
            diff_per_channel = []
            for channel in range(c):
                # 计算每个通道的直方图
                hist_t_1 = torch.histc(frame_t_1_quantized[:, channel].float(), bins=self.bins, min=0, max=255)
                hist_t = torch.histc(frame_t_quantized[:, channel].float(), bins=self.bins, min=0, max=255)

                # 归一化直方图
                hist_t_1 = hist_t_1 / (hist_t_1.sum() + self.eps)
                hist_t = hist_t / (hist_t.sum() + self.eps)

                # 计算MSE作为直方图差异
                hist_diff_channel = F.l1_loss(hist_t_1, hist_t)
                diff_per_channel.append(hist_diff_channel)

            # 求平均通道差异
            hist_diff.append(torch.mean(torch.stack(diff_per_channel)))

        # 返回所有相邻帧之间的直方图差异均值
        return torch.mean(torch.stack(hist_diff))

  
    
if __name__ == '__main__':    
    x_ir = torch.randn(2, 10, 3, 64, 64)  # 假设红外图像
    x_vi = torch.randn(2, 10, 3, 64, 64)  # 假设可见光图像
    x_f = torch.randn(2, 10, 3, 64, 64)  # 假设融合图像
    fusion_loss = Fusion_loss()
    loss = fusion_loss(x_f, x_ir, x_vi)
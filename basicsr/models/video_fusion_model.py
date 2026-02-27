import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import numpy as np

from torch.nn import functional as F
from einops import rearrange
from collections import OrderedDict
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from time import time
import statistics
from thop import profile


@MODEL_REGISTRY.register()
class VideoFusionModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoFusionModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.weight_pix = train_opt['pixel_opt'].get('weight', 1.0)
        else:
            self.cri_pix = None
            
        if train_opt.get('fusion_opt'):
            self.cri_fus = build_loss(train_opt['fusion_opt']).to(self.device)
            self.weight_fus = train_opt['fusion_opt'].get('weight', 1.0)
        else:
            self.cri_fus = None
            
        if train_opt.get('fidelity_opt'):
            self.cri_fid = build_loss(train_opt['fidelity_opt']).to(self.device)
            self.weight_fid = train_opt['fidelity_opt'].get('weight', 1.0)
        else:
            self.cri_fid = None
            # consistency_opt
        if train_opt.get('consistency_opt'):
            self.cri_consist = build_loss(train_opt['consistency_opt']).to(self.device)
            self.weight_consist = train_opt['consistency_opt'].get('weight', 10)
        else:
            self.cri_consist = None
            
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('hem_opt'):
            self.cri_hem = build_loss(train_opt['hem_opt']).to(self.device)
        else:
            self.cri_hem = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if self.cri_pix is None and self.cri_fus is None:
            raise ValueError('Both pixel and fusion losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for differnet lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name or 'raft' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
    def feed_data(self, data):
        self.lq_ir = data['lq_ir'].to(self.device)
        self.lq_vi = data['lq_vi'].to(self.device)
        if 'gt_ir' in data:
            self.gt_ir = data['gt_ir'].to(self.device)
        if 'gt_vi' in data:
            self.gt_vi = data['gt_vi'].to(self.device)
            
    def optimize_parameters(self, current_iter):
        # with torch.cuda.amp.autocast(enabled=True):
        self.results = self.net_g(self.lq_ir, self.lq_vi)
        
        self.output = self.results['fusion']
        self.rec_ir = self.results['ir']
        self.rec_vi = self.results['vi']
        
        if len(self.output.size()) == 4:
            b, c, h, w = self.output.size()
        else:
            b, t, c, h, w = self.output.size()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt_vi)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        if self.cri_fus:
            l_fusion = self.cri_fus(
                rearrange(self.output, 'b t c h w -> (b t) c h w'),
                rearrange(self.gt_ir, 'b t c h w -> (b t) c h w'),
                rearrange(self.gt_vi, 'b t c h w -> (b t) c h w')
            )
            l_fusion_total = l_fusion['loss_fusion']
            l_total += l_fusion_total
            for key, value  in l_fusion.items():
                loss_dict[key] = value
        if self.cri_fid:
            l_fid_ir = self.cri_fid(
                rearrange(self.rec_ir, 'b t c h w -> (b t) c h w'),
                rearrange(self.gt_ir, 'b t c h w -> (b t) c h w'),
                type='ir'
            )
            l_fid_vi = self.cri_fid(
                rearrange(self.rec_vi, 'b t c h w -> (b t) c h w'),
                rearrange(self.gt_vi, 'b t c h w -> (b t) c h w'),
                type='vi'
            )
            l_fid_total = l_fid_ir['ir_loss_fid'] + 1 * l_fid_vi['vi_loss_fid']
            l_total += l_fid_total
            for key, value  in l_fid_ir.items():
                loss_dict[key] = value
            for key, value  in l_fid_vi.items():
                loss_dict[key] = value
                
        if self.cri_consist:
            l_cons_ir = self.weight_consist * self.cri_consist(self.rec_ir, self.gt_ir) 
            l_cons_vi = self.weight_consist * self.cri_consist(self.rec_vi, self.gt_vi)
            l_cons_fusion = self.weight_consist * (self.cri_consist(self.output, self.gt_ir) + self.cri_consist(self.output, self.gt_vi))
            l_cons_total = l_cons_ir + l_cons_vi + l_cons_fusion
            l_total += l_cons_total
            loss_dict['l_cons_ir'] = l_cons_ir
            loss_dict['l_cons_vi'] = l_cons_vi
            loss_dict['l_cons_fusion'] = l_cons_fusion
            
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # hem loss
        if self.cri_hem:
            l_hem = self.cri_hem(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
            l_total += l_hem
            loss_dict['l_hem'] = l_hem

        # fft loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output.view(-1, c, h, w), self.gt_vi.view(-1, c, h, w))
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            
        # 反向传播和优化
        # l_total = l_total / 4.0
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        
        # 记录损失
        self.log_dict = self.reduce_loss_dict(loss_dict)
         # 更新 EMA
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        # 初始化 self.metric_results, self.metric_results_ir, self.metric_results_vi
        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {}
            self.metric_results_ir = {}
            self.metric_results_vi = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                self.metric_results_ir[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
                self.metric_results_vi[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()
            for _, tensor in self.metric_results_ir.items():
                tensor.zero_()
            for _, tensor in self.metric_results_vi.items():
                tensor.zero_()

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        self.time_list = []
        counter = 0
        self.net_g.eval()
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']
            # if not '0111_1753' in folder:
            #     continue
            names = val_data['name']
            # compute outputs
            val_data['lq_vi'].unsqueeze_(0)  # 确保有batch维度
            val_data['lq_ir'].unsqueeze_(0)
            val_data['gt_ir'].unsqueeze_(0)
            val_data['gt_vi'].unsqueeze_(0)
            # print(torch.max(val_data['lq_vi']), torch.min(val_data['lq_vi']))
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                visuals['rec_ir'] = visuals['rec_ir'].unsqueeze(1)
                visuals['rec_vi'] = visuals['rec_vi'].unsqueeze(1)
                visuals['lq_ir'] = visuals['lq_ir'].unsqueeze(1)
                visuals['lq_vi'] = visuals['lq_vi'].unsqueeze(1)
                if 'gt_ir' in visuals:
                    visuals['gt_ir'] = visuals['gt_ir'].unsqueeze(1)
                if 'gt_vi' in visuals:
                    visuals['gt_vi'] = visuals['gt_vi'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    # fusion results
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result], min_max=(torch.min(result), torch.max(result)))  # uint8, bgr
                    # print(torch.min(result), torch.max(result))
                    # Restoration infrared results
                    rec_ir = visuals['rec_ir'][0, idx, :, :, :]
                    rec_ir_img = tensor2img([rec_ir])  # uint8, bgr
                    # Restoration visible results
                    rec_vi = visuals['rec_vi'][0, idx, :, :, :]
                    rec_vi_img = tensor2img([rec_vi])  # uint8, bgr
                    # source visible results
                    lq_vi = visuals['lq_vi'][0, idx, :, :, :]
                    lq_vi_img = tensor2img([lq_vi])  # uint8, bgr
                    # source infrared results
                    lq_ir = visuals['lq_ir'][0, idx, :, :, :]
                    lq_ir_img = tensor2img([lq_ir])  # uint8, bgr

                    if 'gt_ir' in visuals:
                        gt_ir = visuals['gt_ir'][0, idx, :, :, :]
                        gt_ir_img = tensor2img([gt_ir])  # uint8, bgr

                    if 'gt_vi' in visuals:
                        gt_vi = visuals['gt_vi'][0, idx, :, :, :]
                        gt_vi_img = tensor2img([gt_vi])  # uint8, bgr

                    if save_img:
                        only_fusion_flag = False
                        if self.opt['is_train']:
                            img_path = osp.join(
                                self.opt["path"]["visualization"], 
                                'Fusion',
                                "{}".format(current_iter),
                                folder.split('-')[0],
                                names[idx],
                            )
                            ir_path = osp.join(
                                self.opt["path"]["visualization"], 
                                'IR',
                                "{}".format(current_iter),
                                folder.split('-')[0],
                                names[idx],
                            )
                            vi_path = osp.join(
                                self.opt["path"]["visualization"], 
                                'VI',
                                "{}".format(current_iter),
                                folder.split('-')[0],
                                names[idx],
                            )                            
                        else:
                            # 这里根据后面具体的测试需要再进行修改
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Fusion', 
                                folder.split('-')[0], names[idx])
                                ir_path = osp.join(self.opt['path']['visualization'], dataset_name, 'IR', 
                                folder.split('-')[0], names[idx])
                                vi_path = osp.join(self.opt['path']['visualization'], dataset_name, 'VI', 
                                folder.split('-')[0], names[idx])
                                input_ir_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Source_IR', 
                                folder.split('-')[0], names[idx])
                                input_vi_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Source_VI', 
                                folder.split('-')[0], names[idx])
                            else:  # others
                                if self.opt['path']['only_fusion']:
                                    only_fusion_flag = True
                                    img_path = osp.join(self.opt['path']['save_folder'],  'BiCAM',  folder.split('-')[0], names[idx])
                                else:
                                    img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Fusion', 
                                    folder.split('-')[0], names[idx])
                                    ir_path = osp.join(self.opt['path']['visualization'], dataset_name, 'IR', 
                                    folder.split('-')[0], names[idx])
                                    vi_path = osp.join(self.opt['path']['visualization'], dataset_name, 'VI', 
                                    folder.split('-')[0], names[idx])
                                    input_ir_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Source_IR', 
                                    folder.split('-')[0], names[idx])
                                    input_vi_path = osp.join(self.opt['path']['visualization'], dataset_name, 'Source_VI',
                                    folder.split('-')[0], names[idx])
                                    imwrite(lq_ir_img, input_ir_path)
                                    imwrite(lq_vi_img, input_vi_path)
                            # image name only for REDS dataset
                        if not only_fusion_flag:                            
                            imwrite(rec_ir_img, ir_path)
                            imwrite(rec_vi_img, vi_path)
                        imwrite(result_img, img_path)
                        counter += 1
                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            metric_data_ir = dict(img1=result_img, img2=gt_ir_img)
                            metric_data_vi = dict(img1=result_img, img2=gt_vi_img)
                            metric_data_ir_rec = dict(img1=rec_ir_img, img2=gt_ir_img)
                            metric_data_vi_rec = dict(img1=rec_vi_img, img2=gt_vi_img)
                            result = 0.5 * calculate_metric(metric_data_ir, opt_) + 0.5 * calculate_metric(metric_data_vi, opt_)
                            result_ir = calculate_metric(metric_data_ir_rec, opt_)
                            result_vi = calculate_metric(metric_data_vi_rec, opt_)
                            self.metric_results[folder][idx, metric_idx] += result
                            self.metric_results_ir[folder][idx, metric_idx] += result_ir
                            self.metric_results_vi[folder][idx, metric_idx] += result_vi                
                print(img_path)

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                for _, tensor in self.metric_results_ir.items():
                    dist.reduce(tensor, 0)
                for _, tensor in self.metric_results_vi.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metric_results=self.metric_results, type='fusion')
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metric_results=self.metric_results_ir, type='ir_rec')
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metric_results=self.metric_results_vi, type='vi_rec')
        print("Counter:{}, average test time: {:.3f}s".format(counter, sum(self.time_list) / counter))
        self.net_g.train()


    def test(self):
        downsample_scale = 4
        b, n, c, h, w = self.lq_ir.size()
        mod_pad_h, mod_pad_w = 0, 0
        if h % downsample_scale != 0:
            mod_pad_h = downsample_scale - h % downsample_scale
        if w % downsample_scale != 0:
            mod_pad_w = downsample_scale - w % downsample_scale
        
        # 填充操作
        if mod_pad_h > 0 or mod_pad_w > 0:
            print('Padding lq_ir and lq_vi')
            self.lq_ir = F.pad(rearrange(self.lq_ir, 'b t c h w -> (b t) c h w'), (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            self.lq_vi = F.pad(rearrange(self.lq_vi, 'b t c h w -> (b t) c h w'), (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            self.lq_ir = rearrange(self.lq_ir, '(b t) c h w -> b t c h w', t=n).contiguous()
            self.lq_vi = rearrange(self.lq_vi, '(b t) c h w -> b t c h w', t=n).contiguous()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            with torch.no_grad():
                output_sum_f = 0.0
                output_sum_ir = 0.0
                output_sum_vi = 0.0
                
                for k in range(0, 8):
                    if k < 4:
                        start_time = time()
                        results = self.net_g(self.lq_ir.rot90(k, [3, 4]).cuda(), self.lq_vi.rot90(k, [3, 4]).cuda())
                        end_time = time()
                        self.time_list.append(end_time - start_time)
                        output = results['fusion'].rot90(-k, [3, 4]).detach().cpu()
                        rec_ir = results['ir'].rot90(-k, [3, 4]).detach().cpu()
                        rec_vi = results['vi'].rot90(-k, [3, 4]).detach().cpu()
                    else:
                        start_time = time()
                        results = self.net_g(self.lq_ir.rot90(k, [3, 4]).flip(1).cuda(), self.lq_vi.rot90(k, [3, 4]).flip(1).cuda())
                        end_time = time()
                        self.time_list.append(end_time - start_time)
                        output = results['fusion'].flip(1).rot90(-k, [3, 4]).detach().cpu()
                        rec_ir = results['ir'].flip(1).rot90(-k, [3, 4]).detach().cpu()
                        rec_vi = results['vi'].flip(1).rot90(-k, [3, 4]).detach().cpu()

                    output_sum_f += output
                    output_sum_ir += rec_ir
                    output_sum_vi += rec_vi
                    
                    # 清理不必要的计算图
                    del results, output, rec_ir, rec_vi
                    torch.cuda.empty_cache()

                self.output = 0.125 * output_sum_f
                self.rec_ir = 0.125 * output_sum_ir
                self.rec_vi = 0.125 * output_sum_vi

        else:
            with torch.no_grad():
                start_time = time()
                self.results = self.net_g(self.lq_ir, self.lq_vi)
                end_time = time()
                self.time_list.append(end_time - start_time)
                
                self.output = self.results['fusion']
                self.rec_ir = self.results['ir']
                self.rec_vi = self.results['vi']

        # 截取有效区域
        self.output = self.output[:, :, :, 0:h, 0:w]
        self.rec_ir = self.rec_ir[:, :, :, 0:h, 0:w]
        self.rec_vi = self.rec_vi[:, :, :, 0:h, 0:w]
        
        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]  # 只取中心帧
            self.rec_ir = self.rec_ir[:, n // 2, :, :, :]  # 只取中心帧
            self.rec_vi = self.rec_vi[:, n // 2, :, :, :]  # 只取中心帧
        
        
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_ir'] = self.lq_ir.detach().cpu()
        out_dict['lq_vi'] = self.lq_vi.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['rec_ir'] = self.rec_ir.detach().cpu()
        out_dict['rec_vi'] = self.rec_vi.detach().cpu()                
        # tentative for out of GPU memory
        del self.lq_ir
        del self.lq_vi
        del self.output
        del self.rec_ir
        del self.rec_vi
        del self.results
                
        if hasattr(self, 'gt_ir'):
            out_dict['gt_ir'] = self.gt_ir.detach().cpu()
            del self.gt_ir
        if hasattr(self, 'gt_vi'):
            out_dict['gt_vi'] = self.gt_vi.detach().cpu()
            del self.gt_vi            
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger=None, metric_results=None, type='Fusion'):
        # average all frames for each sub-folder
        if metric_results is None:
            metric_results_avg = {
                folder: torch.mean(tensor, dim=0).cpu()
                for (folder, tensor) in self.metric_results.items()
            }
        else:
            metric_results_avg = {
                folder: torch.mean(tensor, dim=0).cpu()
                for (folder, tensor) in metric_results.items()
            }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)

        log_str = f'Validation {type} {dataset_name} {current_iter}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{type}/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{type}/{metric}/{folder}', tensor[metric_idx].item(), current_iter)

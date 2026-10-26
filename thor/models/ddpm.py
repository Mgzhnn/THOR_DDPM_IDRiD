from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thor.models.simplex_noise import generate_noise
from thor.models.unet import DiffusionModelUNet
from thor.models.schedulers.ddpm import DDPMScheduler
from thor.models.schedulers.ddim import DDIMScheduler

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import copy

from tqdm import tqdm
has_tqdm = True
from skimage import exposure
from scipy.ndimage import gaussian_filter, grey_closing, grey_dilation

from scipy import stats

import lpips


class DDPM(nn.Module):

    def __init__(self, spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 num_channels=(128, 256, 256),
                 attention_levels=(False, True, True),
                 num_res_blocks=1,
                 num_head_channels=256,
                 train_scheduler="ddpm",
                 inference_scheduler="ddpm",
                 inference_steps=1000,
                 noise_level_recon=300,
                 noise_type="gaussian",
                 prediction_type="epsilon",
                 threshold_low=1,
                 threshold_high=10000,
                 inference_type='ano',
                 t_harmonization=[700, 600, 500, 400, 300, 150, 50], 
                 t_visualization=[700,650,600,550,500,450,400,350,300,250,200,150,100,50,0],
                 image_path="",):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.inference_type = inference_type
        self.t_harmonization = t_harmonization
        self.t_visualization = t_visualization
        self.noise_level_recon = noise_level_recon
        self.prediction_type = prediction_type
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.image_path = image_path
        self.img_ct = 0

        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        # LPIPS for perceptual anomaly maps
        self.l_pips_sq = lpips.LPIPS(pretrained=True, pnet_rand=False, net='squeeze', eval_mode=True, spatial=True, lpips=True).to(self.device)

        # set up scheduler and timesteps
        if train_scheduler == "ddim":
            print('****** DIFFUSION: Using DDIM Scheduler ******')
            self.train_scheduler = DDIMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        elif train_scheduler == 'ddpm':
            print('****** DIFFUSION: Using DDPM Scheduler ******')
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            raise NotImplementedError(f"{train_scheduler} does is not implemented for {self.__class__}")

        if inference_scheduler == "ddim":
            print('****** DIFFUSION: Using DDIM Scheduler ******')
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            print('****** DIFFUSION: Using DDPM Scheduler ******')
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)

        self.inference_scheduler.set_timesteps(inference_steps)

        ts = np.array(self.inference_scheduler.timesteps, dtype=np.int64)

        def _nearest(v: int) -> int:
            return int(ts[np.argmin(np.abs(ts - int(v)))])

        self.t_harmonization = [_nearest(v) for v in self.t_harmonization]
        self.t_visualization = [_nearest(v) for v in self.t_visualization]

    def forward(self, inputs, noise=None, timesteps=None, condition=None):
        if noise is None:
            noise = torch.randn_like(inputs)
        if timesteps is None:
            timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps,
                                      (inputs.shape[0],), device=inputs.device).long()

        noisy_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps)
        return self.unet(x=noisy_image, timesteps=timesteps, context=condition)
    
    
    def get_anomaly_mask(self, x, x_rec, hist_eq=False, retPerLayer=False):
        x_res = self.compute_residual(x, x_rec, hist_eq=hist_eq)
        lpips_mask = self.get_saliency(x, x_rec, retPerLayer=retPerLayer).clip(0,1)

        combined_mask_np = lpips_mask * x_res 
        combined_mask_np2 = (lpips_mask * x_res) 
        
        # Keep results on the inputs' device (self.device ignores the configured device)
        combined_mask = torch.Tensor(combined_mask_np).to(x.device)
        combined_mask2 = torch.Tensor(combined_mask_np2).to(x.device)

        combined_mask = (combined_mask / (torch.max(combined_mask) + 1e-8)).clip(0,1) 
        return combined_mask, combined_mask2, torch.Tensor(x_res).to(x.device)

    
    def get_region_anomaly_mask(self, ano_map, kernel_size=13):
        final_anomaly_map = (grey_closing(ano_map, size=(1,1,kernel_size,kernel_size), mode='nearest'))
        final_anomaly_map = (grey_dilation(final_anomaly_map, size=(1,1,kernel_size,kernel_size), mode='nearest') + ano_map)/2
        final_anomaly_map = final_anomaly_map.clip(0,1)
        return final_anomaly_map

    def get_thor_anomaly(self, inputs):
        x_rec, z_dict = self.sample_from_image_interpol(inputs, noise_level=self.noise_level_recon, save_intermediates=True, intermediate_steps=self.intermediate_steps, t_harmonization=self.t_harmonization, t_visualization=self.t_visualization)
        
        # Model space is [-1, 1] (the trainer maps [0, 1] inputs with x * 2 - 1); callers map back to [0, 1].
        x_rec = torch.clamp(x_rec, -1, 1)

        np_res = [inter.cpu().detach().numpy() for inter in z_dict['inter_res']]
        x_rec_refined = x_rec 
        self.img_ct += 1 
        
        x_res =  np_res[-1].clip(0, 0.999)
        anomaly_maps = x_res 
        masked_input = x_res
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)

        return anomaly_maps, anomaly_scores, {'x_rec': x_rec_refined, 'mask': masked_input, 'x_res': x_res,
                                            'x_rec_orig': x_rec}
    

    def get_anomaly(self, inputs, noise_level=250):
        if self.inference_type == 'thor':
             # 안전 장치: intermediate_steps가 없을 경우 대비
            if not hasattr(self, 'intermediate_steps'):
                self.intermediate_steps = 100
            return self.get_thor_anomaly(inputs)
            
        x_rec, _ = self.sample_from_image(inputs, self.noise_level_recon)
        x_rec_ = x_rec.cpu().detach().numpy()
        x_ = inputs.cpu().detach().numpy()
        anomaly_maps = np.abs(x_ - x_rec_)
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        return anomaly_maps, anomaly_scores, {'x_rec': x_rec}

    def compute_residual(self, x, x_rec, hist_eq=False):
        if hist_eq:
            x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
            x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
            x_res = np.abs(x_rec_rescale - x_rescale)
        else:
            x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

        return x_res

    def lpips_loss(self, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        """
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        
        # Inputs are already in [-1, 1], the range LPIPS expects, so normalize=False.
        # normalize=True would map [0, 1] -> [-1, 1] and turn these inputs into [-3, 1].
        if anomaly_img.shape[1] == 1:
            anomaly_img = anomaly_img.repeat(1, 3, 1, 1)
        if ph_img.shape[1] == 1:
            ph_img = ph_img.repeat(1, 3, 1, 1)

        loss_lpips = self.l_pips_sq(anomaly_img, ph_img, normalize=False, retPerLayer=retPerLayer)
        if retPerLayer:
            loss_lpips = loss_lpips[1][0]
        return loss_lpips.cpu().detach().numpy()

    def get_saliency(self, x, x_rec, retPerLayer=False):
        saliency = self.lpips_loss(x, x_rec, retPerLayer)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        noise_level: int | None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        image = input_noise
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        for t in progress_bar:
            model_output = self.unet(
                image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
            )
            image, orig_image = self.inference_scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(orig_image)
        if save_intermediates:
            return image, intermediates
        else:
            return image, None

    @torch.no_grad()
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        noise = generate_noise(
            self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],),
                       noise_level, device=inputs.device).long()
        noised_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image, intermediates = self.sample(input_noise=noised_image, noise_level=noise_level, save_intermediates=save_intermediates,
                            intermediate_steps=intermediate_steps, conditioning=conditioning, verbose=verbose)
        return image, {'z': intermediates}
    
    @torch.no_grad()
    def sample_from_image_interpol(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        t_harmonization: [int] | None = [700, 600, 500, 400, 300, 150, 50],
        t_visualization: [int] | None = [700,650,600,550,500,450,400,350,300,250,200,150,100,50,0],
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:

        loss_idx = 0
        do_hmatching = False

        noise = generate_noise(
            self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],),
                       noise_level, device=inputs.device).long()
        input_noise = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image = input_noise
        
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        intermediates_res = []
        intermediates_res_mix = []
        
        for t in progress_bar:
            model_output = self.unet(image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning)
            image, orig_image = self.inference_scheduler.step(model_output, t, image)

            if save_intermediates and (t in t_harmonization or t in t_visualization):
                intermediates.append(orig_image)
                
                res_thor = self.get_anomaly_mask(copy.deepcopy(orig_image), copy.deepcopy(inputs), hist_eq=do_hmatching)[loss_idx]  
                res = res_thor
                resnp = res.cpu().detach().numpy()
                res_mix = resnp

                region_anomaly_map = self.get_region_anomaly_mask(res_mix) 
                res = torch.Tensor(region_anomaly_map).to(inputs.device)
                res = ((res)).clip(0,1)
                intermediates_res.append(res_mix)
                intermediates_res_mix.append(((region_anomaly_map)).clip(0,1))
            
            if t in t_harmonization:
                image_0 = res * orig_image + (1-res) * inputs   
                image_0 = torch.clamp(image_0, -1, 1) # [수정] clamp 범위 조정
                image = self.train_scheduler.add_noise(original_samples=image_0, noise=noise, timesteps=t)

        self.img_ct += 1 
        image_refined = image
        
        hmean = stats.hmean(np.stack(intermediates_res[:]), axis=0)
        hmean_mix = stats.hmean(np.stack(intermediates_res_mix[:]), axis=0)
        intermediates_res.append(hmean)
        intermediates_res_mix.append(hmean_mix)

        intermediates_res = [torch.Tensor(inter).to(inputs.device) for inter in intermediates_res]
        intermediates_res_mix = [torch.Tensor(inter).to(inputs.device) for inter in intermediates_res_mix]

        if save_intermediates:
            return image_refined, {'z': intermediates, 'inter_gt': intermediates, 'inter_res': intermediates_res, 'inter_ddpm': intermediates, 'inter_res_ddpm': intermediates_res, 
                           'inter_res_mix': intermediates_res_mix}
        else:
            return image_refined, {'z': None}

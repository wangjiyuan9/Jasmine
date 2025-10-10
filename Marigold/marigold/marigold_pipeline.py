from typing import Dict, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from my_utils import disp_to_depth
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

# add
import random


# add
# Surface Normals Ensamble from the GeoWizard github repository (https://github.com/fuxiao0719/GeoWizard)
def ensemble_normals(input_images: torch.Tensor):
    normal_preds = input_images
    bsz, d, h, w = normal_preds.shape
    normal_preds = normal_preds / (torch.norm(normal_preds, p=2, dim=1).unsqueeze(1) + 1e-5)
    phi = torch.atan2(normal_preds[:, 1, :, :], normal_preds[:, 0, :, :]).mean(dim=0)
    theta = torch.atan2(torch.norm(normal_preds[:, :2, :, :], p=2, dim=1), normal_preds[:, 2, :, :]).mean(dim=0)
    normal_pred = torch.zeros((d, h, w)).to(normal_preds)
    normal_pred[0, :, :] = torch.sin(theta) * torch.cos(phi)
    normal_pred[1, :, :] = torch.sin(theta) * torch.sin(phi)
    normal_pred[2, :, :] = torch.cos(theta)
    angle_error = torch.acos(torch.clip(torch.cosine_similarity(normal_pred[None], normal_preds, dim=1), -0.999, 0.999))
    normal_idx = torch.argmin(angle_error.reshape(bsz, -1).sum(-1))
    return normal_preds[normal_idx], None


# add
# Pyramid nosie from 
#   https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2?s=31
def pyramid_noise_like(x, discount=0.9):
    b, c, w, h = x.shape
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random() * 2 + 2
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount ** i
        if w == 1 or h == 1:
            break
    return noise / noise.std()


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_tc: torch.Tensor


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
            self,
            unet: UNet2DConditionModel,
            vae: AutoencoderKL,
            scheduler: Union[DDIMScheduler, DDPMScheduler, LCMScheduler],
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            args: Dict = None,
            small_diffusion=None,
            inter_gru=None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.opts = args
        self.empty_text_embed = None
        if args.use_diffusion:
            self.small_diffusion = small_diffusion
        if args.use_gru:
            self.inter_gru = inter_gru

    @torch.no_grad()
    def __call__(
            self,
            input_image: torch.Tensor,
            denoising_steps: int = 10,
            ensemble_size: int = 10,
            processing_res: int = 768,
            match_input_res: str = "Orginal",
            resample_method: str = "bilinear",
            batch_size: int = 0,
            color_map: str = "Spectral",
            show_progress_bar: bool = False,
            ensemble_kwargs: Dict = None,
            noise="gaussian",
            input_depth: torch.Tensor = None,

    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`str` defaults to `Orginal`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
            noise (`str`, *optional*, defaults to `gaussian`):
                Type of noise to be used for the initial depth map.
                Can be one of `gaussian`, `pyramid`, `zeros`.
            normals (`bool`, *optional*, defaults to `False`):
                If `True`, the pipeline will predict surface normals instead of depth maps.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
            - **normal_np** (`np.ndarray`) Predicted normal map, with normal vectors in the range of [-1, 1]
            - **normal_colored** (`PIL.Image.Image`) Colorized normal map
        """

        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        input_size = input_image.shape
        assert (
                4 == input_image.dim() and 3 == input_size[1]
        ), f"Wrong input shape {input_size}, expected [bs, rgb, H, W]"

        # Normalize rgb values
        rgb_norm: torch.Tensor = input_image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # depth_norm = input_depth.repeat(1, 3, 1, 1)
        pred_raw = self.single_infer(
            rgb_in=rgb_norm,
            # depth_in=depth_norm,
            num_inference_steps=denoising_steps,
        )
        # pred_ls.append(pred_raw.detach())
        pred = pred_raw

        # Resize back to original resolution
        if match_input_res == "Orginal":
            pred = resize(pred.unsqueeze(0), (input_size[-2], input_size[-1]), interpolation=resample_method, antialias=True, ).squeeze()
        elif match_input_res == "Args":
            pred = resize(pred, (self.opts.height, self.opts.width), interpolation=resample_method, antialias=True, ).squeeze()
        else:
            pass

        # pred = pred.clip(0, 1).squeeze()
        if pred.dim() == 4:
            pred = pred.squeeze()
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        assert len(pred.shape) == 3, f"Wrong shape {pred.shape}, expected [bs, H, W]"

        return MarigoldDepthOutput(depth_tc=pred )

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(prompt, padding="do_not_pad", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
            self,
            rgb_in: torch.Tensor,
            # depth_in: torch.Tensor,
            num_inference_steps: int,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.
        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        # depth_in = depth_in.to(device)
        task_emb = torch.tensor([0., 1.]).float().unsqueeze(0).repeat(1, 1).to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(rgb_in.shape[0], 1)
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
        # depth_latent = self.encode_rgb(depth_in)
        # add
        # Initial prediction
        latent_shape = rgb_latent.shape
        latent = torch.zeros(latent_shape, device=device, dtype=self.dtype, )

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        iterable = enumerate(timesteps)
        for i, t in iterable:
            # unet_input = torch.cat([rgb_latent, depth_latent, latent], dim=1) HERE
            unet_input = torch.cat([rgb_latent, latent], dim=1)

            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed, class_labels=task_emb
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
            latent = scheduler_step.prev_sample
            # add
            if i == num_inference_steps - 1:
                latent = scheduler_step.pred_original_sample

        depth, disp = self.decode_depth(latent, rgb_in, rgb_latent)
        depth = (depth + 1.0) / 2.0
        # depth = torch.clamp(depth, -1, 1) / 2 + 0.5
        if disp is None and self.opts.pred_mode == "disp":
            depth, _ = disp_to_depth(depth)
        if disp is not None and self.opts.pred_mode == "disp":
            depth, _ = disp_to_depth(disp)
        elif disp is not None and self.opts.pred_mode == "depth":
            depth = disp
        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor, rgb_in=None, rgb_encoded=None) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True) if not self.opts.ground else stacked[:, 0:1]
        disp = None
        if self.opts.use_gru:
            outputs = self.inter_gru(rgb_encoded, stacked)
            depth_mean = outputs['depth_list'][-1] * 2. - 1

        return depth_mean, disp

    # add
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.

        Args:
            normal_latent (`torch.Tensor`):
                normal latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        normal = self.vae.decoder(z)
        return normal
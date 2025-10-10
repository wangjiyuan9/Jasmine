# coding=utf-8
# The test version of the trains.py for Jasmine based. We temporarily only release the test code and the training code will be released later. The code is based on the original E2E-FT codebase.
# Author: Jiyuan Wang
# Created: 2025-10-06
# Origin used for paper: https://arxiv.org/abs/2503.15905
# Hope you can cite our paper if you use the code for your research.
import logging
import math
import shutil
import accelerate
import datasets
import torch.utils.checkpoint
import transformers
import networks
import warnings
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from torch.optim.lr_scheduler import LambdaLR
import sys
from dataloaders import *
from util.loss import *
from util.unet_prep import replace_unet_conv_in
from util.lr_scheduler import IterExponential
from options import MonodepthOptions
from my_utils import *
from torch.utils.data import DataLoader
from Evaluate import *
import os
from datetime import timedelta

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")
debugger = get_logger("debugger", log_level="DEBUG")

def main():
    options = MonodepthOptions()
    args = options.parse()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    mono_tracker = MyCustomTracker(run_name="train", logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=mono_tracker,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(42)

    # Save training arguments in a .txt file
    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)
    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")

    # Load model components
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, timestep_spacing=args.timestep_spacing, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)
    pose_encoder = networks.ResnetEncoder(18, True, num_input_images=2)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    inter_gru = None

    if args.use_gru:
        inter_gru = networks.GRUFrameWork(args, vae)
        if args.pretrain_gru:
            path = args.pretrain_gru_path
            state_dict = torch.load(path, map_location='cpu')
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            inter_gru.load_state_dict(state_dict)
            inter_gru.requires_grad_(False)
            
    if args.pretrain_pose:#MUST TRUE IN PHASE 2
        path = args.pretrain_pose_path
        state_dict = torch.load(path, map_location='cpu')
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pose_decoder.load_state_dict(state_dict)
        pose_decoder.requires_grad_(False)
        path = args.pretrain_pose_encoder_path
        state_dict = torch.load(path, map_location='cpu')
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pose_encoder.load_state_dict(state_dict)
        pose_encoder.requires_grad_(False)

    if args.noise_type is not None:
        if unet.config['in_channels'] != 8:
            replace_unet_conv_in(unet, repeat=2)
            logger.info("Unet conv_in layer is replaced for RGB-depth-noise conditioning")

    # Freeze or set model components to training mode
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(True)
    unet.train()
    logger.info("Unfreezing unet")

    if not args.supervised and not args.define_pose:
        pose_encoder.train()
        pose_decoder.train()
        if args.use_gru:
            inter_gru.train()

    unet.enable_xformers_memory_efficient_attention()
    # unet.enable_gradient_checkpointing() #can be enabled if no high vram
    vae.enable_xformers_memory_efficient_attention()
    vae.enable_gradient_checkpointing()

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if model.__class__.__name__ == "UNet2DConditionModel":
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif model.__class__.__name__ == "AutoencoderKL":
                        model.save_pretrained(os.path.join(output_dir, "vae"))
                    elif isinstance(model, networks.PoseDecoder):
                        to_save = model.state_dict()
                        torch.save(to_save, os.path.join(output_dir, "pose.pth"))
                    elif isinstance(model, networks.ResnetEncoder):
                        to_save = model.state_dict()
                        torch.save(to_save, os.path.join(output_dir, "pose_encoder.pth"))
                    elif isinstance(model, networks.GRUFrameWork):
                        to_save = model.state_dict()
                        torch.save(to_save, os.path.join(output_dir, "gru.pth"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                elif isinstance(model, AutoencoderKL):
                    load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                elif isinstance(model, networks.PoseDecoder):
                    load_model = torch.load(os.path.join(input_dir, "pose.pth"), map_location='cuda:0')
                    model.load_state_dict(load_model)
                elif isinstance(model, networks.ResnetEncoder):
                    load_model = torch.load(os.path.join(input_dir, "pose_encoder.pth"), map_location='cuda:0')
                    model.load_state_dict(load_model)
                elif isinstance(model, networks.GRUFrameWork):
                    load_model = torch.load(os.path.join(input_dir, "gru.pth"), map_location='cuda:0')
                    model.load_state_dict(load_model)
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Optimizer
    scrach_learnable = []
    pretrained_learnable = list(unet.parameters())+ list(pose_encoder.parameters()) + list(pose_decoder.parameters())
    if args.use_gru:
        pretrained_learnable += list(inter_gru.parameters())
    params = [{"params": pretrained_learnable, "lr": args.learning_rate}, {"params": scrach_learnable, "lr": 5e-5}]
    optimizer = torch.optim.AdamW(params)

    # Learning rate scheduler
    lr_func = IterExponential(total_iter_length=args.lr_total_iter_length * accelerator.num_processes, final_ratio=0.01, warmup_steps=args.lr_exp_warmup_steps * accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    train_filenames, val_filenames,stereo_filenames,  cityscapes_filenames, args = prepare_dataset(args)

    if args.debug > 0:
        val_filenames = val_filenames[-40:] if args.debug >= 0.5 else val_filenames
        args.dataloader_num_workers = 0 if args.debug >= 2 else args.dataloader_num_workers
        args.max_train_steps = min(20, args.max_train_steps) if (args.debug >= 1 and not args.resume_from_checkpoint) else args.max_train_steps
        args.checkpointing_steps = 1 if args.debug >= 2 else 10

    train_dataset = KITTIRAWDataset(args, train_filenames, [], is_train=True)#hypersim_filenames
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, True, num_workers=args.dataloader_num_workers, pin_memory=True)
    # Only use val_dataloader on the main process (main GPU)
    if accelerator.is_main_process:
        if "eigen" in args.eval_split:
            val_dataset = KITTIRAWDataset(args, val_filenames, is_train=False)
        elif "city" in args.eval_split:
            val_dataset = CityscapesDataset(args, cityscapes_filenames, is_train=False)
        elif "stereo" in args.eval_split:
            val_dataset = DrivingStereoDataset(args, stereo_filenames, is_train=False)
        val_dataloader = DataLoader(val_dataset, 4, False, num_workers=0 if args.debug >= 1 else 4, pin_memory=True)

    unet, optimizer, train_dataloader, lr_scheduler, pose_encoder, pose_decoder, vae, inter_gru = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler, pose_encoder, pose_decoder, vae, inter_gru
    )

    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Calculate number of training steps and epochs
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    
    # Resume training from checkpoint, must set TRUE if only_test
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = args.resume_from_checkpoint
        if path is None or os.path.exists(path) is False:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.resume_from_checkpoint == "latest":
                path = os.path.join(args.output_dir, path)
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
            accelerator.load_state(path,load_only=["model"])

        

    # Init task
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process, )
    best_abs, best_step = 0.2, 0
    gt_depths, improved_gt_depth = prepare_gt_depths(args)

    if args.only_test:
        print("Only test mode, with {} val samples".format(len(val_dataset)))
        validation(args, accelerator.device, unet, val_dataloader, val_dataset, vae, text_encoder, val_scheduler, tokenizer, global_step, logger, accelerator, mode='test', gt_depths=gt_depths, improvedGT=improved_gt_depth, inter_gru=inter_gru)
        return 0


if __name__ == "__main__":
    main()

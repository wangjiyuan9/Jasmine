import argparse
import os
from collections import namedtuple

defalut_dif_scale = True


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepth options")
        # region MY options
        self.parser.add_argument("--debug", type=float, default=0, help="Debug mode.")
        self.parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=1)
        self.parser.add_argument("--use_gds_loss", '--ugl', type=int, default=0, help="if set,use Ground-contacting-prior Disparity Smoothness Loss to train the model")
        self.parser.add_argument("--only_test", '--ot', action="store_true", help="if set,only test the model")
        self.parser.add_argument("--not_align_with_median", '--nawm', default=False, action="store_true")
        self.parser.add_argument("--save_depth", '--sd', action="store_true", help="if set,save the depth")
        self.parser.add_argument('--use_edge_loss', '--uel', action="store_true", help="if set, use edge loss between the condition and disp")
        self.parser.add_argument('--vit_model', type=str, default='google/vit-base-patch16-224')
        self.parser.add_argument("--easy_vis", action="store_true")
        self.parser.add_argument("--vis_path", type=str, help="the path to save the vis result")
        self.parser.add_argument("--use_gru", '--ug', action="store_true", help="if set,use gru to train the model")
        self.parser.add_argument("--pretrain_gru", '--pg', action="store_true", help="if set,pretrain the gru")
        self.parser.add_argument("--pretrain_pose", '--ptp', action="store_true", help="if set,pretrain the pose")
        self.parser.add_argument("--link_mode", '--lm', type=str, help="the mode to link the depth", default="mean")
        self.parser.add_argument("--use_perceptual_loss", '--upl', action="store_true", help="if set,use perceptual loss")
        self.parser.add_argument("--depth_latent_weight", '--dlw', type=float, help="the weight for the depth latent", default=0.2)
        # endregion MY options

        # region Self-supervised learning
        self.parser.add_argument("--data_path","--dpath", type=str, help="path to the training data", default="YOUR DATA PATH")
        self.parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in", default="mdp")
        self.parser.add_argument("--split", type=str, help="which training split to use", default="eigen_zhou")
        self.parser.add_argument("--height", type=int, help="input image height", default=320)
        self.parser.add_argument("--width", type=int, help="input image width", default=1024)
        self.parser.add_argument("--smooth_weight", type=float, help="disparity smoothness weight", default=8e-3)
        self.parser.add_argument("--novel_frame_ids", nargs="+", type=int, help="frames to load", default=[-1, 1])
        self.parser.add_argument("--img_ext", type=str, choices=[".png", ".jpg"], default=".png", help="the image extension")
        self.parser.add_argument("--supervised", help="if set,use supervised way to train the model", action="store_true")
        self.parser.add_argument("--define_pose", '--dp', action="store_true", help="if set,just load the pretrained pose to train the depth")
        self.parser.add_argument("--define_disp", '--dd', action="store_true", help="if set,just load the pretrained disp as teacher to train the network")
        self.parser.add_argument("--use_sky_loss", '--usl', action="store_true", help="if set,use sky loss", default=False)
        self.parser.add_argument("--ground", action="store_true", help="if set,use ground information to train the model")
        self.parser.add_argument("--attn_precent", type=float, help="the weight for the ground loss", default=0.8)
        self.parser.add_argument("--enhance_ground", '--eg', action="store_true", help="if set,use mask to enhance the ground l1loss")
        self.parser.add_argument("--scale_consistency", '--sc', action="store_true", help="if set,use scale consist loss")
        self.parser.add_argument("--dif_scale", '--ds', action="store_true", help="use LR to train the pose net but use MR or HR to train the depth net", default=defalut_dif_scale)
        self.parser.add_argument("--log_frequency", type=int, help="number of batches between each tensorboard log", default=500)
        self.parser.add_argument("--enhance_teacher", '--et', action="store_true", help="if set, use mask to enhance the teacher gt")
        self.parser.add_argument("--teacher_weight", type=float, help="teacher_weight", default=0.2)
        self.parser.add_argument('--teacher_loss', '--tl', action="store_true", help="if set, use teacher loss")
        self.parser.add_argument("--use_diffusion", "--ud", action="store_true", help="if set,use diffusion")
        self.parser.add_argument("--pretrain_diffusion", "--pd", action="store_true", help="if set,pretrain the diffusion")
        self.parser.add_argument("--use_teacher", "--ut", action="store_true", help="if set,use teacher model result to distill the student model result")
        self.parser.add_argument("--loss", type=str, help="the loss you want to use", default="berhu")
        self.parser.add_argument("--pred_mode", type=str, help="the mode you want to predict", default="depth")
        self.parser.add_argument("--disparity", '--disp', action="store_true", help="if set,use disparity to train the model")
        self.parser.add_argument("--only_ssim", '--oss', action="store_true", help="if set,only use ssim to train the model")
        self.parser.add_argument("--prob_of_mari", '--pom', type=float, help="the probability of using the mari loss", default=0)
        self.parser.add_argument("--sup_in_latent", '--sil', action="store_true", help="if set,use the supervised way to train the latent")
        self.parser.add_argument("--pretrain_gru_path", type=str, help="the path to the pretrained gru model", default="YOUR PRETRAINED GRU MODEL PATH")
        self.parser.add_argument("--pretrain_pose_path", type=str, help="the path to the pretrained pose model", default="YOUR PRETRAINED POSE MODEL PATH")
        self.parser.add_argument("--pretrain_pose_encoder_path", type=str, help="the path to the pretrained pose encoder model", default="YOUR PRETRAINED POSE ENCODER MODEL PATH")
        self.parser.add_argument("--hypersim_filenames", "--hpath", type=str, help="the filenames of the hypersim dataset", default="YOUR HYPERSIM FILENAMES")
        # endregion Self-supervised learning

        # region small diffusion
        self.parser.add_argument("--inference_steps", '--is', type=int, help="inference steps", default=20)
        self.parser.add_argument("--num_train_timesteps", '--ntt', type=int, help="num_train_timesteps", default=1000)
        self.parser.add_argument("--ddim_weight", type=float, help="ddim_weight", default=1)

        self.parser.add_argument("--complex_ddim", '--cd', action="store_true", help="use complex ddim to train the weather condition input")
        self.parser.add_argument("--dfs_after_sigmoid", '--das', action="store_true", help="if set,use dfs after sigmod", default=True)
        self.parser.add_argument("--extra_condition", '--ec', action="store_true", help="if set, use rgb and depth as condition", default=True)

        # endregion small diffusion
        # region Common TRAINING
        self.parser.add_argument("--lr_exp_warmup_steps", type=int, default=100, )
        self.parser.add_argument("--lr_total_iter_length", type=int, default=30000, )
        self.parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        self.parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
        self.parser.add_argument("--num_train_epochs", type=int, default=15, )
        self.parser.add_argument("--max_train_steps", type=int, default=30000, )
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.", )
        self.parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.", )
        self.parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate (after the potential warmup period) to use.", )
        self.parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
        self.parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

        # region Diffusion
        self.parser.add_argument("--modality", type=str, choices=["depth", "normals"], default="depth", help="The modality to train the model on.")
        self.parser.add_argument("--noise_type", type=str, default="zeros", choices=["zeros", "gaussian", "pyramid"])
        self.parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2")
        self.parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.", )
        self.parser.add_argument("--variant", type=str, default=None, help="fp16")
        self.parser.add_argument("--output_dir", type=str, default="log_err")
        # endregion Diffusion

        # region IO options
        self.parser.add_argument("--logging_dir", type=str, default="logs", help=(" *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."), )
        self.parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
        self.parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"], )
        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--checkpointing_steps", type=int, default=1000, help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`."))
        self.parser.add_argument("--checkpoints_total_limit", type=int, default=10, help=("Max number of checkpoints to store."), )
        self.parser.add_argument("--resume_from_checkpoint",
            type=str, default=None, help=(
                'Whether training should be resumed from a previous checkpoint. Use a path saved by "latest" to automatically select the last available checkpoint with model_name. When only_test is True, it can set to specific checkpoint path.'
            ),
        )
        self.parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
        self.parser.add_argument("--tracker_project_name", type=str, default="e2e-ft-diffusion", help=("The `project_name` argument passed to Accelerator.init_trackers for"), )

        # region TRI-DEPTH options
        self.parser.add_argument("--use_triplet_loss", "--utl", action="store_true", help="if set,use triplet loss")
        self.parser.add_argument("--disable_hardest_neg", default=False, action="store_true")
        self.parser.add_argument("--disable_isolated_triplet", default=False, action="store_true")
        self.parser.add_argument("--sgt", type=float, default=0.1, help='weight factor for sgt loss')
        self.parser.add_argument("--sgt_scales", nargs='+', type=int, default=[0, 1, 2], help='layer configurations for sgt loss')
        self.parser.add_argument("--sgt_margin", type=float, default=0.35, help='margin for sgt loss')
        self.parser.add_argument("--sgt_isolated_margin", type=float, default=0.65, help='margin for isolated sgt loss')
        self.parser.add_argument("--sgt_kernel_size", type=int, nargs='+', default=[5, 5, 5], help='kernel size (local patch size) for sgt loss')
        self.parser.add_argument("--seg_target", type=str, default="pano_seg", help="the target of the segmentation")
        # endregion TRI-DEPTH options

        # endregion TRAINING

        # region VALIDATION
        self.parser.add_argument("--denoise_steps", type=int, default=1, )
        self.parser.add_argument("--ensemble_size", type=int, default=1)
        self.parser.add_argument("--processing_res", type=int, default=0, help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.", )
        self.parser.add_argument("--output_processing_res", action="store_true", help="When input is resized, out put depth at resized operating resolution. Default: False.", )
        self.parser.add_argument("--resample_method", type=str, default="bilinear", help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.", )
        self.parser.add_argument("--timestep_spacing", type=str, default='trailing', choices=["trailing", "leading"], )
        # self-superivsed learning
        self.parser.add_argument("--post_process", "--pp", help="if set will perform the flipping post processing from the original monodepth paper", action="store_true")
        self.parser.add_argument("--test_improved", "--ti", help="if set will test the improved GT depth, as supervise way!", action="store_true")
        self.parser.add_argument("--test_mode", "--tm", type=str, help="test--original,compare--compare the result between mine and supervised", default="test")
        self.parser.add_argument("--eval_stereo", help="if set evaluates in stereo mode", action="store_true")
        self.parser.add_argument("--eval_mono", help="if set evaluates in mono mode", action="store_true")
        self.parser.add_argument("--disable_median_scaling", help="if set disables median scaling in evaluation", action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor", help="if set multiplies predictions by this number", type=float, default=1)
        self.parser.add_argument("--eval_split", type=str, default="eigen_raw", help="which split to run eval on")
        self.parser.add_argument("--pose_seqs", "--ps", nargs="+", type=int, help="sequences to be evaluated", default=None)
        self.parser.add_argument('--pose_align', "--pa", type=str, choices=['scale', 'scale_7dof', '7dof', '6dof'], default='7dof', help="alignment type")
        self.parser.add_argument("--crop_test", "--ct", type=int, help="1: use the cropped image to test,2: use full image, but test in cropped", default=0)
        # endregion VALIDATION

    def parse(self):
        self.options = self.parser.parse_args()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.options.local_rank:
            self.options.local_rank = env_local_rank
        return self.options


Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])
labels = [
    # name  id  trainId  category catId  hasInstances ignoreInEval color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 1, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

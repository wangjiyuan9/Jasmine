from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import torch.nn as nn
import os
from accelerate.tracking import GeneralTracker, on_main_process
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torch.nn.functional as F
from diffusers.utils.torch_utils import is_compiled_module
import matplotlib as mpl
import cv2
from PIL import Image
from options import MonodepthOptions
from tqdm import tqdm, trange
import torchvision

mpl.use('Agg')
import matplotlib.pyplot as plt

MIN_DEPTH = 1e-3
MAX_DEPTH = 80
index_map = {0: 'abs_rel', 1: 'sq_rel', 2: 'rmse', 3: 'rmse_log', 4: 'a1', 5: 'a2', 6: 'a3', }


########################
# Basic Helper Functions
########################
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def depth_to_disp(depth, min_depth=0.1, max_depth=100):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = 1 / depth - min_disp
    return disp / (max_disp - min_disp)


def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class doNothing():
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass


def depth_read(filename):
    # loads depth map from png file and returns it as a numpy array,
    try:
        depth_png = np.array(Image.open(filename), dtype=int)
    except:
        return None
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

########################
# VAE Helper Functions
########################

# Apply VAE Encoder to image
def encode_image(vae, image):
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent


# Apply VAE Decoder to latent
def decode_image(vae, latent):
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image


########################
# Depth Project 3D
########################
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
            requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


########################
# Training Aid Functions
########################
class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.abs(grad_x), torch.abs(grad_y)


# Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def prepare_dataset(args):
    fpath = os.path.join(os.path.dirname(__file__), "./splits", args.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train")) if not args.supervised else readlines(fpath.format("train_gt"))
    validation_name = "test" if args.test_mode == "test" else "train_test"
    val_filenames = readlines(fpath.format(validation_name))
    hypersim_filenames = readlines("./Marigold/data_split/hypersim/filename_list_train_filtered.txt")
    vkitti_filenames = readlines("./Marigold/data_split/vkitti/vkitti_train.txt")
    eth3d_filenames = readlines("./splits/eth3d/eth3d_filename_list.txt")
    stereo_filenames = readlines("./splits/stereo/{}.txt".format(args.eval_split.split("_")[0])) 
    cityscapes_filenames = readlines("./splits/cityscapes/test_files.txt")
    if args.debug > 0:
        val_filenames = val_filenames[-40:] if args.debug >= 0.5 else val_filenames
        eth3d_filenames = eth3d_filenames[:20] if args.debug >= 0.5 else eth3d_filenames
        stereo_filenames = stereo_filenames[:20] if args.debug >= 0.5 else stereo_filenames
        cityscapes_filenames = cityscapes_filenames[:20] if args.debug >= 0.5 else cityscapes_filenames
        args.dataloader_num_workers = 0 if args.debug >= 2 else args.dataloader_num_workers
        args.max_train_steps = min(20, args.max_train_steps) if (args.debug >= 1 and not args.resume_from_checkpoint) else args.max_train_steps
        args.checkpointing_steps = 1 if args.debug >= 2 else 10

    return train_filenames, val_filenames, stereo_filenames, cityscapes_filenames, args

def prepare_gt_depths(opt=None):
    print("Loading ground truth depths...", end=' ')
    prefix = "" if not "train" in opt.test_mode else "train_"

    gt_depths = np.load(os.path.join(opt.data_path,"kitti", "{}gt_depths.npy".format(prefix)), allow_pickle=True)
    # improved_gt_depths = np.load(os.path.join(opt.data_path, "kitti", "{}improved_gt_depths.npy".format(prefix)), allow_pickle=True)

    if opt.debug:
        gt_depths = gt_depths[-40:] if gt_depths is not None else None
        # if improved_gt_depths is not None:
        #     improved_gt_depths = improved_gt_depths[-40:]
    return gt_depths,None#, improved_gt_depths


class MyCustomTracker(GeneralTracker):
    """
    Custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir,
            **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, **kwargs)

    @on_main_process
    def add_image(self, tag, img_tensor, gloabl_step=None, **kwargs):
        self.writer.add_image(tag=tag, img_tensor=img_tensor, global_step=gloabl_step, **kwargs)


########################
# Pose Transformation
########################

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def move_folder(basepath=None, path=None, mode=None):
    import shutil
    if mode is None:
        raise ValueError("mode is None")
    if path is None:
        raise ValueError("path is None")

    src = os.path.join(basepath.split('visualize')[0], path, 'logs', mode)
    if not os.path.exists(src):
        print("No such file or directory: ", src)
        return
    dst = os.path.join(basepath, "logs", path)
    shutil.copytree(src, dst)


def remove_logfolder(log_path, overwrite=False):
    '''Use this function to remove the duplicate log files'''
    import shutil
    if os.path.exists(log_path) and overwrite:
        shutil.rmtree(log_path)
        print("Has Removed old log files at:  ", log_path)


def easy_visualize(basepath=None, mode=None):
    list = os.listdir(basepath)
    list = [i for i in list if i != 'visualize' and i != 'offical' and "test_" not in i]
    basepath = os.path.join(basepath, 'visualize')
    remove_logfolder(basepath, True)
    os.mkdir(basepath)
    for path in tqdm(list):
        move_folder(basepath, path, mode)


if __name__ == '__main__':
    options = MonodepthOptions()
    opts = options.parse()
    basepath = ':' if opts.vis_path is None else opts.vis_path
    easy_visualize(basepath, 'train')
    

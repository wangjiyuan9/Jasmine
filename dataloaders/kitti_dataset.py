from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset
from .kitti_utils import load_calib_cam_to_cam
import torch


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: 确保您的内部矩阵按原始图像大小进行归一化。若要规范化，需要将第一行缩放 1 image_width将第二行缩放 1 image_height。单深度2假定主点正中。如果您的主点远离中心，则可能需要禁用水平翻转增强。
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, folder, frame_index, side, do_flip):
        image_path = self.get_image_path(folder, frame_index, side,full=True)
        color = self.loader(image_path)
        if self.opts.debug >= 3:
            print(image_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    # segmantation
    def get_seg_map(self, folder, frame_index, side, do_flip, mode='panoptic'):
        path = self.get_image_path(folder, frame_index, side, False if mode == 'panoptic' else True)
        path = path.replace('rgb', '{}_segmentation'.format(mode))
        path = path.replace('/data/0', '/0') if mode == 'semantic' else path
        seg = self.loader(path)
        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg

    def get_pose(self, folder, frame_index, side, do_flip):
        pose_path = self.get_image_path(folder, frame_index, side, False)
        pose_path = pose_path.replace('rgb', 'pose').replace('.png', '.npy')
        if do_flip:
            pose_path = pose_path.replace('pose', 'pose_flip')
        pose = np.load(pose_path, allow_pickle=True).item()
        return pose

    def get_pseudo_disp(self, folder, frame_index, side, do_flip):
        disp_path = self.get_image_path(folder, frame_index, side, False)
        disp_path = disp_path.replace('rgb', 'pseudo_disp')
        disp = self.loader(disp_path, mode='P')
        if do_flip:
            disp = disp.transpose(pil.FLIP_LEFT_RIGHT)

        return disp

    def get_mari_depth(self, folder, frame_index, side, do_flip):
        depth_path = self.get_image_path(folder, frame_index, side, False)
        depth_path = depth_path.replace('rgb', 'mari_depth')
        depth = self.loader(depth_path, mode='P')
        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)
        disp_path = depth_path.replace('mari_depth', 'mari_disp')
        disp = self.loader(disp_path, mode='P')
        if do_flip:
            disp = disp.transpose(pil.FLIP_LEFT_RIGHT)

        return depth, disp


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side, full=False):
        sequence = frame_index[0][11:] if not full else frame_index[0]
        frame = max(int(frame_index[1]), 0)  # 避免负数
        load_folder = folder.split("/")[0]
        addtion_folder = folder.split("/")[1]
        side_folder = {"l": "image_02", "r": "image_03"}[side]
        get_path = os.path.join(self.opts.data_path, load_folder, sequence, side_folder, addtion_folder, "{:010d}".format(frame) + self.img_ext)
        if addtion_folder == "None":
            get_path = os.path.join(self.opts.data_path, load_folder, sequence, "{:010d}".format(frame) + self.img_ext)
        return get_path

    def get_depth(self, folder, frame_index, side, do_flip):
        sequence = frame_index[0][11:]
        frame = int(frame_index[1])
        depth_path = os.path.join(self.opts.data_path, folder, sequence, "proj_depth/groundtruth/image_0{}".format(self.side_map[side]), "{:010d}".format(frame) + self.img_ext)

        depth_gt = pil.open(depth_path)
        # depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256.0

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing    """

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        # 9 0 l
        frame = "{:06d}".format(int(frame_index[1]))
        side_folder = {"l": "image_2", "r": "image_3"}[side]
        get_path = os.path.join(self.opts.data_path, "odometry/{:02d}".format(int(frame_index[0])), side_folder, frame + self.img_ext)

        return get_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}}".format(frame_index, self.img_ext)
        depth_path = os.path.join(self.data_path, folder, "proj_depth/groundtruth/image_0{}".format(self.side_map[side]), f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256.0

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

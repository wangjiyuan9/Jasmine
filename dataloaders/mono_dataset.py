from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from PIL import Image
import random
import torch.utils.data as data
from torchvision import transforms as T
import torch.nn.functional as F
import cv2
import os

cv2.setNumThreads(0)


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode == 'P':
        return Image.open(path)
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


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


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloader    """

    def __init__(self, opts, filenames, hypersim_filenames=None, hypersim_data_path=None, is_train=False, ):
        super(MonoDataset, self).__init__()
        self.opts = opts
        self.filenames = filenames
        self.frame_ids = opts.novel_frame_ids + [0]
        self.is_train = is_train
        self.img_ext = self.opts.img_ext
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.get_all_resize_function()
        self.test_load_map = {"train": [0], "test": [0], "compare": [0], "pose": [0, 1], "dual": [-1, 0], "all_pose": [-1, 0, 1], "physic": [0], "ground": [0]}
        self.target_size = (opts.height, opts.width)
        self.hypersim_filenames = hypersim_filenames
        self.hypersim_data_path=hypersim_data_path
    def rescale_aug(self, inputs, color_aug):
        """ 我们提前创建color_aug对象，并将相同的增强应用于该项目中的所有图像。这可确保输入到姿势网络的所有图像都接收相同的增强。"""
        # 将彩色图像调整为所需的比例并根据需要进行扩充
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                color = self.resize(f)
                inputs["color", im] = self.to_tensor(color)
                inputs[("color_aug", im)] = self.to_tensor(color_aug(color))
                if self.opts.dif_scale:
                    inputs[("color_full", im)] = self.to_tensor(color_aug(self.resize_low(f)))
            if "semantic_seg" in k or "dynamic_mask" in k or "ground_seg" in k or "pano_seg" in k:
                inputs[k] = self.resize_seg(f)
            if "pseudo_disp" in k or "ground" in k or "mari_depth" in k or "mari_disp" in k:
                f = F.interpolate(f[None, ...], size=[self.opts.height, self.opts.width], mode="bilinear")[0]
                inputs[k] = f.clamp(min=1e-7, max=1 - 1e-7) if not "ground" in k else f
                if "pseudo_disp" in k:
                    inputs['disp'] = inputs[k] * 2 - 1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs, base_folder = {'grid': torch.tensor([1])}, 'kitti/data'
        # region test
        if not self.is_train:
            inputs.update(self.get_test_inputs(base_folder, index))
            return inputs
        # endregion
        # region load intric
        K = self.K.copy()
        K[0, :] *= self.opts.width
        K[1, :] *= self.opts.height
        inv_K = np.linalg.pinv(K)
        inputs["K"] = torch.from_numpy(K)
        inputs["inv_K"] = torch.from_numpy(inv_K)
        # endregion

        # # region test
        # if not self.is_train:
        #     info = self.filenames[index].split()
        #     assert len(info) == 3, "The length of info must be equal to 3"
        #     test_frame_ids = self.test_load_map[self.opts.test_mode]
        #     inputs.update(self.get_test_inputs(test_frame_ids, info, base_folder, index))
        #     return inputs
        # # endregion

        # region load_img
        do_flip = random.random() > 0.5
        do_color_aug = random.random() > 0.5
        frame = self.filenames[index].split()
        side = frame[2]
        for i in self.frame_ids:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]  # 全部图片的翻转都是独立的，所以不存在不一致的情况
                inputs[("color", i, -1)] = self.get_color(base_folder, frame, other_side, do_flip)
            else:
                # 此处是-1和0和1
                frame_copy = frame.copy()
                frame_copy[1] = str(int(frame_copy[1]) + int(i))
                inputs[("color", i, -1)] = self.get_color(base_folder, frame_copy, side, do_flip)

        # endregion

        # region load_others and color_aug
        if do_color_aug:
            color_aug = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        # pano_seg
        seg = self.get_seg_map(base_folder, frame, side, do_flip)
        inputs["semantic_seg"] = torch.tensor(np.array(seg)[:, :, 1]).unsqueeze(0)
        inputs["pano_seg"] = torch.tensor(np.array(seg)[:, :, 0]).unsqueeze(0)
        inputs["semantic_seg"][inputs["semantic_seg"] == 255] = 20
        if self.opts.use_gds_loss:
            inputs["dynamic_mask"] = torch.tensor((np.array(seg)[:, :, 2] / 255).astype(np.uint8)).unsqueeze(0)

        if self.opts.define_disp:
            inputs["pseudo_disp", 0] = torch.tensor((np.array(self.get_pseudo_disp(base_folder, frame, side, do_flip)) / 65536.0).astype(np.float32)).unsqueeze(0)

        # if self.opts.use_edge_loss:
        mari_depth, mari_disp = self.get_mari_depth(base_folder, frame, side, do_flip)
        inputs["mari_depth", 0], inputs["mari_disp", 0] = torch.tensor((np.array(mari_depth) / 65536.0).astype(np.float32)).unsqueeze(0), torch.tensor((np.array(mari_disp) / 65536.0).astype(np.float32)).unsqueeze(0)

        # do aug
        if not self.opts.supervised:
            self.rescale_aug(inputs, color_aug)
            if "s" in self.frame_ids:
                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1
                inputs["stereo_T"] = torch.from_numpy(stereo_T)
            inputs["rgb"] = 2 * inputs[("color", 0)] - 1  # [0, 1] -> [-1, 1]
            inputs["depth"] = 2 * inputs["mari_depth", 0] - 1
        else:
            for k in list(inputs):
                f = inputs[k]
                if "color" in k:
                    n, im, i = k
                    inputs[n, im, i] = self.to_tensor(f)
                    inputs[("color_aug", im, i)] = self.to_tensor(color_aug(f))
            depth_gt = self.get_depth("depth_raw", frame, side, do_flip)
            inputs["depth_gt"] = torch.from_numpy(np.expand_dims(depth_gt, axis=0).copy())
            inputs.update(self.do_kb_crop(inputs))
        # endregion 

        # region pose
        pose = self.get_pose(base_folder, frame, side, do_flip)
        inputs[("cam_T_cam", 0, -1)] = torch.from_numpy(pose['negpose'])
        inputs[("cam_T_cam", 0, 1)] = torch.from_numpy(pose['pospose'])
        for i in self.frame_ids:
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
        # endregion

        to_load = self.hypersim_filenames[index].split(" ")[0]
        inputs["hr_rgb"] = self.loader(os.path.join(self.hypersim_data_path, "tr", to_load))
        if do_flip:
            inputs["hr_rgb"] = inputs["hr_rgb"].transpose(Image.FLIP_LEFT_RIGHT)
        # crop 1024 *768 to 1024*320
        crop_start = random.randint(0, 448)
        inputs["hr_rgb"] = inputs["hr_rgb"].crop((0, crop_start, 1024, crop_start + 320))
        inputs["hr_rgb"] = self.resize(self.resize_low(inputs["hr_rgb"]))
        inputs["hhr_rgb"] = self.to_tensor(inputs["hr_rgb"]) * 2. - 1.

        del inputs["hr_rgb"]
        return inputs

    def get_test_inputs(self, base_folder, index):
        inputs = {}
        if "eigen" in self.opts.eval_split:
            info = self.filenames[index].split()
            assert len(info) == 3, "The length of info must be equal to 3"
            test_frame_ids = self.test_load_map[self.opts.test_mode]
            for i in test_frame_ids:
                info_copy = info.copy()
                info_copy[1] = str(int(info_copy[1]) + i)
                get_image = self.get_color(base_folder, info_copy, info[2], False)
                inputs['rgb_int', i] = self.to_tensor(self.resize(get_image))
                if self.opts.test_improved:
                    width, height = get_image.size
                    KB_CROP_HEIGHT = 352
                    KB_CROP_WIDTH = 1216
                    top_margin = int(height - KB_CROP_HEIGHT)
                    left_margin = int((width - KB_CROP_WIDTH) / 2)
                    get_image = self.to_tensor(get_image)
                    get_image = get_image[:, top_margin:top_margin + KB_CROP_HEIGHT, left_margin:left_margin + KB_CROP_WIDTH]
                    get_image = F.interpolate(get_image.unsqueeze(0), (self.opts.height, self.opts.width), mode='bilinear', align_corners=False).squeeze(0)
                    inputs['rgb_int', i] = get_image.clip(0, 1)
            inputs["path"] = self.get_image_path(base_folder, info, info[2])
        elif "sunny" in self.opts.eval_split:
            filename = self.index_to_name("sunny", index)
            get_image, inputs["name"] = self.get_color("sunny", filename, False)
            inputs["depth_gt"] = self.get_depth("sunny", filename)
            inputs["rgb_int", 0] = self.to_tensor(self.resize(get_image))
        elif "foggy" in self.opts.eval_split:
            filename = self.index_to_name("foggy", index)
            get_image, inputs["name"] = self.get_color("foggy", filename, False)
            inputs["depth_gt"] = self.get_depth("foggy", filename)
            inputs["rgb_int", 0] = self.to_tensor(self.resize(get_image))
        elif "rainy" in self.opts.eval_split:
            filename = self.index_to_name("rainy", index)
            get_image, inputs["name"] = self.get_color("rainy", filename, False)
            inputs["depth_gt"] = self.get_depth("rainy", filename)
            inputs["rgb_int", 0] = self.to_tensor(self.resize(get_image))
        elif "cloudy" in self.opts.eval_split:
            filename = self.index_to_name("cloudy", index)
            get_image, inputs["name"] = self.get_color("cloudy", filename, False)
            inputs["depth_gt"] = self.get_depth("cloudy", filename)
            inputs["rgb_int", 0] = self.to_tensor(self.resize(get_image))            
        elif "city" in self.opts.eval_split:
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            get_image = self.get_color(folder, frame_index, side, False)
            inputs["rgb_int", 0] = self.to_tensor(self.resize(self.resize_low(get_image)))
            inputs["depth_gt"]  = self.get_depth(index)
        return inputs

    def do_kb_crop(self, inputs):
        result = {}
        image, depth_gt, org_image = inputs[("color_aug", 0, -1)], inputs["depth_gt"], inputs["color", 0, -1]
        height = image.shape[1]
        width = image.shape[2]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]
        image = F.interpolate(image.unsqueeze(0), (self.opts.height, self.opts.width), mode='bilinear', align_corners=False).squeeze(0)
        org_image = org_image[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]
        org_image = F.interpolate(org_image.unsqueeze(0), (self.opts.height, self.opts.width), mode='bilinear', align_corners=False).squeeze(0)
        depth_gt = depth_gt[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]
        result["color_aug", 0, 0], result["color", 0, 0], result["depth_gt"] = image, org_image, depth_gt
        return result

    def get_all_resize_function(self):
        self.resize = T.Resize((self.opts.height, self.opts.width))
        self.resize_low = T.Resize((192, 640), interpolation=Image.LANCZOS)
        self.resize_high = T.Resize((384 if self.opts.width == 640 else 400, 1280), interpolation=Image.LANCZOS)
        self.resize_seg = T.Resize((self.opts.height, self.opts.width), interpolation=Image.NEAREST)
        self.resize_hyp = T.Resize((480, 640))

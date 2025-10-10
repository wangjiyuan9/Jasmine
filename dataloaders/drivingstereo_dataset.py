import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset
from PIL import Image


class DrivingStereoDataset(MonoDataset):
    RAW_HEIGHT = 800
    RAW_WIDTH = 1762

    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)
        self.forename = {"rainy": "2018-08-17-09-45-58_2018-08-17-10-", "foggy": "2018-10-25-07-37-26_2018-10-25-", "sunny": "2018-10-19-09-30-39_2018-10-19-", "cloudy": "2018-10-31-06-55-01_2018-10-31-"}

    def get_color(self, weather, name, do_flip):
        path, name = self.get_path(weather, name, 'rgb')
        color = self.loader(path)
        return color, name

    def get_depth(self, weather, name, do_flip=False):
        path, _ = self.get_path(weather, name, 'depth')
        depth_png = np.array(Image.open(path), dtype=int)
        assert (np.max(depth_png) > 255)
        gt_depth = depth_png.astype(np.float32) / 256
        return gt_depth

    def get_path(self, weather, frame_name, mode):
        folder = "left-image-full-size" if mode == 'rgb' else "depth-map-full-size"
        image_path = os.path.join(self.opts.data_path,"drivingstereo", weather, folder, frame_name)
        image_name = os.path.join(weather, folder, frame_name)
        if self.opts.debug >= 3:
            print(image_name)
        return image_path, image_name

    def index_to_name(self, weather, index):
        return self.forename[weather] + self.filenames[index] + ".png"

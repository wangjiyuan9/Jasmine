from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image, ImageFile
from scipy.io import loadmat

import torch
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# refer to https://github.com/nianticlabs/monodepth2/issues/392

class Make3DDataset(data.Dataset):
    def __init__(self, data_path, filenames, input_resolution=None):
        super().__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.input_resolution = input_resolution #  H*W
        self.path_format_dict = {
                        'color': ('Test134', 'img-', 'jpg'),
                        'depth': ('Gridlaserdata', 'depth_sph_corr-', 'mat')
                    }
        
        if self.input_resolution is not None:
            ## for the interpolation mode
            ## Image.LANCZOS can replicate the results in monodepth2 paper
            ## Image.NEAREST can be better than LANCZOS
            self.resize = transforms.Resize(self.input_resolution,
                                               interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        line = self.filenames[index]
        base_path = os.path.join(self.data_path, '{}',
                                 '{}' + line + '.{}')
        
        color_path = base_path.format(*self.path_format_dict['color'])
        raw_img = Image.open(color_path).convert('RGB')

        depth_path = base_path.format(*self.path_format_dict['depth'])
        raw_depth = loadmat(depth_path)['Position3DGrid'][:, :, 3]
  
        inputs = {}

        img = raw_img.crop((0, 710, 1704, 1562))
        if self.input_resolution is not None:
            img = self.resize(img)
        img = self.to_tensor(img)
        inputs['color'] = img

        depth = raw_depth[17:38, :]
        depth = torch.from_numpy(depth)
        inputs['depth']  = depth

        return inputs
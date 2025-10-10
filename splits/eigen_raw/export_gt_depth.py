# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

import sys
sys.path.append("../..")
sys.path.append(".")
from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():
    data_path="/opt/data/private/wjy/backup/weather_datasets/kitti/"
    split_folder = os.path.dirname(__file__)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))
    rgb_path=os.path.join(data_path,"rgb")
    split_name = "eigen_raw"

    print("Exporting ground truth depths for {}".format(split_name))

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if split_name == "eigen_raw":
            calib_dir = os.path.join(rgb_path, folder.split("/")[0])
            velo_filename = os.path.join(rgb_path, folder.split("/")[1],
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(data_path, "gt_depths.npy")

    print("Saving to {}".format(split_name))
    np.save(output_path, gt_depths)
    # np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()

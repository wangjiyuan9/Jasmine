from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def load_calib_rigid(filepath):
    """Read a rigid transform calibration file as a numpy.array."""
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data["R"], data["T"])

def load_calib_cam_to_cam(path):
    # We'll return the camera calibration as a dictionary
    data = {}
    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(os.path.join(path, "calib_cam_to_cam.txt"))

    names = ["P_rect_00", "P_rect_01", "P_rect_02", "P_rect_03"]
    if "P0" in filedata:
        names = ["P0", "P1", "P2", "P3"]

    # Create 3x4 projection matrices
    p_rect = [np.reshape(filedata[p], (3, 4)) for p in names]

    for i, p in enumerate(p_rect):
        data[f"P_rect_{i}0"] = p

    # Get image sizes

    for i in range(4):
        data[f"im_size_{i}"] = filedata[f"S_rect_0{i}"]

    # Compute the rectified extrinsics from cam0 to camN
    rectified_extrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        rectified_extrinsics[i][0, 3] = p_rect[i][0, 3] / p_rect[i][0, 0]
        data[f"T_cam{i}_rect"] = rectified_extrinsics[i]

        # Compute the camera intrinsics
        data[f"K_cam{i}"] = p_rect[i][0:3, 0:3]

    # Create 4x4 matrices from the rectifying rotation matrices
    r_rect = None
    if "R_rect_00" in filedata:
        r_rect = [np.eye(4) for _ in range(4)]
        for i in range(4):
            r_rect[i][0:3, 0:3] = np.reshape(filedata["R_rect_0" + str(i)], (3, 3))
            data[f"R_rect_{i}0"] = r_rect[i]

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates

    t_cam0unrect_velo = load_calib_rigid(os.path.join(path, "calib_velo_to_cam.txt"))

    intrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        intrinsics[i][:3] = p_rect[i]

    velo_to_cam = [intrinsics[i].dot(r_rect[0].dot(t_cam0unrect_velo)) for i in range(4)]

    for i in range(4):
        data[f"T_velo_to_cam{i}"] = velo_to_cam[i]

    return data

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(" "))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

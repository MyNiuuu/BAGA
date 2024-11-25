#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from tqdm import tqdm
import torch
import json
import imageio
import cv2
import pickle
import random
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from smpl.smpl_numpy import SMPL
from smplx.body_models import SMPLX


class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    view_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
   

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    eval_cameras: list
    novel_test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


##################################   ZJUMoCapRefine   ##################################

def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def readMydataInfo(path, white_background, output_path, train_view, test_view, data_blur_num=11, test_novel_full_only=False):

    

    if not test_novel_full_only:

        if 'BlurZJU' in path:

            assert data_blur_num % 2 == 1

            train_path = path
            eval_path = path.replace(f'blur{data_blur_num}', 'sharp')
            test_path = eval_path

            image_scaling = 0.5
            xyz_bound_flex = 0.05

            pose_start, pose_interval, pose_num = 0, 1, 50

            eval_pose_start = pose_start * data_blur_num
            eval_pose_interval = pose_interval
            eval_pose_num = pose_num * data_blur_num

            test_pose_start = pose_start * data_blur_num + (data_blur_num - 1) // 2
            test_pose_interval = data_blur_num
            test_pose_num = pose_num
        
            train_zfill, eval_zfill, test_zfill = 3, 3, 6

            train_smpl_mat, eval_smpl_mat, test_smpl_mat = 'json', 'npy', 'npy'

            train_mask, eval_mask, test_mask = 'mask', 'mask', 'mask'

            undistort = True

        elif 'BSHuman' in path:
            train_path = path
            eval_path = path
            test_path = eval_path

            image_scaling = 0.25
            xyz_bound_flex = 0.05

            pose_start, pose_interval, pose_num = 0, 1, 50

            eval_pose_start, eval_pose_interval, eval_pose_num = pose_start, pose_interval, pose_num
            
            test_pose_start, test_pose_interval, test_pose_num = \
                eval_pose_start, eval_pose_interval, eval_pose_num

            train_zfill, eval_zfill, test_zfill = 3, 3, 6

            train_smpl_mat, eval_smpl_mat, test_smpl_mat = 'json', 'json', 'json'

            train_mask, eval_mask, test_mask = 'mask', 'mask', 'mask'

            undistort = True
        
        else:
            assert False, "Unsupported dataset type!"

        print("Reading Training Transforms")
        train_cam_infos = readCamerasMydataTrain(
            train_path, train_view, white_background, image_scaling, xyz_bound_flex, 
            pose_start, pose_interval, pose_num, train_zfill, train_smpl_mat, train_mask, undistort
        )

        eval_cam_infos = train_cam_infos

        print("Reading Test Transforms")
        test_cam_infos = readCamerasMydataTrain(
            test_path, test_view, white_background, image_scaling, xyz_bound_flex, 
            test_pose_start, test_pose_interval, test_pose_num, test_zfill, test_smpl_mat, test_mask, undistort
        )
    
    else:


        assert data_blur_num % 2 == 1 and 'BlurZJU' in path

        train_path = path
        eval_path = path.replace(f'blur{data_blur_num}', 'sharp')
        test_path = eval_path

        image_scaling = 0.5
        xyz_bound_flex = 0.05

        pose_start, pose_interval, pose_num = 0, 1, 50

        eval_pose_start = pose_start * data_blur_num
        eval_pose_interval = pose_interval
        eval_pose_num = pose_num * data_blur_num

        test_pose_start = pose_start * data_blur_num + (data_blur_num - 1) // 2
        test_pose_interval = data_blur_num
        test_pose_num = pose_num
    
        train_zfill, eval_zfill, test_zfill = 3, 3, 6

        train_smpl_mat, eval_smpl_mat, test_smpl_mat = 'json', 'npy', 'npy'

        train_mask, eval_mask, test_mask = 'mask', 'mask', 'mask'

        undistort = True

        train_cam_infos = readCamerasMydataTrain(
            test_path, test_view, white_background, image_scaling, xyz_bound_flex, 
            test_pose_start, test_pose_interval, test_pose_num, test_zfill, test_smpl_mat, test_mask, undistort
        )
        
        eval_cam_infos = readCamerasMydataTrain(
            test_path, test_view, white_background, image_scaling, xyz_bound_flex, 
            eval_pose_start, eval_pose_interval, eval_pose_num, test_zfill, test_smpl_mat, test_mask, undistort
        )

        test_cam_infos = train_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           novel_test_cameras=test_cam_infos, 
                           eval_cameras=eval_cam_infos, 
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasMydataTrain(path, output_view, white_background, image_scaling=0.5, xyz_bound_flex=0.05, pose_start=0, pose_interval=1, pose_num=50, zfill_num=6, smpl_format='json', mask_tag='mask', undistort=True):
    cam_infos = []

    img_names = os.listdir(os.path.join(path, 'images', output_view[0]))
    img_names.sort()

    pose_num = min(pose_num, len(img_names))

    ims = np.array([
        np.array([
            os.path.join('images', view, fname) for view in output_view
        ]) for fname in img_names[pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])

    cam_inds = np.array([
        np.array(output_view) for _ in img_names[pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])

    smpl_model = SMPL(sex='neutral')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # print(big_pose_xyz)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= xyz_bound_flex
    big_pose_max_xyz += xyz_bound_flex
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    extrinsics = {}
    
    for json_name in os.listdir(os.path.join(path, 'camera_extris')):
        with open(os.path.join(path, 'camera_extris', json_name), 'r', encoding='utf-8') as file:
            camera_id = os.path.splitext(json_name)[0]
            data = json.load(file)
            extrinsics[camera_id] = data

    idx = 0
    for pose_index in tqdm(range(pose_num)):
        for view_index in range(len(output_view)):

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            image_name = ims[pose_index][view_index].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32)/255.)

            msk_path = image_path.replace('images', mask_tag).replace('.jpg', '.png').replace('.jpeg', '.png')

            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

            cam_ind = cam_inds[pose_index][view_index]

            K = np.array(extrinsics[cam_ind]['K'])
            D = np.array(extrinsics[cam_ind]['dist'])
            R = np.array(extrinsics[cam_ind]['R'])
            T = np.array(extrinsics[cam_ind]['T'])

            if 'BlurZJU' in path:
                T = T / 1000.

            if undistort:
                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)

            image[msk == 0] = 1 if white_background else 0

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.:

                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)

                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])

            # load smpl data
            if smpl_format == 'json':
                i = os.path.splitext(os.path.basename(image_path))[0]
                vertices_path = os.path.join(path, 'smpl_vertices', f'{str(i).zfill(6)}.json')
                with open(vertices_path, 'r', encoding='utf-8') as file:
                    xyz = np.array(json.load(file)[0]['vertices']).astype(np.float32)
                
                smpl_param_path = os.path.join(path, "smpl/smpl", f'{str(i).zfill(6)}.json')
                with open(smpl_param_path, 'r', encoding='utf-8') as file:
                    smpl_param = json.load(file)[0]
                smpl_param['Rh'] = np.array(smpl_param['Rh'])
                smpl_param['R'] = cv2.Rodrigues(smpl_param['Rh'])[0].astype(np.float32)
                smpl_param['Th'] = np.array(smpl_param['Th']).astype(np.float32)
                smpl_param['shapes'] = np.array(smpl_param['shapes']).astype(np.float32)
                smpl_param['poses'] = np.array(smpl_param['poses']).astype(np.float32)
            elif smpl_format == 'npy':
                i = int(os.path.basename(image_path)[:-4])
                vertices_path = os.path.join(path, 'smpl_vertices', '{}.npy'.format(i))
                xyz = np.load(vertices_path).astype(np.float32)

                smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
                smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
                Rh = smpl_param['Rh']
                smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
                smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
                smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
                smpl_param['poses'] = smpl_param['poses'].astype(np.float32)
            else:
                assert False

            # obtain the original bounds for point sampling
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= xyz_bound_flex
            max_xyz += xyz_bound_flex
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))

            cam_infos.append(
                CameraInfo(
                    uid=idx, pose_id=pose_index, view_id=output_view[view_index], 
                    R=R, T=T, K=K, FovY=FovY, FovX=FovX, 
                    image=image, image_path=image_path, image_name=image_name, 
                    bkgd_mask=bkgd_mask, bound_mask=bound_mask, 
                    width=image.size[0], height=image.size[1], 
                    smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                    big_pose_smpl_param=big_pose_smpl_param, 
                    big_pose_world_vertex=big_pose_xyz, 
                    big_pose_world_bound=big_pose_world_bound
                )
            )

            idx += 1
            
    return cam_infos



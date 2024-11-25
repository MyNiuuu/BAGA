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
import time
from random import randint
import torch
import torchvision
import random

import cv2
import numpy as np
from tqdm import tqdm
import lpips
from natsort import natsorted
from argparse import ArgumentParser, Namespace

from gaussian_renderer import render_posemodel, bat_rod
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from error_map_jet import colorize_np
from Spline import *
from posemodel_spline import PoseModel
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr



loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))




def cal_pose_loss(pose_model, gaussians: GaussianModel, blur_num):
    
    _, smpl_poses_69, smpl_Rh, smpl_Th = pose_model.get_intermediate_poses_from_indices(list(range(pose_model.img_num)), blur_num)

    poses_start, poses_end = smpl_poses_69[:, 0, :], smpl_poses_69[:, -1, :]
    Rhs_start, Rhs_end = smpl_Rh[:, 0, :], smpl_Rh[:, -1, :]
    Ths_start, Ths_end = smpl_Th[:, 0, :], smpl_Th[:, -1, :]

    pose_diff = torch.abs(poses_start[1:] - poses_end[:-1])
    Th_diff = torch.abs(Ths_start[1:] - Ths_end[:-1])

    Rhs_start_matrices = bat_rod(Rhs_start)[1:]
    Rhs_end_matrices = bat_rod(Rhs_end)[:-1]

    means3D = gaussians.get_xyz
    indices = torch.randperm(means3D.shape[0])[:50]
    points = means3D[indices].detach()

    Rh_diff = 0.
    for iii in range(points.shape[0]):
        point = points[iii].reshape(1, 3, 1).repeat(Rhs_start_matrices.shape[0], 1, 1)
        Rhs_start_points = torch.bmm(Rhs_start_matrices, point).squeeze(-1)
        Rhs_end_points = torch.bmm(Rhs_end_matrices, point).squeeze(-1)
        Rh_diff += torch.abs(Rhs_start_points - Rhs_end_points).mean()

    return pose_diff.mean() + Rh_diff * 100 + Th_diff.mean()



def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))



@torch.no_grad()
def eval_novel_pose(pose_model, iteration, testing_iterations, scene: Scene, pipeline, background, csv_path, combine_path, crop_path, use_pose_offset=True, use_lbs_offset=True, save_video=False, no_mask=False, data_blur_num=5):

    if iteration == testing_iterations[0]:
        with open(csv_path, 'a') as outfile:
            outfile.write("iteration,datalength,mPSNR,mSSIM,mLPIPS\n")

    torch.cuda.empty_cache()
    eval_cameras = natsorted(scene.getNovelTestCameras().copy(), key=lambda x: x.image_name)

    if eval_cameras and len(eval_cameras) > 0:

        masked_psnrs = 0.0
        masked_ssims = 0.0
        masked_lpipss = 0.0

        bbox_images = []
        gt_images = []
        err_maps = []
        nomask_images = []
        for idx, viewpoint in enumerate(tqdm(eval_cameras)):

            gt_image = viewpoint.original_image[0:3, :, :].cuda()
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            bound_mask = viewpoint.bound_mask
            bkgd_mask = viewpoint.bkgd_mask.cuda().repeat(3, 1, 1).bool()

            render_output = render_posemodel(
                viewpoint, pose_model, scene.gaussians, pipeline, background, return_smpl_rot=False, return_avg=False, use_pose_offset=use_pose_offset, use_lbs_offset=use_lbs_offset, only_middle=True, model_blur_num=data_blur_num
                )
            tmp = torch.clamp(render_output["render"], 0.0, 1.0)

            tmp.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

            imag1 = tmp.permute(1, 2, 0).detach().cpu().numpy()
            imag2 = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            err_map = np.mean(np.abs(imag1 - imag2), axis=-1)
            err_map_colored = torch.from_numpy(colorize_np(err_map, cmap_name='jet', range=(0., 1.))).cuda().float().permute(2, 0, 1)

            non_zero_indices = torch.nonzero(bkgd_mask[0], as_tuple=True)
            if non_zero_indices[0].numel() > 0:
                min_x = max(0, torch.min(non_zero_indices[1]).item() - 5)
                min_y = max(0, torch.min(non_zero_indices[0]).item() - 5)
                max_x = min(bkgd_mask.shape[-1], torch.max(non_zero_indices[1]).item() + 5)
                max_y = min(bkgd_mask.shape[-2], torch.max(non_zero_indices[0]).item() + 5)
            
            rendering_crop = tmp[:, min_y:max_y, min_x:max_x]
            gt_crop = gt_image[:, min_y:max_y, min_x:max_x]
            bkgd_mask_crop = bkgd_mask[:, min_y:max_y, min_x:max_x]

            imag1 = rendering_crop.permute(1, 2, 0).detach().cpu().numpy()
            imag2 = gt_crop.permute(1, 2, 0).detach().cpu().numpy()
            err_map = np.mean(np.abs(imag1 - imag2), axis=-1)
            err_map_colored_crop = torch.from_numpy(colorize_np(err_map, cmap_name='jet', range=(0., 1.))).cuda().float().permute(2, 0, 1)

            masked_psnrs += psnr(rendering_crop, gt_crop).mean().double().cpu()
            masked_ssims += ssim(rendering_crop, gt_crop).mean().double().cpu()
            masked_lpipss += loss_fn_vgg(rendering_crop[None], gt_crop[None]).mean().double().cpu()

            torchvision.utils.save_image(
                torch.cat([bkgd_mask_crop, rendering_crop, gt_crop, err_map_colored_crop], dim=-1), 
                os.path.join(crop_path, f"{str(viewpoint.pose_id).zfill(3)}_{str(viewpoint.view_id).zfill(2)}.png")
            )

            if save_video:

                bbox_images.append(tmp)
                gt_images.append(gt_image)
                err_maps.append(err_map_colored)

        if save_video:

            if no_mask:
                nomask_images = torch.stack(nomask_images)

            bbox_images = torch.stack(bbox_images)
            gt_images = torch.stack(gt_images)
            err_maps = torch.stack(err_maps)

        masked_psnrs /= len(eval_cameras)   
        masked_ssims /= len(eval_cameras)
        masked_lpipss /= len(eval_cameras)

        print(f"\n[ITER {iteration}] Novel View w/ Pose Test {len(eval_cameras)}: mPSNR {masked_psnrs} mSSIM {masked_ssims} mLPIPS {masked_lpipss}")
        
        with open(csv_path, 'a') as outfile:
            outfile.write(f"{iteration},{len(eval_cameras)},{masked_psnrs},{masked_ssims},{masked_lpipss}\n")

        if save_video:

            concat_tensor = torch.cat((bbox_images, gt_images, err_maps), dim=3)

            concat_tensor = concat_tensor.permute(0, 2, 3, 1).detach().cpu()
            concat_np = np.uint8(concat_tensor.numpy()*255)

            save_path = os.path.dirname(combine_path) + '.mp4'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torchvision.io.write_video(save_path, concat_np, fps=10, video_codec='h264')


@torch.no_grad()
def eval_interpolate(pose_model, iteration, train_views, testing_iterations, scene : Scene, csv_path, combine_path, crop_path, renderArgs, use_pose_offset=True, use_lbs_offset=True, save_video=True, data_blur_num=5, model_blur_num=11, pose_mode='spline_4', full_resolution=False):

    csv_path_view = os.path.join(os.path.dirname(csv_path), 'test_metrics_views.txt')

    all_psnr, all_ssim, all_lpips = 0., 0., 0.
    all_psnr_mid, all_ssim_mid, all_lpips_mid = 0., 0., 0.

    all_np = []

    if iteration == testing_iterations[0]:
        with open(csv_path, 'a') as outfile:
            outfile.write("iteration,datalength,view,mPSNR,mSSIM,mLPIPS\n")

    for vidx, train_view in enumerate(train_views):

        torch.cuda.empty_cache()
        train_cameras = natsorted(scene.getTrainCameras().copy(), key=lambda x: x.image_name)
        eval_cameras = natsorted(scene.getEvalCameras().copy(), key=lambda x: x.image_name)

        images, blur_gt_images, gt_sharp_images, bkgd_masks = [], [], [], []
        for _, viewpoint in enumerate(train_cameras):

            if viewpoint.view_id != train_view:
                continue
            
            render_output = render_posemodel(
                viewpoint, pose_model, scene.gaussians, *renderArgs, 
                model_blur_num=data_blur_num if pose_mode != 'free' else model_blur_num, 
                return_smpl_rot=False, 
                return_avg=False, use_pose_offset=use_pose_offset, 
                use_lbs_offset=use_lbs_offset, 
                only_middle=False, 
            )

            image = torch.clamp(render_output["render"], 0.0, 1.0)

            if pose_mode == 'free':
                image = image[::((model_blur_num - data_blur_num) // (data_blur_num - 1) + 1)]

            bkgd_mask = viewpoint.bkgd_mask.cuda().repeat(3, 1, 1).bool()

            blur_gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            images.append(image)
            blur_gt_images.append(blur_gt_image[None, :].repeat(data_blur_num, 1, 1, 1))
            bkgd_masks.append(bkgd_mask[None, :].repeat(data_blur_num, 1, 1, 1))
        images = torch.cat(images, dim=0)  # [50*11, 3, 512, 512]
        bkgd_masks = torch.cat(bkgd_masks, dim=0)
        blur_gt_images = torch.cat(blur_gt_images, dim=0)

        for _, viewpoint in enumerate(eval_cameras):
            if viewpoint.view_id != train_view:
                continue
            gt_sharp_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_sharp_images.append(gt_sharp_image)
        gt_sharp_images = torch.stack(gt_sharp_images)

        if gt_sharp_images.shape[0] != images.shape[0]:
            gt_sharp_images = gt_sharp_images[:, None].repeat(1, data_blur_num, 1, 1, 1).flatten(0, 1)
            loop_len = data_blur_num
        else:
            loop_len = 1

        masked_psnrs = 0.0
        masked_ssims = 0.0
        masked_lpipss = 0.0

        masked_psnrs_mid = 0.0
        masked_ssims_mid = 0.0
        masked_lpipss_mid = 0.0
        
        bbox_images = []
        err_maps = []
        
        idx = 0
        for _, viewpoint in enumerate(eval_cameras):
            if viewpoint.view_id != train_view:
                continue

            for _ in range(loop_len):

                tmp = images[idx]
                    
                blur_gt = blur_gt_images[idx]
                gt_image = gt_sharp_images[idx]

                bkgd_mask = bkgd_masks[idx]

                imag1 = tmp.permute(1, 2, 0).detach().cpu().numpy()
                imag2 = gt_image.permute(1, 2, 0).detach().cpu().numpy()
                err_map = np.mean(np.abs(imag1 - imag2), axis=-1)
                err_map_colored = torch.from_numpy(colorize_np(err_map, cmap_name='jet', range=(0., 1.))).cuda().float().permute(2, 0, 1)

                if full_resolution:

                    rendering_crop = tmp
                    gt_crop = gt_image
                    bkgd_mask_crop = bkgd_mask
                    blur_gt = blur_gt

                else:

                    non_zero_indices = torch.nonzero(bkgd_mask[0], as_tuple=True)
                    if non_zero_indices[0].numel() > 0:
                        min_x = max(0, torch.min(non_zero_indices[1]).item() - 5)
                        min_y = max(0, torch.min(non_zero_indices[0]).item() - 5)
                        max_x = min(bkgd_mask.shape[-1], torch.max(non_zero_indices[1]).item() + 5)
                        max_y = min(bkgd_mask.shape[-2], torch.max(non_zero_indices[0]).item() + 5)

                    rendering_crop = tmp[:, min_y:max_y, min_x:max_x]
                    gt_crop = gt_image[:, min_y:max_y, min_x:max_x]
                    bkgd_mask_crop = bkgd_mask[:, min_y:max_y, min_x:max_x]
                    blur_gt = blur_gt[:, min_y:max_y, min_x:max_x]

                masked_psnr = psnr(rendering_crop, gt_crop).mean().double()
                masked_ssim = ssim(rendering_crop, gt_crop).mean().double()
                masked_lpips = loss_fn_vgg(rendering_crop[None], gt_crop[None]).mean().double()

                masked_psnrs += masked_psnr
                masked_ssims += masked_ssim
                masked_lpipss += masked_lpips

                is_middle = idx - (data_blur_num // 2)
                if is_middle >=0 and is_middle % data_blur_num == 0:
                    masked_psnrs_mid += masked_psnr
                    masked_ssims_mid += masked_ssim
                    masked_lpipss_mid += masked_lpips

                imag1 = rendering_crop.permute(1, 2, 0).detach().cpu().numpy()
                imag2 = gt_crop.permute(1, 2, 0).detach().cpu().numpy()
                err_map = np.mean(np.abs(imag1 - imag2), axis=-1)
                err_map_colored_crop = torch.from_numpy(colorize_np(err_map, cmap_name='jet', range=(0., 1.))).cuda().float().permute(2, 0, 1)

                torchvision.utils.save_image(
                    torch.cat([
                        rendering_crop, gt_crop, err_map_colored_crop], dim=-1), 
                    os.path.join(crop_path, f"{idx}_{str(train_view).zfill(2)}.png")
                )

                if save_video:

                    bbox_images.append(tmp)
                    err_maps.append(err_map_colored)
                
                idx += 1

        if save_video:
            bbox_images = torch.stack(bbox_images)
            err_maps = torch.stack(err_maps)

        

        total_num = images.shape[0]

        masked_psnrs /= total_num
        masked_ssims /= total_num
        masked_lpipss /= total_num

        masked_psnrs_mid /= (total_num / data_blur_num)
        masked_ssims_mid /= (total_num / data_blur_num)
        masked_lpipss_mid /= (total_num / data_blur_num)

        with open(csv_path_view, 'a') as outfile:
            outfile.write(f"{iteration},{len(eval_cameras)},{str(train_view).zfill(2)},{masked_psnrs},{masked_ssims},{masked_lpipss}\n")
            outfile.write(f"{iteration},{len(eval_cameras)},{str(train_view).zfill(2)},Middle Frame,{masked_psnrs_mid},{masked_ssims_mid},{masked_lpipss_mid}\n")

        all_psnr += masked_psnrs
        all_ssim += masked_ssims
        all_lpips += masked_lpipss

        all_psnr_mid += masked_psnrs_mid
        all_ssim_mid += masked_ssims_mid
        all_lpips_mid += masked_lpipss_mid

        if save_video and vidx == 0:
            
            concat_tensor = torch.cat((blur_gt_images, bbox_images, gt_sharp_images, err_maps), dim=-1)
            
            print(concat_tensor.shape)
            
            concat_tensor = concat_tensor.permute(0, 2, 3, 1).detach().cpu()
            concat_np = np.uint8(concat_tensor.numpy()*255)

            all_np.append(concat_np)

    all_psnr /= len(train_views)   
    all_ssim /= len(train_views)
    all_lpips /= len(train_views)

    all_psnr_mid /= len(train_views)   
    all_ssim_mid /= len(train_views)
    all_lpips_mid /= len(train_views)

    print(f"\n[ITER {iteration}] Training View Interpolation {len(eval_cameras)} View All: mPSNR {all_psnr} mSSIM {all_ssim} mLPIPS {all_lpips}")
    print(f"\n[ITER {iteration}] Training View Interpolation {len(eval_cameras)} View All Middle Frame: mPSNR {all_psnr_mid} mSSIM {all_ssim_mid} mLPIPS {all_lpips_mid}")
    
    with open(csv_path, 'a') as outfile:
        outfile.write(f"{iteration},{len(eval_cameras)},All,{all_psnr},{all_ssim},{all_lpips}\n")
        outfile.write(f"{iteration},{len(eval_cameras)},All,Middle Frame,{all_psnr_mid},{all_ssim_mid},{all_lpips_mid}\n")

    if save_video:

        all_np = np.concatenate(all_np, axis=1)

        save_path = os.path.dirname(combine_path) + '.mp4'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torchvision.io.write_video(save_path, all_np, fps=10, video_codec='h264')


def training(
        dataset, opt, pipe, debug_from, 
        data_blur_num, model_blur_num, pose_mode, use_pose_offset, use_lbs_offset, max_sh, use_poseloss, eval_every, ssim_weight, lpips_weight, load_iter=None, test_novel_full_only=False, test_novel_view_index=0
    ):

    load_pretrain = test_novel_full_only
    if load_pretrain:
        assert load_iter != 0
    
    load_pose_model = load_pretrain

    print(f'using pose offset: {use_pose_offset}!')
    print(f'using lbs offset: {use_lbs_offset}!')
    print(f'using pose loss: {use_poseloss}!')
    print(f'using pose mode: {pose_mode}!')
    print(f'Dataset Path: {dataset.source_path}')

    if 'BlurZJU' in dataset.source_path:
        all_views = [str(i).zfill(2) for i in range(0, 23)]
        all_views.remove('03')
        train_views = ['04', '10', '16', '22']
        test_views = [x for x in all_views if x not in train_views]
        if test_novel_full_only:
            test_views = test_views[test_novel_view_index:(test_novel_view_index+1)]
    elif 'BSHuman' in dataset.source_path:
        train_views = ['19305328', '19305322', '19308875', '19061154', ]
        all_views = os.listdir(os.path.join(dataset.source_path, 'images'))
        test_views = [x for x in all_views if x not in train_views]
    else:
        assert False, "Unsupported Dataset!"

    print(f'Training Views: {train_views}')
    print(f'Testing Views: {test_views}')

    testing_iterations = [x * eval_every for x in range(1, opt.iterations // eval_every + 1)]
    saving_iterations = [x * eval_every for x in range(1, opt.iterations // eval_every + 1)]

    print('test_iterations:', testing_iterations)
    print('save_iterations:', saving_iterations)

    first_iter = 0
    prepare_output_and_logger(dataset)

    dataset.sh_degree = max_sh

    gaussians = GaussianModel(
        dataset.sh_degree, dataset.smpl_type, 
        dataset.motion_offset_flag, dataset.actor_gender
    )
    scene = Scene(
        dataset, gaussians, 
        train_views=train_views, test_views=test_views, data_blur_num=data_blur_num, 
        load_iteration=load_iter if load_pretrain else None, 
        test_novel_full_only=test_novel_full_only
    )

    print("Scene Setup Finished...")

    view_flag = train_views[0] if not test_novel_full_only else test_views[0]

    viewpoint_stack4poseinit = natsorted(scene.getTrainCameras().copy(), key=lambda x: x.uid)
    smpl_shapes, smpl_poses, smpl_Rh, smpl_Th = [], [], [], []
    while len(viewpoint_stack4poseinit) != 0:
        v_cam = viewpoint_stack4poseinit.pop(0)
        C, H, W = v_cam.original_image.shape
        if v_cam.view_id != view_flag:
            continue
        shape, pose, Rh, Th = \
                v_cam.smpl_param['shapes'], v_cam.smpl_param['poses'][:, 3:], \
                    v_cam.smpl_param['Rh'], v_cam.smpl_param['Th']
        smpl_shapes.append(shape)
        smpl_poses.append(pose)
        smpl_Rh.append(Rh)
        smpl_Th.append(Th)
    smpl_shapes_nomean = torch.cat(smpl_shapes, dim=0)
    smpl_shapes = torch.mean(torch.cat(smpl_shapes, dim=0), dim=0, keepdim=True)
    smpl_poses = torch.cat(smpl_poses, dim=0)
    smpl_Rh = torch.cat(smpl_Rh, dim=0)
    smpl_Th = torch.cat(smpl_Th, dim=0)

    print(f'Init SMPL Shapes: {smpl_shapes.shape}')
    print(f'Init SMPL Poses: {smpl_poses.shape}')
    print(f'Init SMPL Rotation: {smpl_Rh.shape}')
    print(f'Init SMPL Translation: {smpl_Th.shape}')

    del viewpoint_stack4poseinit

    knot_num = int(pose_mode.split('_')[1])
    pose_model = PoseModel(smpl_shapes, smpl_poses, smpl_Rh, smpl_Th, knot_num).cuda()
    
    if load_pose_model:
        pose_model.load_state_dict(torch.load(os.path.join(scene.model_path.replace('_stage2', ''), f'chkpnt_posemodel{load_iter}.pth'))['pose_model'])
        lr_pose, pose_optimizer = None, None
        pose_model.requires_grad_(False)
    else:
        lr_pose = 1e-3
        pose_optimizer = torch.optim.Adam(pose_model.parameters(), lr=lr_pose)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    if test_novel_full_only:

        csv_path = os.path.join(scene.model_path, 'inference/novel_full_testtime', "text_metrics.txt")
        combine_path = os.path.join(scene.model_path, 'inference/novel_full_testtime', "ours_{}".format(load_pretrain), "all")
        crop_path = os.path.join(scene.model_path, 'inference/novel_full_testtime', "ours_{}".format(load_pretrain), "crop")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(combine_path, exist_ok=True)
        os.makedirs(crop_path, exist_ok=True)

        eval_interpolate(
            pose_model, load_iter, 
            test_views, 
            [load_iter], scene, csv_path=csv_path, combine_path=combine_path, 
            crop_path=crop_path, renderArgs=(pipe, background), 
            use_pose_offset=use_pose_offset, use_lbs_offset=use_lbs_offset, 
            save_video=True, 
            data_blur_num=data_blur_num if 'zju' in dataset.source_path else model_blur_num, 
            model_blur_num=model_blur_num, 
            pose_mode=pose_mode, 
            full_resolution=True
        )

        return

    gaussians.training_setup(opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            # never used
            assert False
            pipe.debug = True

        
        render_pkg = render_posemodel(
            viewpoint_cam, pose_model, gaussians, pipe, background, 
            model_blur_num=model_blur_num, 
            use_pose_offset=use_pose_offset, use_lbs_offset=use_lbs_offset, 
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()

        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])

        if use_poseloss != 0. and 0 < iteration < 10000:
            pose_loss = cal_pose_loss(pose_model, gaussians, model_blur_num) * use_poseloss
        else:
            pose_loss = torch.tensor(0., device=gt_image.device)

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)

        # ssim loss
        ssim_loss = ssim(img_pred, img_gt) if ssim_weight != 0 else torch.tensor(0., device=gt_image.device)
        # lipis loss

        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1) if lpips_weight != 0 else torch.tensor(0., device=gt_image.device)

        loss = Ll1 + ssim_weight * (1.0 - ssim_loss) + lpips_weight * lpips_loss + pose_loss

        loss.backward()

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            Ll1_loss_for_log = Ll1.item()
            ssim_loss_for_log = ssim_loss.item()
            lpips_loss_for_log = lpips_loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "pose Loss": f"{pose_loss:.{2}f}", "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            if iteration in testing_iterations:

                csv_path = os.path.join(scene.model_path, 'eval_novel_pose', "text_metrics.txt")
                combine_path = os.path.join(scene.model_path, 'eval_novel_pose', "iter_{}".format(iteration), "all")
                crop_path = os.path.join(scene.model_path, 'eval_novel_pose', "iter_{}".format(iteration), "crop")

                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                os.makedirs(combine_path, exist_ok=True)
                os.makedirs(crop_path, exist_ok=True)

                eval_novel_pose(
                    pose_model, iteration, 
                    testing_iterations, scene=scene, pipeline=pipe, background=background, 
                    csv_path=csv_path, combine_path=combine_path, crop_path=crop_path, 
                    use_pose_offset=use_pose_offset, use_lbs_offset=use_lbs_offset, 
                    data_blur_num=data_blur_num if 'zju' in dataset.source_path else model_blur_num
                )

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if not load_pose_model:
                pose_optimizer.step()
                pose_optimizer.zero_grad()

            if not load_pose_model:
                decay_rate_pose = 0.01
                decay_end = min(20000, opt.iterations)
                new_lrate_pose = lr_pose * (decay_rate_pose ** min(1, iteration / decay_end))
                for param_group in pose_optimizer.param_groups:
                    param_group['lr'] = new_lrate_pose

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                if not load_pose_model:
                    torch.save({
                        'pose_model': pose_model.state_dict(), 
                        'iteration': iteration, 
                        'pose_opt': pose_optimizer.state_dict(), 
                        }, 
                    scene.model_path + "/chkpnt_posemodel" + str(iteration) + ".pth")
                else:
                    torch.save({
                        'pose_model': pose_model.state_dict(), 
                        'iteration': iteration, 
                        # 'pose_opt': pose_optimizer.state_dict(), 
                        }, 
                    scene.model_path + "/chkpnt_posemodel" + str(iteration) + ".pth")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
if __name__ == "__main__":

    setup_seed(20240819)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model_blur_num", type=int, default=11)
    parser.add_argument("--data_blur_num", type=int, default=0)
    parser.add_argument("--pose_mode", type=str, default='polyline')
    parser.add_argument("--use_pose_offset", action='store_true')
    parser.add_argument("--use_lbs_offset", action='store_true')
    parser.add_argument("--max_sh", type=int, default=1)
    parser.add_argument("--pose_loss", type=float, default=0.)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--ssim_weight", type=float, default=0.01)
    parser.add_argument("--lpips_weight", type=float, default=0.01)
    parser.add_argument("--load_iter", type=int, default=0)
    parser.add_argument("--test_novel_full_only", action='store_true')
    parser.add_argument("--test_novel_view_index", type=int, default=0)
    
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.debug_from, args.data_blur_num, args.model_blur_num, args.pose_mode, args.use_pose_offset, args.use_lbs_offset, args.max_sh, args.pose_loss, args.eval_every, args.ssim_weight, args.lpips_weight, args.load_iter, test_novel_full_only=args.test_novel_full_only, test_novel_view_index=args.test_novel_view_index)

    # All done
    print("\nTraining complete.")

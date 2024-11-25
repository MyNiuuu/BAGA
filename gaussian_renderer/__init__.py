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
import io
import numpy as np
from PIL import Image
import torch
import math
import matplotlib.pyplot as plt

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

from posemodel_spline import PoseModel


def bat_rod(rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)



def world_to_camera(P_world, RT):
    RT = RT.to(P_world.device)[None, :].repeat(P_world.shape[0], 1, 1)
    ones = torch.ones_like(P_world[..., 0:1], device=P_world.device).float()
    P_world = torch.cat([P_world, ones], dim=-1)  # [11, N, 4]
    P_camera_hom = torch.bmm(P_world, RT)
    P_camera = P_camera_hom[..., :3] / P_camera_hom[..., 3:4]
    return P_camera



def camera_to_image(P_camera, K):
    P_image = torch.matmul(P_camera, K.T)  # [11, N, 3]
    P_image = P_image[..., :2] / P_image[..., 2:3]
    return P_image


def get_max_2d_diff(means3D, viewpoint_camera):

    P_image = camera_to_image(
        world_to_camera(
            means3D.detach().float(), viewpoint_camera.world_view_transform
        ), torch.from_numpy(viewpoint_camera.K).to(means3D.device).float()
    )

    max_coords, _ = torch.max(P_image, dim=0)
    min_coords, _ = torch.min(P_image, dim=0)

    coord_diff = torch.sqrt(torch.sum((max_coords - min_coords) ** 2, dim=-1, keepdim=True))

    return torch.max(coord_diff)



def render_posemodel(viewpoint_camera, pose_model: PoseModel, pc : GaussianModel, pipe, bg_color : torch.Tensor, model_blur_num, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None, return_avg=True, use_pose_offset=True, use_lbs_offset=True, only_middle=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """


    '''
        affected by smpl init: smpl_param, world_vertex, world_bound, bound_mask
    '''

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz

    if not pc.motion_offset_flag:
        # not used
        assert False
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None])
    else:
        # assert False
        if transforms is None:

            smpl_shapes, smpl_poses_69, smpl_Rh, smpl_Th = \
                pose_model.get_intermediate_poses_from_indices([viewpoint_camera.pose_id], model_blur_num)

            smpl_shapes, smpl_poses_69, smpl_Rh, smpl_Th = \
                smpl_shapes.squeeze(), smpl_poses_69.squeeze(), smpl_Rh.squeeze(), smpl_Th.squeeze()

            smpl_poses = torch.cat(
                [torch.tensor([[0, 0, 0]], device=smpl_poses_69.device).repeat(smpl_poses_69.shape[0], 1), smpl_poses_69], dim=1
            )

            smpl_R = bat_rod(smpl_Rh).squeeze()

            if only_middle:
                pos = (smpl_shapes.shape[0] - 1) // 2
                smpl_shapes, smpl_poses, \
                    smpl_poses_69, smpl_R, \
                        smpl_Rh, smpl_Th = \
                            smpl_shapes[pos:(pos+1)], smpl_poses[pos:(pos+1)], \
                                smpl_poses_69[pos:(pos+1)], smpl_R[pos:(pos+1)], \
                                    smpl_Rh[pos:(pos+1)], smpl_Th[pos:(pos+1)]

            # SMPL rotations
            if use_pose_offset:
                correct_Rs = pc.pose_decoder(smpl_poses_69)['Rs']
            else:
                correct_Rs = None

            # SMPL lbs weights
            if use_lbs_offset:
                lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
                lbs_weights = lbs_weights.permute(0, 2, 1).repeat(smpl_poses_69.shape[0], 1, 1)
            else:
                lbs_weights = None

            _, means3D, _, transforms, translation, segs = pc.coarse_deform_c2source(
                means3D[None].repeat(smpl_shapes.shape[0], 1, 1), smpl_shapes, smpl_poses, smpl_R, smpl_Rh, smpl_Th, 
                viewpoint_camera.big_pose_smpl_param, 
                viewpoint_camera.big_pose_world_vertex[None].repeat(smpl_shapes.shape[0], 1, 1), 
                lbs_weights=lbs_weights if lbs_weights is not None else None, 
                correct_Rs=correct_Rs if correct_Rs is not None else None, 
                return_transl=return_smpl_rot, 
                view_camera=viewpoint_camera
            )

        else:
            assert False
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation
    

    if means3D.shape[0] == 1:

        means3D = means3D.squeeze()
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color


        rendered_points_image = None

        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        depth, alpha = rendered_image, rendered_image

        return {"render": rendered_image,
                "render_depth": depth,
                "render_alpha": alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "transforms": transforms,
                "translation": translation,
                "correct_Rs": correct_Rs,
                "rendered_points_image": rendered_points_image,
                }
    
    else:

        rendered_image, radii, depth, alpha = [], [], [], []

        rendered_points_image = []

        for blur_num in range(means3D.shape[0]):
            
            means3D_tmp = means3D[blur_num]
            means2D = screenspace_points
            opacity = pc.get_opacity

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = None
            if pipe.compute_cov3D_python:
                cov3D_precomp = pc.get_covariance(scaling_modifier, transforms[blur_num])  # [point_num, 6]
            else:
                scales = pc.get_scaling
                rotations = pc.get_rotation

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            if override_color is None:
                if pipe.convert_SHs_python:
                    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                    dir_pp = (means3D_tmp - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = pc.get_features
            else:
                colors_precomp = override_color

            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            rendered_image_tmp, radii_tmp = rasterizer(
                means3D = means3D_tmp,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            depth_tmp, alpha_tmp = rendered_image_tmp, rendered_image_tmp
            
            rendered_image.append(rendered_image_tmp)
            radii.append(radii_tmp)
            depth.append(depth_tmp)
            alpha.append(alpha_tmp)
        
        rendered_image = torch.stack(rendered_image)
        radii = torch.stack(radii)
        depth = torch.stack(depth)
        alpha = torch.stack(alpha)

        rendered_image_stacked = rendered_image

        if return_avg:

            rendered_image = torch.mean(rendered_image, dim=0)
            
            radii = torch.mean(radii.float(), dim=0).int()

            depth = torch.mean(depth, dim=0)
            alpha, _ = torch.max(alpha, dim=0)
        

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image, 
                "render_stacked": rendered_image_stacked, 
                "render_alpha": alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "means3D": means3D, 
                "rendered_points_image": rendered_points_image, 
                }


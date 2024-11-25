import torch
import math
import numpy as np


import numpy as np
from math import factorial


def combination(n, i):
    return factorial(n) // (factorial(i) * factorial(n - i))


def m_ij(k, i, j):
    sum_term = 0
    for s in range(j, k):
        sum_term += (-1)**(s - j) * combination(k, s - j) * (k - s - 1)**(k - 1 - i)
    return combination(k - 1, k - 1 - i) * sum_term


def generate_M_k(k):
    coef = 1 / factorial(k - 1)
    M_k = torch.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M_k[i, j] = m_ij(k, i, j)
    return coef * M_k


class PoseModel(torch.nn.Module):
    def __init__(self, smpl_shapes, smpl_poses, smpl_Rh, smpl_Th, control_points=4):
        super().__init__()

        '''
            smpl_shapes: [1, 10]
            smpl_poses: [img_num, 69]
            smpl_Rhs: [img_num, 3]
            smpl_Ths: [img_num, 3]
        '''

        self.img_num = smpl_poses.shape[0]

        # self.blur_num = 11
        self.control_points = control_points

        self.shape_dim = smpl_shapes.shape[-1]
        self.pose_dim = smpl_poses.shape[-1]
        self.Rh_dim = smpl_Rh.shape[-1]
        self.Th_dim = smpl_Th.shape[-1]

        self.register_buffer("M", generate_M_k(control_points))

        print(self.M * factorial(control_points - 1))

        low, high = 0.001, 0.05

        pose_perturb = [smpl_poses]
        Rh_perturb = [smpl_Rh]
        Th_perturb = [smpl_Th]
        
        for _ in range(1, self.control_points):
            rand_pose = (high - low) * torch.rand(*smpl_poses.shape) + low
            rand_Rh = (high - low) * torch.rand(*smpl_Rh.shape) + low
            rand_Th = (high - low) * torch.rand(*smpl_Th.shape) + low
            
            pose_perturb.append(smpl_poses + rand_pose.to(smpl_poses.device))
            Rh_perturb.append(smpl_Rh + rand_Rh.to(smpl_Rh.device))
            Th_perturb.append(smpl_Th + rand_Th.to(smpl_Th.device))

        self.smpl_shapes = torch.nn.Embedding(*smpl_shapes.shape)
        self.smpl_shapes.weight.data = torch.nn.Parameter(smpl_shapes)
        
        self.smpl_poses = torch.nn.Embedding(*pose_perturb[0].shape[:-1], self.pose_dim * self.control_points)
        self.smpl_poses.weight.data = torch.nn.Parameter(torch.cat(pose_perturb, -1))
        
        self.smpl_Rhs = torch.nn.Embedding(*Rh_perturb[0].shape[:-1], self.Rh_dim * self.control_points)
        self.smpl_Rhs.weight.data = torch.nn.Parameter(torch.cat(Rh_perturb, -1))
        
        self.smpl_Ths = torch.nn.Embedding(*Th_perturb[0].shape[:-1], self.Th_dim * self.control_points)
        self.smpl_Ths.weight.data = torch.nn.Parameter(torch.cat(Th_perturb, -1))

    def get_start_end_pose_from_indices(self, indices):
        indices = np.array(indices)

        shape = self.smpl_shapes.weight

        poses = self.smpl_poses.weight[indices]
        Rhs = self.smpl_Rhs.weight[indices]
        Ths = self.smpl_Ths.weight[indices]

        pose_list = torch.stack([poses[:, i*self.pose_dim:(i+1)*self.pose_dim] for i in range(self.control_points)], dim=1)
        Rh_list = torch.stack([Rhs[:, i*self.Rh_dim:(i+1)*self.Rh_dim] for i in range(self.control_points)], dim=1)
        Th_list = torch.stack([Ths[:, i*self.Th_dim:(i+1)*self.Th_dim] for i in range(self.control_points)], dim=1)

        return shape, pose_list, Rh_list, Th_list
    
    def get_intermediate_poses_from_indices(self, indices, blur_num):

        shape, pose_basis, Rh_basis, Th_basis = self.get_start_end_pose_from_indices(indices)

        timesteps = torch.arange(blur_num) / (blur_num - 1)  # [blur_num]
        timesteps = timesteps.to(shape.device, dtype=shape.dtype)

        pos_0 = torch.where(timesteps == 0)
        timesteps[pos_0] = timesteps[pos_0] + 0.000001
        pos_1 = torch.where(timesteps == 1)
        timesteps[pos_1] = timesteps[pos_1] - 0.000001

        timesteps = timesteps[..., None]  # [blur_num, 1]

        timesteps_matrix = [torch.ones_like(timesteps)]
        for t in range(1, self.control_points):
            timesteps_matrix.append(timesteps**t)
        timesteps_matrix = torch.stack(timesteps_matrix, dim=-1)  # [blur_num, 1, control_points]

        coeffs = torch.matmul(timesteps_matrix, self.M[None, :]).squeeze(-2)  # [blur_num, control_points]

        coeffs = coeffs[None, :]

        coeffs_list = torch.split(coeffs, 1, dim=-1)  # [blur_num, control_points]

        pose_t, Rh_t, Th_t = 0., 0., 0.

        for indx in range(len(coeffs_list)):
            pose_t += coeffs_list[indx] * pose_basis[:, indx][:, None, :]
            Rh_t += coeffs_list[indx] * Rh_basis[:, indx][:, None, :]
            Th_t += coeffs_list[indx] * Th_basis[:, indx][:, None, :]

        shape_t = shape[None].repeat(pose_t.shape[0], pose_t.shape[1], 1)

        return shape_t, pose_t, Rh_t, Th_t



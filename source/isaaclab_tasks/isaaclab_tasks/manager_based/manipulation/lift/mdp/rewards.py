# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets.articulation.articulation import Articulation
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def ee_height_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tcp_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "wrist_3_link",
) -> torch.Tensor:
    """
    Penalizes the robot's tool-center-point (TCP) for being below a certain height.
    """
    # 1. 从场景中获取机器人对象
    robot = env.scene[robot_cfg.name]

    # 2. 获取末端执行器连杆在世界坐标系下的位姿
    ee_link_pos_w = robot.data.body_pos_w[:, robot.body_names.index(ee_body_name)]
    ee_link_quat_w = robot.data.body_quat_w[:, robot.body_names.index(ee_body_name)]

    # 3. 计算工具中心点(TCP)在世界坐标系下的位置
    
    # [修正] 创建一个形状为 (num_envs, 3) 的偏移向量批次
    # a. 先创建一个单独的向量
    tcp_offset_tensor = torch.tensor(tcp_offset, device=env.device)
    # b. 使用 .expand() 将其扩展到与环境数量匹配的批次，这非常高效
    tcp_offset_batch = tcp_offset_tensor.expand(env.num_envs, -1)

    # c. 现在，quat_apply 的两个输入维度完全匹配:
    #    ee_link_quat_w: (4096, 4)
    #    tcp_offset_batch: (4096, 3)
    offset_w = quat_apply(ee_link_quat_w, tcp_offset_batch)
    
    # d. 将世界坐标系下的偏移量加到连杆的位置上
    tcp_pos_w = ee_link_pos_w + offset_w

    # 4. 提取TCP的高度
    tcp_height = tcp_pos_w[:, 2]

    # 5. 计算并返回惩罚
    penalty = torch.where(tcp_height < minimal_height, minimal_height - tcp_height, 0.0)
    return penalty



def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

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

# def encourage_grasp_at_target(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg,
#     tcp_offset: tuple[float, float, float],
#     ee_body_name: str,
#     distance_threshold: float,
# ) -> torch.Tensor:
#     """
#     Penalizes the agent for NOT closing the gripper when the TCP is close to the object.
#     This indirectly encourages the grasping action at the right moment.
#     """

#     # -- 1. 提取所需的对象和数据
#     robot: Articulation = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]

#     # -- 2. 计算TCP的真实世界位置 (与ee_height_penalty中的逻辑完全相同)
#     ee_link_idx = robot.body_names.index(ee_body_name)
#     ee_link_pos_w = robot.data.body_pos_w[:, ee_link_idx]
#     ee_link_quat_w = robot.data.body_quat_w[:, ee_link_idx]
    
#     tcp_offset_tensor = torch.tensor(tcp_offset, device=env.device)
#     tcp_offset_batch = tcp_offset_tensor.expand(env.num_envs, -1)
#     offset_w = quat_apply(ee_link_quat_w, tcp_offset_batch)
#     tcp_pos_w = ee_link_pos_w + offset_w
    
#     object_pos_w = object.data.root_pos_w

#     # -- 3. 检查距离条件：TCP是否足够靠近物体
#     distance = torch.norm(tcp_pos_w - object_pos_w, dim=1)
#     is_close_to_target = distance < distance_threshold

#     # -- 4. 检查智能体的意图：是否正在命令夹爪闭合
#     last_actions = env.action_manager.action
#     # print(f"Last actions shape: {last_actions.shape}, values: {last_actions}")    
#     gripper_action = last_actions[:, -1] # 夹爪动作是最后一个分量
#     is_commanding_close = gripper_action > 0.5

#     # -- 5. [核心逻辑] 计算惩罚
#     # 当“靠近物体”为True，但“命令闭合”为False时，我们施加惩罚
#     # (is_close_to_target & ~is_commanding_close)
#     # torch.where 会在条件为True时返回-1.0（惩罚），否则返回0.0（无操作）
#     penalty = torch.where(is_close_to_target & ~is_commanding_close, -1.0, 0.0)

#     return penalty

def cylinder_is_grasped_and_controlled(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    left_finger_body_name: str,
    right_finger_body_name: str,
    gripper_joint_names: list[str],
    open_angle_threshold: float,
    close_angle_threshold: float,
    grasp_distance_threshold: float,
    height_difference_threshold: float,
) -> torch.Tensor:
    """
    Reward for a robust grasp, verified by 3 physical facts:
    1. 3D proximity between TCP and the object.
    2. Gripper is in a "partially closed" state.
    3. The object's height is being controlled by the gripper (small Z distance).
    """
    # -- 1. 提取所需的对象和数据
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # -- 2. [事实一 & 三] 检查距离和高度控制
    # 获取指尖中心点(TCP)的位置
    left_finger_idx = robot.body_names.index(left_finger_body_name)
    right_finger_idx = robot.body_names.index(right_finger_body_name)
    left_finger_pos_w = robot.data.body_pos_w[:, left_finger_idx]
    right_finger_pos_w = robot.data.body_pos_w[:, right_finger_idx]
    tcp_pos_w = (left_finger_pos_w + right_finger_pos_w) / 2.0
    
    object_pos_w = object.data.root_pos_w

    # 条件1: 检查TCP和物体的三维距离
    distance_3d = torch.norm(tcp_pos_w - object_pos_w, dim=1)
    is_3d_close = distance_3d < grasp_distance_threshold

    # 条件2: [新增] 检查TCP和物体的高度(Z轴)差
    height_difference = torch.abs(tcp_pos_w[:, 2] - object_pos_w[:, 2])
    is_height_controlled = height_difference < height_difference_threshold

    # -- 3. [事实二] 检查夹爪的真实物理状态：是否处于“半开合”
    gripper_joint_indices = [robot.joint_names.index(name) for name in gripper_joint_names]
    gripper_joints_pos = robot.data.joint_pos[:, gripper_joint_indices]
    
    not_fully_closed = torch.all(gripper_joints_pos > open_angle_threshold, dim=1)
    not_fully_open = torch.all(gripper_joints_pos < close_angle_threshold, dim=1)
    is_gripper_partially_closed = not_fully_closed & not_fully_open

    # -- 4. 组合最终奖励
    # 必须同时满足所有三个物理事实
    grasp_reward = torch.where(
        is_3d_close & is_height_controlled & is_gripper_partially_closed, 
        1.0, 
        0.0
    )

    return grasp_reward

# def ee_height_penalty(
#     env: ManagerBasedRLEnv,
#     minimal_height: float,
#     tcp_offset: tuple[float, float, float],
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ee_body_name: str = "wrist_3_link",
# ) -> torch.Tensor:
#     """
#     Penalizes the robot's tool-center-point (TCP) for being below a certain height.
#     """
#     # 1. 从场景中获取机器人对象
#     robot = env.scene[robot_cfg.name]

#     # 2. 获取末端执行器连杆在世界坐标系下的位姿
#     ee_link_pos_w = robot.data.body_pos_w[:, robot.body_names.index(ee_body_name)]
#     ee_link_quat_w = robot.data.body_quat_w[:, robot.body_names.index(ee_body_name)]

#     # 3. 计算工具中心点(TCP)在世界坐标系下的位置
    
#     # [修正] 创建一个形状为 (num_envs, 3) 的偏移向量批次
#     # a. 先创建一个单独的向量
#     tcp_offset_tensor = torch.tensor(tcp_offset, device=env.device)
#     # b. 使用 .expand() 将其扩展到与环境数量匹配的批次，这非常高效
#     tcp_offset_batch = tcp_offset_tensor.expand(env.num_envs, -1)

#     # c. 现在，quat_apply 的两个输入维度完全匹配:
#     #    ee_link_quat_w: (4096, 4)
#     #    tcp_offset_batch: (4096, 3)
#     offset_w = quat_apply(ee_link_quat_w, tcp_offset_batch)
    
#     # d. 将世界坐标系下的偏移量加到连杆的位置上
#     tcp_pos_w = ee_link_pos_w + offset_w

#     # 4. 提取TCP的高度
#     tcp_height = tcp_pos_w[:, 2]

#     # 5. 计算并返回惩罚
#     penalty = torch.where(tcp_height < minimal_height, minimal_height - tcp_height, 0.0)
#     return penalty
# )

def object_relative_zero_momentum_after_lift(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg,
    ee_body_name: str,
    minimal_height: float, 
    std: float,
) -> torch.Tensor:
    """
    奖励物体相对于夹爪的线速度接近于零，但仅在物体被举起后生效。
    
    参数:
        ee_body_name (str): 机械臂末端执行器（例如 'gripper_link'）的名称。
        minimal_height (float): 判定物体被举起的最低高度 (m)。
        std (float): 负指数函数的标准差，控制奖励的敏感度。
    """
    # 提取所需的对象和数据
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # 1. 获取 EE 速度
    # 查找 body 索引，然后访问 body 状态 (Isaac Lab标准做法)
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_vel_w = robot.data.body_state_w[:, ee_idx, 7:10]
    
    # 2. 获取物体速度
    object_vel_w = object.data.root_lin_vel_w
    
    # 3. 计算相对速度: v_rel = v_object - v_ee
    relative_vel = object_vel_w - ee_vel_w
    
    # 4. 计算相对速度的平方模长 (L2范数的平方)
    vel_sq_norm = torch.sum(torch.square(relative_vel), dim=-1)
    
    # 5. 负指数奖励
    reward = torch.exp(-vel_sq_norm / (2 * std**2))
    
    # 6. 应用门控：只有在物体被举起后，奖励才生效
    object_height = object.data.root_pos_w[:, 2]
    is_lifted = object_height > minimal_height
    # 使用 .float() 确保类型兼容性
    reward = reward * is_lifted.float()
    
    return reward

def grasp_quality_alignment(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg, 
    object_cfg: SceneEntityCfg, 
    ee_body_name: str,
    distance_threshold: float,
    std_pos: float, 
    std_ori: float,
    minimal_height: float = 0.04,  # 新增：高度判定阈值
) -> torch.Tensor:
    """
    奖励夹爪中心对齐物体中心，并且夹爪的Z轴与物体Z轴对齐。
    逻辑：仅在物体被举起后激活，作为“抓取质量”的后期评价。
    """
    # 提取机器人和物体
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # 1. 判定高度：只有物体高度超过 minimal_height 时才给分
    object_pos_w = object.data.root_pos_w
    is_lifted = object_pos_w[:, 2] > minimal_height
    
    # 获取 EE 索引和姿态
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx]
    
    # --- 位置对齐计算 ---
    distance_sq_norm = torch.sum(torch.square(object_pos_w - ee_pos_w), dim=-1)
    reward_pos = torch.exp(-distance_sq_norm / (2 * std_pos**2))
    
    # --- 姿态对齐计算 ---
    object_quat_w = object.data.root_quat_w
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    # 提取 Z 轴向量
    ee_z_axis = quat_apply(ee_quat_w, world_z_axis)
    object_z_axis = quat_apply(object_quat_w, world_z_axis) 

    # 点积 (cos(theta))
    dot_product = torch.sum(ee_z_axis * object_z_axis, dim=-1)
    reward_ori = torch.exp(-torch.square(1.0 - dot_product) / (2 * std_ori**2))
    
    # 姿态门控：EE 离物体近才算姿态
    is_close = distance_sq_norm < distance_threshold**2
    reward_ori_gated = reward_ori * is_close.float()
    
    # 计算总对齐得分
    total_alignment_reward = reward_pos + reward_ori_gated
    
    # [核心修改]：只有 lift 成功后才返回得分，否则返回 0
    return total_alignment_reward * is_lifted.float()

def reward_predictive_interception(
    env: ManagerBasedRLEnv,
    dt: float,
    alpha: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    ee_body_name: str,
    cutoff_height: float = 0.03  # 截止高度/最小抓取高度
) -> torch.Tensor:
    """
    预测拦截奖励 (R_pred) - 当物体被抓取后奖励归零。
    """
    # 1. 获取物体状态
    object_state = env.scene[object_cfg.name].data.root_state_w
    p_obj = object_state[:, 0:3]
    v_obj = object_state[:, 7:10]
    obj_height = p_obj[:, 2] # 获取 Z 轴高度

    # 2. 获取 EE 状态
    robot = env.scene[robot_cfg.name]
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    p_ee = robot.data.body_state_w[:, ee_idx, 0:3]

    # 3. 计算预测位置
    p_pred = p_obj + v_obj * dt

    # 4. 计算原始高斯奖励
    error_sq = torch.sum(torch.square(p_ee - p_pred), dim=-1)
    raw_reward = torch.exp(-alpha * error_sq)

    # 5. [核心逻辑] 应用抓取掩码
    # 判断是否被提起 (is_lifted 的逻辑与你的 object_is_lifted 相同)
    is_lifted = (obj_height > cutoff_height)
    
    # 如果 is_lifted 为 True，最终奖励为 0.0；否则保持 raw_reward
    final_reward = torch.where(is_lifted, torch.zeros_like(raw_reward), raw_reward)

    return final_reward.view(-1)

def object_ee_pre_distance(
    env: ManagerBasedRLEnv,
    std: float,
    dt: float = 0.1,  # 预判时间步，建议设定在 0.1 - 0.2 之间
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """奖励末端执行器接近物体的预测位置（仅在XY平面做速度补偿）。"""
    
    # 1. 获取资产
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 2. 获取当前状态
    curr_pos_w = object.data.root_pos_w    # (num_envs, 3)
    curr_vel_w = object.data.root_vel_w[:, 0:3]  # (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :] # (num_envs, 3)

    # 3. [核心修改] 仅在 XY 平面进行位置预测
    # 这样当物体被提起产生 Z 向速度时，目标点不会飞到夹爪上方
    predict_pos_w = curr_pos_w.clone()
    predict_pos_w[:, :2] += curr_vel_w[:, :2] * dt 

    # 4. 计算末端到预测位置的欧几里得距离
    dist = torch.norm(predict_pos_w - ee_w, dim=1)

    # 5. 使用 tanh 核函数映射到 (0, 1]
    # std 越大，奖励曲线越平缓，容错率越高
    return 1 - torch.tanh(dist / std)

def reward_velocity_matching(
    env: ManagerBasedRLEnv,
    beta: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    ee_body_name: str,
    direction_axis: str = None,
    cutoff_height: float = 0.03 # 截止高度/最小抓取高度
) -> torch.Tensor:
    """
    速度匹配奖励 (R_vel) - 当物体被抓取后奖励归零。
    """
    # 1. 获取物体状态
    object_state = env.scene[object_cfg.name].data.root_state_w
    v_obj = object_state[:, 7:10]
    p_obj = object_state[:, 0:3] # 获取 Z 轴高度

    # 2. 获取 EE 速度
    robot = env.scene[robot_cfg.name]
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    v_ee = robot.data.body_state_w[:, ee_idx, 7:10]

    # 3. 处理速度差异
    if direction_axis in ["x", "y", "z"]:
        idx = ["x", "y", "z"].index(direction_axis)
        diff_sq = torch.square(v_ee[:, idx] - v_obj[:, idx])
    else:
        diff_sq = torch.sum(torch.square(v_ee - v_obj), dim=-1)

    # 4. 计算原始高斯奖励
    raw_reward = torch.exp(-beta * diff_sq)

    # 5. [核心逻辑] 应用抓取掩码
    is_lifted = (p_obj[:, 2] > cutoff_height)
    final_reward = torch.where(is_lifted, torch.zeros_like(raw_reward), raw_reward)

    return final_reward.view(-1)
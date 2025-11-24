# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height

def root_drop_after_lift(
    env: ManagerBasedRLEnv,  
    lift_threshold: float,
    drop_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    状态终止条件：判断物体是否在达到 lift_threshold 后又掉落到 drop_threshold 以下。
    
    注意：因为这是一个独立的函数，我们将状态变量 _custom_max_height 挂载到 env 对象上。
    """
    # 1. 获取物体当前高度
    object_pos_w = env.scene[asset_cfg.name].data.root_pos_w
    current_height = object_pos_w[:, 2]

    # 2. 初始化状态容器 (挂载到 env 上)
    # 如果 env 上还没有这个变量，我们就创建一个
    if not hasattr(env, "_custom_object_max_height"):
        env._custom_object_max_height = torch.zeros(env.num_envs, device=env.device)

    # 3. 处理回合重置 (Reset Logic)
    # 关键点：当环境重置时(episode_length=0或刚开始)，我们需要把记录的最高高度清零。
    # 在 step() 内部，计算 termination 时 episode_length 通常至少为 1。
    # 所以我们判断：如果 episode_length <= 1，说明是新回合的开始，重置该环境的 max_height。
    reset_indices = (env.episode_length_buf <= 1)
    if torch.any(reset_indices):
        # 将重置了的环境的 max_height 设为当前高度（通常是桌面高度）
        env._custom_object_max_height[reset_indices] = current_height[reset_indices]

    # 4. 更新当前回合中达到的最高高度
    env._custom_object_max_height = torch.max(env._custom_object_max_height, current_height)
    
    # 5. 终止条件判断
    
    # 条件 A: 物体是否曾被成功举起 (历史最高高度 > 举起阈值)
    has_been_lifted_mask = (env._custom_object_max_height > lift_threshold)

    # 条件 B: 物体是否当前掉落到安全高度以下 (当前高度 < 掉落阈值)
    is_dropped_mask = (current_height < drop_threshold)

    # 终止：当条件 A 和 B 都满足时 (即先被举起，后又掉落)
    is_done = torch.logical_and(has_been_lifted_mask, is_dropped_mask)

    # 返回布尔值 (Isaac Lab 会自动处理类型转换)
    return is_done

def object_out_of_workspace(
    env: ManagerBasedRLEnv,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    当物体移出指定的工作空间范围（边界框）时终止回合 (已修正为局部坐标系)。
    """
    # 获取物体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 1. 获取世界坐标系下的位置
    root_pos_w = asset.data.root_pos_w
    
    # 2. 获取所有环境的坐标原点 (形状: [num_envs, 3])
    env_origins = env.scene.env_origins
    
    # 3. ***** 关键修正：通过减法计算出局部坐标系下的位置 *****
    root_pos = root_pos_w - env_origins
    
    # 初始化一个全为 False 的 tensor (即默认不终止)
    out_of_bounds = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 检查 X 轴限制 (其余逻辑保持不变，但使用的是新的 root_pos)
    if x_limits is not None:
        out_of_x = (root_pos[:, 0] < x_limits[0]) | (root_pos[:, 0] > x_limits[1])
        out_of_bounds |= out_of_x

    # 检查 Y 轴限制
    if y_limits is not None:
        out_of_y = (root_pos[:, 1] < y_limits[0]) | (root_pos[:, 1] > y_limits[1])
        out_of_bounds |= out_of_y

    # 检查 Z 轴限制
    if z_limits is not None:
        out_of_z = (root_pos[:, 2] < z_limits[0]) | (root_pos[:, 2] > z_limits[1])
        out_of_bounds |= out_of_z

    return out_of_bounds

"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques. If they are not, it means that clipping has occurred.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = ~torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

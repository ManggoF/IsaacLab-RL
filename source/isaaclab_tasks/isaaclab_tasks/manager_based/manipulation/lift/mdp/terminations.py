# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def root_drop_after_lift(
    env: ManagerBasedRLEnv,  # <--- 注意：这里去掉了 self
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

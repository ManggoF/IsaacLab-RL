# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import torch
import gymnasium as gym

# --- ADDED FOR PLOTTING ---
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
# --------------------------

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    timestep = 0

    ## --- ADDED FOR PLOTTING (7 DOF: 6 Arm + 1 Gripper) --- ##
    try:
        robot_entity = env.unwrapped.scene["robot"]
    except KeyError:
        print("[ERROR] Could not find 'robot' in scene entities.")
        raise

    # [核心配置]
    ARM_DIM = 6      # 前6个是机械臂
    GRIPPER_DIM = 1  # 第7个是夹爪(Index 6)
    
    # 打印真实维度供调试
    real_dim = robot_entity.data.joint_pos.shape[-1]
    print(f"\n[INFO] Real Robot DOF: {real_dim}. We will plot indices 0-5 (Arm) and 6 (Gripper).\n")

    history_len = 200
    
    # 两个独立的队列列表
    arm_history = [deque(maxlen=history_len) for _ in range(ARM_DIM)]
    gripper_history = [deque(maxlen=history_len) for _ in range(GRIPPER_DIM)]

    plt.ion()
    # 创建两个子图：上面画机械臂，下面画夹爪
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    ax_arm, ax_gripper = axes[0], axes[1]
    
    fig.suptitle("Real-time Joint Positions", fontsize=16)

    # --- 配置机械臂子图 ---
    ax_arm.set_title(f"Arm Joints (Indices 0-5)")
    ax_arm.set_ylabel("Position (rad)")
    arm_lines = [ax_arm.plot([], [], label=f'Arm J{i+1}')[0] for i in range(ARM_DIM)]
    ax_arm.legend(loc='upper right', fontsize='small', ncol=3)
    ax_arm.grid(True)

    # --- 配置夹爪子图 ---
    ax_gripper.set_title(f"Gripper Main Joint (Index 6)")
    ax_gripper.set_ylabel("Position (rad/m)")
    ax_gripper.set_xlabel("Time Steps")
    gripper_lines = [ax_gripper.plot([], [], label=f'Gripper', color='orange')[0] for i in range(GRIPPER_DIM)]
    ax_gripper.legend(loc='upper right')
    ax_gripper.grid(True)
    ## ---------------------------------------------------- ##


    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            
            # 获取所有关节位置
            full_joint_pos = robot_entity.data.joint_pos[0].cpu().numpy()
            
            # 1. 更新机械臂数据 (前6个)
            for i in range(ARM_DIM):
                if i < len(full_joint_pos):
                    arm_history[i].append(full_joint_pos[i])
            
            # 2. 更新夹爪数据 (第7个，即 index 6)
            gripper_idx = 6
            if gripper_idx < len(full_joint_pos):
                gripper_history[0].append(full_joint_pos[gripper_idx])
            
            # --- 绘图更新 ---
            
            # 机械臂曲线
            for i, line in enumerate(arm_lines):
                line.set_data(np.arange(len(arm_history[i])), arm_history[i])
            ax_arm.relim()
            ax_arm.autoscale_view()
            
            # 夹爪曲线
            for i, line in enumerate(gripper_lines):
                line.set_data(np.arange(len(gripper_history[i])), gripper_history[i])
            ax_gripper.relim()
            ax_gripper.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001) 
            
            obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    plt.close(fig)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
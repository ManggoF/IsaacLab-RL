# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg


from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # 加入目标坐标系的可视化
    # object_frame: FrameTransformerCfg = MISSING
    world_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    # # plane
    # plane = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
    #     spawn=GroundPlaneCfg(),
    # )

    # # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    # 为每一个环境创建一个独立的、看不见的、光滑的刚体盒子
    # conveyor_surface = RigidObjectCfg(
    #     # 关键1: 使用 {ENV_REGEX_NS}，确保每个环境都有自己的传送带
    #     prim_path="{ENV_REGEX_NS}/ConveyorSurface",
        
    #     # 关键2: 使用 RigidObjectCfg 专属的 InitialStateCfg
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0, 0.0],lin_vel=[0.0, 0.0, 0.0],), 
    #     spawn=CuboidCfg(
    #         size=(1, 1, 0.01),
    #         rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False,disable_gravity=True),
    #         mass_props=MassPropertiesCfg(mass=100.0),
    #         collision_props=CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=RigidBodyMaterialCfg(
    #             static_friction=0.8,
    #             dynamic_friction=0.5,
    #             restitution=0.0,
    #         ),
    #         # 关键3: 使用唯一正确的 "visible" 参数来实现隐形
    #         visible=True,
    #     ),
    # )
    # -- 定义传送带平面 --
    conveyor_surface = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorSurface",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0, 0.0]),
        spawn=CuboidCfg(
            size=(1.0, 1.0, 0.01),
            # 物理材质保持不变
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            ),
            # [关键] 将物体设置为运动学模式
            rigid_props=RigidBodyPropertiesCfg(
                # 启用运动学模式，这将使其不受外力影响
                kinematic_enabled=True,
                # 通常运动学物体不需要重力
                disable_gravity=True, 
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, -0.63]),
        spawn=UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR5/table.usd",)
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.63]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.6, 0.6), pos_y=(0.0, 0.0), pos_z=(0.35, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # 修改这一行，让智能体只能“看到”目标位置
        target_object_position = ObsTerm(func=mdp.generated_command_position, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),  # 修改为圆柱体
        },
    )

    move_object_smartly = EventTerm(
        func=mdp.move_object_unless_lifted, # <<--- 使用新的、基于高度的函数
        mode="interval",
        # 为了每一步都执行，将最小和最大间隔都设置为环境的步长时间
        # 假设你的 sim.dt=0.01, decimation=2, 那么 env.step_dt = 0.02
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "speed_range": (0.0, 0.3),
            "threshold_steps": 80,
            # [关键] 设置一个判断“被举起”的高度阈值 (m)
            # 这个值应该比物体在传送带上的高度略高一点
            # 例如，如果物体在传送带上时高度是0.03m
            "lift_height_threshold": 0.04, 
        },
    )
    
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="usd",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (0.8, 1.2), # 在 0.8 到 1.2 之间均匀缩放
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.12}, weight=5.0)  ## .25
    
    grasping_cylinder = RewTerm(
    func=mdp.cylinder_is_grasped_and_controlled, # <<--- 使用最终的、无懈可击的函数
    weight=50.0,
    params={
        "robot_cfg": SceneEntityCfg("robot"),
        "object_cfg": SceneEntityCfg("object"),
        "left_finger_body_name": "robotiq_85_left_finger_tip_link",
        "right_finger_body_name": "robotiq_85_right_finger_tip_link",
        "gripper_joint_names": [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint"
        ],
        # [关键] 定义“半开合”的范围，这需要你通过实验来精确测量
        "open_angle_threshold": math.radians(10.0),
        "close_angle_threshold": math.radians(35.0),
        
        # [关键] 定义三维空间中的接近阈值 (m)
        # 这个值应该比你的夹爪内部宽度略大
        "grasp_distance_threshold": 0.03,

        # [关键] 定义TCP和物体的高度差阈值 (m)
        # 一个非常小的值，表示物体正被夹爪的中心“托住”
        "height_difference_threshold": 0.01, #     cylinder:5毫米   cube:0.01
    },
)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=100.0)  #cube0.04/cylinder0.03

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-3e-5)  #-2e-5

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.01, "asset_cfg": SceneEntityCfg("object")}
    )

    object_drop_after_lift_and_drop = DoneTerm(
        func=mdp.root_drop_after_lift, 
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # 判定为“成功举起”的高度 (需大于物体在传送带上的高度)
            "lift_threshold": 0.06, 
            # 判定为“掉落”的高度 (需接近于地面的高度，例如 1cm)
            "drop_threshold": 0.02 
        }
    )

    # object_out_of_bounds = DoneTerm(
    #     func=mdp.object_out_of_workspace, # 指向刚才写的函数
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "x_limits": (0.35, 0.6),   # 限制 X 轴在 0.35 到 0.6 米之间
    #         "y_limits": (-0.5, 0.5),  # 限制 Y 轴在 -0.5 到 0.5 米之间
    #         "z_limits": (0.0, 1.0),   # (可选) 限制高度不超过 1 米
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    fade_out_lifting_reward = CurrTerm(
        func=mdp.modify_reward_weight_linearly, # <<--- 使用我们新的平滑函数
        params={
            "term_name": "lifting_object",
            "start_weight":100.0,  # 衰减前的权重
            "end_weight": 15.0,     # 衰减后的权重
            # 定义衰减过程的起止步数
            # 总步数约75000(iteration=103时，step=2500 )
            "start_step": 15000,
            "end_step": 40000,
        }
    )

    target_tracking_reward = CurrTerm(
        func=mdp.modify_reward_weight_linearly,
        params={
            "term_name": "object_goal_tracking",
            "start_weight": 16.0,
            "end_weight": 88.0,
            "start_step": 20000,
            "end_step": 70000,
        }
    )

    target_fine_tracking_reward = CurrTerm(
        func=mdp.modify_reward_weight_linearly,
        params={
            "term_name": "object_goal_tracking_fine_grained",
            "start_weight": 5.0,
            "end_weight": 80.0,
            "start_step": 50000,
            "end_step": 70000,
        }
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight_linearly, 
        params={
            "term_name": "action_rate",                                         
            "start_weight": -3e-5,
            "end_weight": -1.0,
            "start_step": 24000,
            "end_step": 48000,
        }
    )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight_linearly, 
    #     params={
    #         "term_name": "joint_vel", 
    #         "start_weight": -1e-5, 
    #         "end_weight": -0.1,
    #         "start_step": 30000,
    #         "end_step": 70000,
    #     }
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

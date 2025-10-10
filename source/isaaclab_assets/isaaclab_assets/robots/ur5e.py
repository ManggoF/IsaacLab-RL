# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR5e.

The following configurations are available:

* :obj:`UR5E_CFG`: Universal Robots UR5e with Robotiq 2F-85 gripper.
* :obj:`UR5E_HIGH_PD_CFG`: Universal Robots UR5e with Robotiq 2F-85 gripper with stiffer PD control.

"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e_2f85.usd",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/ur5_2f_v4.5.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # joint_pos={
        #     "shoulder_pan_joint": 0.0,
        #     "shoulder_lift_joint": -1.2,  # 抬起一点
        #     "elbow_joint": 1.57,        # 弯曲手肘
        #     "wrist_1_joint": -2.355,      # 调整手腕姿态
        #     "wrist_2_joint": -1.57,       # 让手腕朝下
        #     "wrist_3_joint": 0.0,
        #     "finger_joint": 0.785,        # 夹爪张开
        # },
        joint_pos={
            "shoulder_pan_joint": math.radians(-7.0),             # 132.0°
            "shoulder_lift_joint": math.radians(-85.0),           # -8.9°
            "elbow_joint": math.radians(113.0),                   # -86.3°
            "wrist_1_joint": math.radians(-117.0),                # -104.0°
            "wrist_2_joint": math.radians(-80.0),                 # -1.0°
            "wrist_3_joint": math.radians(-8.0),                  # 33.0°
            "finger_joint": math.radians(40.0),                   # 夹爪张开0.785
        },
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim=0.5,  # 0.5,
            effort_limit_sim=300.0,  # 300
            stiffness=2000.0,        # 2000
            damping=100.0,           # 100
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=5.0,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Universal Robots UR5e robot with Robotiq 2F-85 gripper."""


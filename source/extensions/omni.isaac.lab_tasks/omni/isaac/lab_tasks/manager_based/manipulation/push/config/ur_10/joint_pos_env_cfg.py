# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.push.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg


##
# Added for activate_contact_sensors
##
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from omni.isaac.lab.sensors import ContactSensorCfg

from pxr import Usd, UsdGeom

##
# Environment configuration
##


@configclass
class UR10PushEnvCfg(PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = ArticulationCfg(
                                spawn=sim_utils.UsdFileCfg(
                                    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                        disable_gravity=False,
                                        max_depenetration_velocity=5.0,
                                    ),
                                    activate_contact_sensors=True,
                                ),
                                init_state=ArticulationCfg.InitialStateCfg(
                                    joint_pos={
                                        "shoulder_pan_joint": 0.0,
                                        "shoulder_lift_joint": -1.712,
                                        "elbow_joint": 1.712,
                                        "wrist_1_joint": 0.0,
                                        "wrist_2_joint": 0.0,
                                        "wrist_3_joint": 0.0,
                                    },
                                ),
                                actuators={
                                    "arm": ImplicitActuatorCfg(
                                        joint_names_expr=[".*"],
                                        velocity_limit=100.0,
                                        effort_limit=87.0,
                                        stiffness=800.0,
                                        damping=40.0,
                                    ),
                                },
                                prim_path="{ENV_REGEX_NS}/Robot"
                                )
        
        
        
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR10PushEnvCfg_PLAY(UR10PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

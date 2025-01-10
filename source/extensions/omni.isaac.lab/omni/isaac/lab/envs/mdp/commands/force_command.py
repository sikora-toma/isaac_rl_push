# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for force tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import UniformForceCommandCfg


class UniformForceCommand(CommandTerm):
    """Command generator for generating force commands uniformly.
    TODO update
    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformForceCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformForceCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.force_command_b = torch.zeros(self.num_envs, 3, device=self.device)
       
        # -- metrics
        # self.metrics["error_force"] = torch.zeros(self.num_envs, device=self.device)
        

    def __str__(self) -> str:
        msg = "UniformForceCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired force command. Shape is (num_envs, 3).
        Returns the x, y, z command in the body frame
        """
        return self.force_command_b
    
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        # self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        r = torch.empty(len(env_ids), device=self.device)
        self.force_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.force_x) 
        self.force_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.force_y) 
        self.force_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.force_z) 

    def _update_command(self):
        """Re-target the position command to the current root state."""
        pass

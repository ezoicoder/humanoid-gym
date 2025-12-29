# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain


class XBotLFreeEnv(LeggedRobot):
    '''
    XBotLFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        
        # Ensure num_height_points is set even if measure_heights is False (for dimension consistency)
        if not hasattr(self, 'num_height_points'):
            # Calculate expected height points from config
            num_x = len(cfg.terrain.measured_points_x) if hasattr(cfg.terrain, 'measured_points_x') else 15
            num_y = len(cfg.terrain.measured_points_y) if hasattr(cfg.terrain, 'measured_points_y') else 15
            self.num_height_points = num_x * num_y
        
        print("fuck num_height_points:", self.num_height_points)

        # Initialize measured_heights with correct shape to avoid dimension mismatch
        if cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        else:
            self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        
        # Success tracking buffers (成功只记录，不终止！)
        self.episode_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_distance_traveled = torch.zeros(self.num_envs, device=self.device)
        self.episode_success_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Curriculum tracking: consecutive successes (for "three in a row" rule)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Terrain type identification (根据列索引判断地形类型)
        # 注意：父类已经初始化了 self.terrain_types (列索引)，不要覆盖！
        # 我们创建 self.actual_terrain_types 来存储地形类型编号
        self.actual_terrain_types = None
        self._init_terrain_types()
        
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()
    
    def _init_terrain_types(self):
        """
        从地形对象读取每个环境的地形类型（鲁棒！不硬编码！）
        
        关键理解：
        - 父类 LeggedRobot 已经初始化了 self.terrain_types（列索引）
        - 父类在 _update_terrain_curriculum() 中需要用它来索引 terrain_origins
        - 因此我们**不能覆盖** self.terrain_types，必须创建新变量！
        
        解决方案：
        - 保留父类的 self.terrain_types (列索引，用于 curriculum 更新)
        - 创建新的 self.actual_terrain_types (地形类型编号，用于判断)
        
        地形类型编号（由 Terrain.make_terrain() 生成）：
        - 0: flat, 1: obstacles, 2: random, 3: slope+, 4: slope-
        - 5: stairs+, 6: stairs-, 7: balancing_beams, 8: stones_everywhere
        """
        if not hasattr(self, 'terrain') or self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            print("WARNING: No terrain or not using heightfield/trimesh. Terrain-specific features disabled.")
            self.actual_terrain_types = None
            return
        
        # 检查地形对象是否有 terrain_type_map
        if not hasattr(self.terrain, 'terrain_type_map'):
            print("WARNING: terrain object does not have terrain_type_map! Using fallback.")
            self.actual_terrain_types = None
            return
        
        # 检查父类是否已初始化 terrain_types
        if not hasattr(self, 'terrain_types') or self.terrain_types is None:
            print("WARNING: Parent class terrain_types not initialized!")
            self.actual_terrain_types = None
            return
        
        # ⚠️ 不要覆盖 self.terrain_types！父类需要它来索引 terrain_origins！
        # 创建新变量来存储实际的地形类型编号
        self.actual_terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 从 terrain_type_map 读取实际的地形类型
        # 由于 curriculum 模式下同一列的所有行都是同一类型，我们查第 0 行即可
        for i in range(self.num_envs):
            col_idx = self.terrain_types[i].item()  # 父类的列索引
            # 读取该列的地形类型（查第 0 行，因为同一列都一样）
            actual_type = self.terrain.terrain_type_map[0, col_idx]
            self.actual_terrain_types[i] = actual_type
        
        # 打印统计信息
        unique_types = torch.unique(self.actual_terrain_types)
        print(f"\n{'='*60}")
        print("TERRAIN TYPE DISTRIBUTION")
        print(f"{'='*60}")
        terrain_names = {
            0: 'flat', 1: 'obstacles', 2: 'random', 3: 'slope+', 4: 'slope-',
            5: 'stairs+', 6: 'stairs-', 7: 'balancing_beams', 8: 'stones_everywhere',
            9: 'stepping_stones'
        }
        for t in unique_types:
            count = (self.actual_terrain_types == t).sum().item()
            name = terrain_names.get(t.item(), f'unknown_{t.item()}')
            print(f"  Type {t.item()} ({name}): {count} environments")
        print(f"{'='*60}\n")

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
            
            New observation structure (272D):
            - 0:2    - phase (sin, cos)
            - 2:5    - commands (v_x, v_y, omega_yaw)
            - 5:17   - joint positions (q)
            - 17:29  - joint velocities (dq)
            - 29:41  - previous actions
            - 41:44  - base angular velocity
            - 44:47  - projected gravity
            - 47:272 - height map (15x15=225)

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        print(f"size of noise_vec: {noise_vec.shape}")
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        
        noise_vec[0: 2] = 0.  # phase (sin, cos) - no noise
        noise_vec[2: 5] = 0.  # commands - no noise
        noise_vec[5: 17] = noise_scales.dof_pos * self.obs_scales.dof_pos  # joint positions
        noise_vec[17: 29] = noise_scales.dof_vel * self.obs_scales.dof_vel  # joint velocities
        noise_vec[29: 41] = 0.  # previous actions - no noise
        noise_vec[41: 44] = noise_scales.ang_vel * self.obs_scales.ang_vel  # base angular velocity
        noise_vec[44: 47] = noise_scales.quat * self.obs_scales.quat  # projected gravity (similar to orientation)
        noise_vec[47: 272] = noise_scales.height_measurements * self.obs_scales.height_measurements  # height map
        
        return noise_vec


    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        
        # Call parent step
        obs, privileged_obs, rew, done, info = super().step(actions)
        
        # Check success criteria (不终止，只记录！)
        self._check_success_criteria()
        
        return obs, privileged_obs, rew, done, info


    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        # Paper spec: use projected_gravity instead of euler angles for better representation
        # Both actor and critic use the same observation
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3D - gravity direction in robot frame (replaces euler angles)
        ), dim=-1)

        # Add height map to observations (paper spec: 15x15 elevation map)
        if self.cfg.terrain.measure_heights:
            # measured_heights shape: (num_envs, num_height_points=225)
            # root_states[:, 2] shape: (num_envs,)
            # We need to compute relative height: robot_height - terrain_height for each point
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, 
                -1, 1.
            ) * self.obs_scales.height_measurements
            # heights shape should be: (num_envs, 225)
            obs_buf = torch.cat((obs_buf, heights), dim=-1)  # Add heightmap: 47 + 225 = 272D
        else:
            # If heights not measured, use zeros as placeholder to maintain dimension consistency
            heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
            obs_buf = torch.cat((obs_buf, heights), dim=-1)
        # Critic uses the same observation as actor (aligned)
        self.privileged_obs_buf = obs_buf.clone()
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _reset_root_states(self, env_ids):
        """
        重写父类方法：根据地形类型设置个性化起始位置
        
        Type 7 - Balancing Beams:
          起始位置 = 起始平台 (x = -3.25m, y ± 0.3m)
        
        Type 8 - Stones Everywhere:
          起始位置 = 中心平台 (xy = center ± 1.5m)
        
        Type 9 - Stepping Stones:
          起始位置 = 起始平台 (x = -3.5m, y ± 0.3m) - 与 Balancing Beams 相同
        
        其他地形：使用默认逻辑（地形中心）
        """
        if len(env_ids) == 0:
            return
        
        # 调用父类设置基本位置
        super()._reset_root_states(env_ids)
        
        # 如果没有地形类型信息，直接返回
        if self.actual_terrain_types is None:
            return
        
        # 根据地形类型调整起始位置
        for env_id in env_ids:
            terrain_type = self.actual_terrain_types[env_id].item()
            
            if terrain_type == 7 or terrain_type == 9:
                # Type 7: Balancing Beams 和 Type 9: Stepping Stones - 起始平台
                # 地形格子：8.5m × 8.5m，border = 0.25m
                # 起始平台：y ∈ [0.25m, 1.25m]（从边缘算），中心在 0.75m
                # 相对于格子中心：0.75m - 4.25m = -3.5m
                # 平台尺寸：1.5m × 1m，安全偏移范围：x ∈ [-0.55m, +0.55m]（考虑机器人尺寸）
                x_offset = -3.5  # 起始平台中心（已考虑 0.25m border）
                y_offset = torch_rand_float(-0.3, 0.3, (1, 1), device=self.device).item()  # 横向小幅随机
                
                # 重置位置（覆盖父类的随机偏移）
                self.root_states[env_id, 0] = self.env_origins[env_id, 0] + x_offset
                self.root_states[env_id, 1] = self.env_origins[env_id, 1] + y_offset
                
            elif terrain_type == 8:
                # Type 8: Stones Everywhere - 中心平台
                # 地形格子：8.5m × 8.5m，border = 0.25m（对称，不影响中心位置）
                # 中心平台：4m × 4m 在格子中心，安全偏移范围：±1.8m（考虑机器人尺寸）
                # 使用 ±1.5m 保守估计，确保不会太靠近边缘
                x_offset = torch_rand_float(-1.5, 1.5, (1, 1), device=self.device).item()
                y_offset = torch_rand_float(-1.5, 1.5, (1, 1), device=self.device).item()
                
                # 重置位置
                self.root_states[env_id, 0] = self.env_origins[env_id, 0] + x_offset
                self.root_states[env_id, 1] = self.env_origins[env_id, 1] + y_offset
            
            # 其他地形类型：保持父类的默认行为（已经设置好了）
        
        # 应用新的 root states
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
    
    def _update_terrain_curriculum(self, env_ids):
        """
        Override: Curriculum based on "three in a row" success rule
        
        Rules:
        1. Robot progresses to next level after 3 consecutive successes
        2. Robot is NOT downgraded before reaching max level (stays at current level on failure)
        3. After reaching max level, randomly sample level (same as parent class)
        
        Reference: "the robot progresses to the next terrain level when it 
        successfully traverses the current terrain level three times in a row. 
        Furthermore, the robot will not be sent back to an easier terrain level 
        before it pass all levels"
        """
        if not self.init_done or not hasattr(self, 'episode_success_flags'):
            return
        
        for env_id in env_ids:
            success = self.episode_success_flags[env_id].item()
            current_level = self.terrain_levels[env_id].item()
            
            if success:
                # Increment consecutive success counter
                self.consecutive_successes[env_id] += 1
                
                # Check if ready to level up (3 consecutive successes)
                if self.consecutive_successes[env_id] >= 3:
                    # Level up
                    self.terrain_levels[env_id] = current_level + 1
                    
                    # Reset consecutive success counter for new level
                    self.consecutive_successes[env_id] = 0
            else:
                # Failure: reset consecutive success counter
                self.consecutive_successes[env_id] = 0
                # No downgrade (stay at current level)
        
        # Handle reaching max level: sample random level (same as parent class)
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        
        # Update environment origins based on new terrain levels
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], 
            self.terrain_types[env_ids]
        ]
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        
        # Reset success tracking
        self.episode_distance_traveled[env_ids] = 0.
        self.episode_success_flags[env_ids] = False
        
        # Record starting position for distance calculation
        # 注意：必须在 _reset_root_states 之后记录
        self.episode_start_pos[env_ids] = self.root_states[env_ids, :3].clone()
    
    def _check_success_criteria(self):
        """
        根据地形类型检测成功条件 - 但不终止！
        
        个性化成功标准：
        
        Type 7 - Balancing Beams:
          成功 = 到达终点平台 (x > +1.0m)
        
        Type 8 - Stones Everywhere:
          成功 = 距中心 > 3.75m AND 行走距离 >= 8m
        
        Type 9 - Stepping Stones:
          成功 = 到达终点平台 (x > +1.0m) - 与 Balancing Beams 相同
        
        其他地形 - 原始距离逻辑:
          成功 = 走得远 (distance > env_length / 2)
        
        关键：成功只设置标志，不改变 reset_buf！
        """
        if self.actual_terrain_types is None:
            return
        
        # 计算当前位置相对于地形中心
        rel_pos = self.root_states[:, :3] - self.env_origins
        
        # 更新行走距离（记录最大值）
        travel_dist = torch.norm(self.root_states[:, :2] - self.episode_start_pos[:, :2], dim=1)
        self.episode_distance_traveled = torch.maximum(self.episode_distance_traveled, travel_dist)
        
        # 初始化成功检测
        newly_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Type 7: Balancing Beams 和 Type 9: Stepping Stones - 到达终点平台
        beams_or_stepping_mask = (self.actual_terrain_types == 7) | (self.actual_terrain_types == 9)
        beams_success = beams_or_stepping_mask & (rel_pos[:, 0] > 1.0)
        newly_success |= beams_success
        
        # Type 8: Stones Everywhere - 到达边缘 + 行走足够远
        stones_mask = (self.actual_terrain_types == 8)
        dist_from_center = torch.norm(rel_pos[:, :2], dim=1)
        stones_success = stones_mask & (dist_from_center > 3.75) & (self.episode_distance_traveled >= 8.0)
        newly_success |= stones_success
        
        # 其他地形类型 - 使用原始距离逻辑
        other_mask = (self.actual_terrain_types != 7) & (self.actual_terrain_types != 8) & (self.actual_terrain_types != 9)
        if torch.any(other_mask):
            # 原始逻辑：走得远 (distance > env_length / 2) → 成功
            distance = torch.norm(rel_pos[:, :2], dim=1)
            other_success = other_mask & (distance > self.terrain.env_length / 2)
            newly_success |= other_success

        newly_success = newly_success & ~self.episode_success_flags

        self.episode_success_flags |= newly_success
        
        # 记录到 extras（用于 tensorboard）
        if torch.any(newly_success):
            self.extras['episode_success_count'] = newly_success.sum().item()
            self.extras['episode_avg_distance'] = self.episode_distance_traveled[newly_success].mean().item() if torch.any(newly_success) else 0.
            
            # 分地形类型统计
            for terrain_type in [7, 8, 9]:  # 统计 beams, stones, stepping_stones
                type_mask = newly_success & (self.actual_terrain_types == terrain_type)
                if torch.any(type_mask):
                    terrain_name = {7: 'beams', 8: 'stones', 9: 'stepping_stones'}[terrain_type]
                    self.extras[f'{terrain_name}_success_count'] = type_mask.sum().item()

# ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_foothold(self):
        """
        Foothold reward: encourages robot to place feet on safe footholds (e.g., stones, beams).
        
        Based on the sampling-based foothold reward from the paper:
        r_foothold = -∑(i=1→2) C_i · ∑(j=1→n) 𝟙{d_ij < ε}
        
        where:
        - C_i: contact indicator for foot i (1 if in contact, 0 otherwise)
        - d_ij: terrain height at sample point j on foot i
        - ε: depth tolerance threshold (e.g., 0.03m)
        
        The reward penalizes foot placements where sample points are significantly below
        the expected terrain height, indicating unsafe footholds.
        """
        # Only compute for heightfield/trimesh terrains
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 1. Get contact mask (C_i): only check feet that are in contact
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0  # (num_envs, 2)
        
        # 2. Define foot sampling points in local foot frame
        # Conservative estimate: 0.14m × 0.06m (inset 10% from URDF box 0.16m × 0.07m)
        # Using 3×3 grid = 9 sample points
        foot_length = 0.14
        foot_width = 0.06
        
        # Foot center offset in ankle_roll_link frame (based on URDF analysis)
        # The ankle_roll_link origin is approximately at the foot center, but may have slight offset
        # If needed, adjust this based on actual robot geometry
        foot_center_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # (x, y, z) in local frame
        
        # Create sampling grid (relative to foot center)
        x_samples = torch.tensor([-foot_length/2, 0.0, foot_length/2], device=self.device)
        y_samples = torch.tensor([-foot_width/2, 0.0, foot_width/2], device=self.device)
        
        # Generate 9 sample points: (9, 3) with z=0, plus offset to foot center
        sample_points_local = torch.stack([
            torch.stack([x, y, torch.tensor(0.0, device=self.device)]) + foot_center_offset
            for x in x_samples for y in y_samples
        ])  # (9, 3)
        
        num_samples = sample_points_local.shape[0]
        
        # 3. Get feet states
        feet_pos = self.rigid_state[:, self.feet_indices, 0:3]  # (num_envs, 2, 3)
        feet_quat = self.rigid_state[:, self.feet_indices, 3:7]  # (num_envs, 2, 4)
        
        # 4. Transform sample points to global coordinates and query terrain heights
        penalty = torch.zeros(self.num_envs, device=self.device)
        epsilon = -0.1  # Depth tolerance threshold (-10cm)
        
        for foot_idx in range(2):  # For each foot (left, right)
            # 4.1 Transform all sample points for this foot to global frame
            # Expand dimensions for broadcasting: (num_envs, 1, 3) and (9, 3) -> (num_envs, 9, 3)
            foot_pos_expanded = feet_pos[:, foot_idx, :].unsqueeze(1)  # (num_envs, 1, 3)
            foot_quat_expanded = feet_quat[:, foot_idx, :]  # (num_envs, 4)
            
            # Rotate sample points by foot orientation
            # quat_rotate expects: quat (num_envs, 4), vec (num_envs, 3)
            # We need to process each sample point separately
            sample_points_global = []
            for sample_idx in range(num_samples):
                sample_local = sample_points_local[sample_idx].unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, 3)
                sample_rotated = quat_rotate(foot_quat_expanded, sample_local)  # (num_envs, 3)
                sample_global = sample_rotated + feet_pos[:, foot_idx, :]  # (num_envs, 3)
                sample_points_global.append(sample_global)
            
            # Stack: (num_envs, num_samples, 3)
            sample_points_global = torch.stack(sample_points_global, dim=1)
            
            # 4.2 Query terrain heights at sample points
            # Follow the same method as _get_heights()
            points = sample_points_global.clone()
            points += self.terrain.cfg.border_size
            points = (points / self.terrain.cfg.horizontal_scale).long()
            px = points[:, :, 0].view(-1)  # Flatten: (num_envs * num_samples,)
            py = points[:, :, 1].view(-1)
            
            # Clamp to valid heightfield range
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
            
            # Sample terrain heights (take minimum of nearby grid points for safety)
            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            terrain_heights = torch.min(torch.min(heights1, heights2), heights3)
            terrain_heights = terrain_heights.view(self.num_envs, num_samples) * self.terrain.cfg.vertical_scale
            
            # 4.3 Get sample point z-coordinates (height above ground)
            sample_z = sample_points_global[:, :, 2]  # (num_envs, num_samples)
            
            # 4.4 Check if terrain height is below threshold (indicating poor foothold)
            # Low terrain height means the foot is over a gap/void
            below_threshold = (terrain_heights < epsilon).float()  # (num_envs, num_samples)
            
            # 4.5 Sum violations for this foot
            foot_penalty = below_threshold.sum(dim=1)  # (num_envs,)
            
            # 4.6 Apply contact mask: only penalize if foot is in contact
            penalty += contact[:, foot_idx] * foot_penalty
        
        # Return negative penalty (since we want to minimize violations)
        return -penalty
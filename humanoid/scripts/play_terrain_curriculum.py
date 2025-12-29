# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Configure for Terrain Curriculum (Balancing Beams + Stones Everywhere)
    env_cfg.terrain.mesh_type = 'trimesh'
    
    # Increase resolution for better visual of terrain details
    # Resolution: 0.02m (2cm) for good detail
    env_cfg.terrain.horizontal_scale = 0.02
    
    # Follow design convention: num_rows = difficulty levels, num_cols = terrain type variants
    # Modify these two parameters to change the terrain layout
    num_rows = 9  # Number of difficulty levels
    num_cols = 3   # Number of terrain types (balancing beams, stones everywhere, stepping stones)
    
    env_cfg.terrain.num_rows = num_rows
    env_cfg.terrain.num_cols = num_cols
    
    # Terrain dimensions: uniform 8.5m × 8.5m per terrain cell
    # For balancing beams: 2m effective width × 8m length (center carved, sides are pits)
    # For stones everywhere: 8m × 8m effective area (corners are pits)
    # For stepping stones: 2m effective width × 8m length (alternating stones, sides are pits)
    # All include 0.25m borders on all sides
    env_cfg.terrain.terrain_width = 8.5   # 8.5m width
    env_cfg.terrain.terrain_length = 8.5  # 8.5m length
    
    # Ensure standard terrain parameters are compatible
    env_cfg.terrain.border_size = 5
    
    # Enable curriculum mode
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = num_rows - 1  # Max difficulty level
    
    # Override num_envs based on terrain layout
    env_cfg.env.num_envs = num_rows * num_cols  # One robot per terrain cell
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    
    # Terrain proportions control which terrain appears in each column:
    # Proportions array: [flat, obstacles, uniform, slope+, slope-, stairs+, stairs-, beams, stones, stepping]
    # With num_cols=3 and proportions=[0, 0, 0, 0, 0, 0, 0, 0.33, 0.67, 1.0]:
    #   - Column 0 (choice ≈ 0.001): falls in range (0.0, 0.33] → balancing_beams_terrain
    #   - Column 1 (choice ≈ 0.334): falls in range (0.33, 0.67] → stones_everywhere_terrain
    #   - Column 2 (choice ≈ 0.667): falls in range (0.67, 1.0] → stepping_stones_terrain
    env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0.33, 0.33, 0.34]
    
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # Manually distribute robots to different terrain cells (row, col)
    # env.env_origins is a tensor of shape (num_envs, 3)
    # terrain.env_origins is numpy array of shape (num_rows, num_cols, 3)
    # Following design convention: 
    #   - num_rows = difficulty levels (configurable via num_rows variable above)
    #   - num_cols = terrain types (configurable via num_cols variable above)
    if hasattr(env, 'terrain') and env.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        terrain_num_rows = env.terrain.cfg.num_rows
        terrain_num_cols = env.terrain.cfg.num_cols
        
        print(f"Distributing robots across terrain curriculum ({terrain_num_rows} difficulty × {terrain_num_cols} terrain types)...")
        print("Terrain layout (uniform 8.5m × 8.5m per cell):")
        print("  Column 0: Balancing Beams (2m width × 8m length effective, 1.5m×1m platforms)")
        print("  Column 1: Stones Everywhere (8m × 8m effective, 4m×4m central platform)")
        print("  Column 2: Stepping Stones (2m width × 8m length effective, 1.5m×1m platforms, alternating stones)")
        print(f"  Rows: {terrain_num_rows} difficulty levels from 0.0 to 1.0")
        
        # Assign each robot to a different terrain cell
        # Robot 0-2 at difficulty 0 (row 0), Robot 3-5 at difficulty 1 (row 1), etc.
        for i in range(env.num_envs):
            row_idx = (i // terrain_num_cols) % terrain_num_rows  # Difficulty level
            col_idx = i % terrain_num_cols  # Terrain type
            
            # Get the origin for this terrain cell [row_idx, col_idx]
            terrain_origin = torch.from_numpy(env.terrain.env_origins[row_idx, col_idx]).to(env.device).to(torch.float)
            env.env_origins[i] = terrain_origin
            
            # Reset robot state to this new origin
            env.root_states[i, :3] = env.env_origins[i]
            env.root_states[i, :2] += torch_rand_float(-0.5, 0.5, (2,1), device=env.device).squeeze(1) # Add slight noise
            
            difficulty_val = row_idx / (terrain_num_rows - 1) if terrain_num_rows > 1 else 0.5
            terrain_name = {0: 'Beams', 1: 'Stones', 2: 'Stepping'}[col_idx]
            print(f"  Robot {i}: row={row_idx} (diff={difficulty_val:.3f}), col={col_idx} ({terrain_name})")
    
    # Trigger a reset to apply the new root states
    obs, critic_obs = env.reset()
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1200000 # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach()) # * 0.
        
        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()
    logger.plot_states()
    
    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)



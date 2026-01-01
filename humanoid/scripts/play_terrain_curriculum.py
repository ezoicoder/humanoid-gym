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

import torch
from tqdm import tqdm
from datetime import datetime


def play(args, fast_viewer=False):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # ========================================
    # CONFIGURATION: Simple and Clear
    # ========================================
    num_rows = 9  # Difficulty levels (0 to 8)
    num_cols = 1  # Terrain types
    
    # Set number of robots - must be >= num_rows for best visualization
    # Recommended: multiple of num_rows for even distribution
    target_num_envs = 18  # 2 robots per difficulty level
    
    # ⚠️ CRITICAL: Set these BEFORE creating environment (used in parent class __init__)
    env_cfg.terrain.num_rows = num_rows
    env_cfg.terrain.num_cols = num_cols
    
    # Enable curriculum mode and set max init level to cover ALL difficulty levels
    # This ensures robots can spawn at any difficulty level (0 to num_rows-1)
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = num_rows - 1  # Max difficulty level (0 to 8 for num_rows=9)
    
    # ⚠️ IMPORTANT: terrain levels are 0-indexed
    # num_rows=9 means levels [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # max_init_terrain_level must be num_rows-1 to avoid index out of bounds
    
    # Configure for Terrain Curriculum (Balancing Beams + Stones Everywhere)
    env_cfg.terrain.mesh_type = 'trimesh'
    
    # Increase resolution for better visual of terrain details
    # Resolution: 0.02m (2cm) for good detail
    env_cfg.terrain.horizontal_scale = 0.02
    
    # Terrain dimensions: uniform 8.5m × 8.5m per terrain cell
    # For balancing beams: 2m effective width × 8m length (center carved, sides are pits)
    # For stones everywhere: 8m × 8m effective area (corners are pits)
    # For stepping stones: 2m effective width × 8m length (alternating stones, sides are pits)
    # All include 0.25m borders on all sides
    env_cfg.terrain.terrain_width = 8.5   # 8.5m width
    env_cfg.terrain.terrain_length = 8.5  # 8.5m length
    
    # Ensure standard terrain parameters are compatible
    env_cfg.terrain.border_size = 5
    
    # Override num_envs based on terrain layout
    # ⚠️ IMPORTANT: Both must match to avoid conflicts
    # update_cfg_from_args will use args.num_envs to override env_cfg.env.num_envs
    
    # Set both to the same value
    env_cfg.env.num_envs = target_num_envs
    args.num_envs = target_num_envs
    
    print(f"[INFO] Final configuration:")
    print(f"  - env_cfg.env.num_envs = {env_cfg.env.num_envs}")
    print(f"  - args.num_envs = {args.num_envs}\n")
    
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    
    # Terrain proportions control which terrain appears in each column:
    # Proportions array: [flat, obstacles, uniform, slope+, slope-, stairs+, stairs-, beams, stones, stepping]
    # With num_cols=3 and proportions=[0, 0, 0, 0, 0, 0, 0, 0.33, 0.67, 1.0]:
    #   - Column 0 (choice ≈ 0.001): falls in range (0.0, 0.33] → balancing_beams_terrain
    #   - Column 1 (choice ≈ 0.334): falls in range (0.33, 0.67] → stones_everywhere_terrain
    #   - Column 2 (choice ≈ 0.667): falls in range (0.67, 1.0] → stepping_stones_terrain
    env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]
    
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)
    
    # Debug: Print configuration before environment creation
    print(f"\n=== Configuration Before Environment Creation ===")
    print(f"num_rows: {num_rows}, num_cols: {num_cols}")
    print(f"num_envs: {env_cfg.env.num_envs}")
    print(f"max_init_terrain_level: {env_cfg.terrain.max_init_terrain_level}")
    print(f"curriculum: {env_cfg.terrain.curriculum}")
    print(f"\n📌 Valid terrain levels: [0, {num_rows-1}] (total: {num_rows} levels)")
    print(f"   If max_init_terrain_level >= num_rows, you'll get index out of bounds!")
    print(f"=================================================\n")

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    
    # Debug: Print actual environment state after creation
    print(f"\n=== Environment State After Creation ===")
    print(f"env.num_envs: {env.num_envs}")
    print(f"env.max_terrain_level: {env.max_terrain_level}")
    print(f"Initial terrain_levels: {env.terrain_levels}")
    print(f"========================================\n")

    # ========================================
    # Simple and Safe: Override terrain_levels, then use reset_idx
    # (Similar to play.py but with custom terrain distribution)
    # ========================================
    print("=" * 60)
    print("Setting up terrain curriculum distribution...")
    print("=" * 60)
    
    if hasattr(env, 'terrain') and env.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        terrain_num_rows = env.terrain.cfg.num_rows
        terrain_num_cols = env.terrain.cfg.num_cols
        
        # Override terrain_levels to ensure even distribution across difficulty levels
        for i in range(env.num_envs):
            row_idx = (i // terrain_num_cols) % terrain_num_rows
            col_idx = i % terrain_num_cols
            env.terrain_levels[i] = row_idx
            env.terrain_types[i] = col_idx
        
        # Show distribution
        print(f"\n📊 Robot Distribution ({env.num_envs} robots across {terrain_num_rows} levels):")
        for level in range(terrain_num_rows):
            count = (env.terrain_levels == level).sum().item()
            print(f"  Level {level}: {count} robot(s)")
        print("=" * 60 + "\n")
        
        # Now reset all robots to apply the new terrain_levels
        # ⚠️ Temporarily disable init_done to prevent _update_terrain_curriculum from running
        print(f"\n⚠️  Calling env.reset_idx(all_env_ids)...")
        print(f"   (Temporarily setting init_done=False to preserve manual assignment)")
        
        original_init_done = env.init_done
        env.init_done = False  # This makes _update_terrain_curriculum return early
        
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(all_env_ids)
        
        env.init_done = original_init_done  # Restore for runtime curriculum
        print(f"   (Restored init_done=True, curriculum will work normally during runtime)")
        
        print(f"\n🔍 DEBUG: After reset_idx")
        print(f"   terrain_levels: {env.terrain_levels}")
        
        # Show distribution AFTER reset
        print(f"\n📊 Distribution AFTER reset_idx:")
        all_perfect = True
        expected_count = env.num_envs // terrain_num_rows
        for level in range(terrain_num_rows):
            count = (env.terrain_levels == level).sum().item()
            status = "✅" if count == expected_count or count == expected_count + 1 else "❌"
            print(f"  Level {level}: {count} robot(s) {status}")
            if count == 0:
                all_perfect = False
        
        if all_perfect:
            print("\n✅ SUCCESS: All difficulty levels have robots!")
        else:
            print("\n❌ WARNING: Some levels have no robots (curriculum might still be active)")
        print("=" * 60 + "\n")
    
    # Get observations (same as play.py - simple and safe)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
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
            env.commands[:, 0] = 1.0    # Forward velocity command (m/s) - increase for faster motion
            env.commands[:, 1] = 0.     # Lateral velocity
            env.commands[:, 2] = 0.     # Yaw rate
            env.commands[:, 3] = 0.     # Heading

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        
        # Render with optional frame sync control for speed
        if fast_viewer and not RENDER:
            # Fast mode: don't sync to real time, run as fast as possible
            env.render(sync_frame_time=False)
        elif not RENDER:
            # Normal mode: sync to real time for smooth viewing
            env.render(sync_frame_time=True)

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
    RENDER = False  # Set to True to record video
    FIX_COMMAND = True
    FAST_VIEWER = True  # Set to True to speed up live viewer (disable frame sync)
    
    print("=" * 60)
    print("PLAYBACK SETTINGS")
    print("=" * 60)
    print(f"  RENDER (record video): {RENDER}")
    print(f"  FAST_VIEWER (no sync): {FAST_VIEWER}")
    print(f"  FIX_COMMAND: {FIX_COMMAND}")
    if FAST_VIEWER and not RENDER:
        print("\n⚡ Fast viewer mode enabled - simulation will run as fast as possible!")
    print("=" * 60 + "\n")
    
    args = get_args()
    play(args, fast_viewer=FAST_VIEWER)



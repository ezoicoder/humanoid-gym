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


import numpy as np

from isaacgym import terrain_utils
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        # Special handling for virtual terrain modes
        # If using plane but need virtual terrain (e.g., Stage 1 training),
        # we still need to generate the virtual heightfield
        self.use_virtual_terrain = getattr(cfg, 'use_virtual_terrain', False)
        
        if self.type in ["none", 'plane']:

            print(f"nmd type {self.type}")

            # For plane + virtual terrain mode, we need to generate virtual terrain
            if self.type == 'plane' and self.use_virtual_terrain:
                print("[Terrain] Using plane for physics + virtual terrain for perception")
                self._init_plane_with_virtual(cfg)
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        
        # 记录每个地形格子的类型（不要硬编码！）
        # -1: 未知, 0-6: 其他地形, 7: balancing_beams, 8: stones_everywhere
        self.terrain_type_map = np.full((cfg.num_rows, cfg.num_cols), -1, dtype=np.int32)

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # Initialize virtual height field (for Stage 1 training with imaginary terrain)
        # Will be populated by terrain generation functions that support virtual terrain
        self.height_field_virtual = None
        
        if cfg.curriculum:
            self.curiculum()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        # If any terrain generated virtual heights, store them
        self.heightsamples_virtual = self.height_field_virtual
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain, terrain_type = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            self.terrain_type_map[i, j] = terrain_type  # 记录地形类型
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if self.cfg.num_rows > 1:
                    difficulty = i / (self.cfg.num_rows - 1)
                else:
                    difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain, terrain_type = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
                self.terrain_type_map[i, j] = terrain_type  # 记录地形类型
    
    def make_terrain(self, choice, difficulty):
        """
        生成地形并返回地形类型
        
        Returns:
            terrain: SubTerrain 对象
            terrain_type: int, 地形类型编号
                0: slope (positive)
                1: slope (negative) + random
                2: stairs (up)
                3: stairs (down)
                4: discrete obstacles
                5: stepping stones
                6: gap
                7: pit
        """
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # Add cfg reference for terrain generation functions that need it
        terrain.cfg = self.cfg
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        terrain_type = -1  # 默认未知
        
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
                terrain_type = 1  # negative slope
            else:
                terrain_type = 0  # positive slope
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_type = 1  # slope + random
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
                terrain_type = 3  # stairs down
            else:
                terrain_type = 2  # stairs up
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            terrain_type = 4  # discrete obstacles
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_type = 5  # stepping stones
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            terrain_type = 6  # gap
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            terrain_type = 7  # pit
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain, terrain_type

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        
        # SubTerrain height_field_raw is (width, length), but we need (length, width)
        # so we transpose it
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw.T
        
        # If terrain has virtual height field, add it to the virtual map
        if hasattr(terrain, 'height_field_virtual') and terrain.height_field_virtual is not None:
            # Initialize global virtual height field if not already done
            if self.height_field_virtual is None:
                self.height_field_virtual = np.zeros_like(self.height_field_raw)
            self.height_field_virtual[start_x: end_x, start_y:end_y] = terrain.height_field_virtual.T

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # Also transpose when accessing terrain.height_field_raw
        env_origin_z = np.max(terrain.height_field_raw.T[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
    
    def _init_plane_with_virtual(self, cfg):
        print(f"fuck init plane virtual")
        """
        Initialize virtual terrain for plane physics mode.
        
        This is used for Stage 1 training where:
        - Physical terrain: simple plane (最轻量)
        - Virtual terrain: stones heightfield (for perception and reward)
        
        This avoids the overhead of creating trimesh for flat ground.
        """
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type_map = np.full((cfg.num_rows, cfg.num_cols), -1, dtype=np.int32)
        
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        
        # No physical heightfield needed (using plane)
        self.height_field_raw = None
        
        # But we need virtual heightfield for perception
        self.height_field_virtual = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        
        print(f"fuck plane curiculum {cfg.curriculum}")

        # Generate virtual terrain only
        if cfg.curriculum:
            self._curiculum_virtual_only()
        else:
            self._randomized_virtual_only()
        
        # Set heightsamples for virtual terrain
        self.heightsamples = None  # No physical heightfield
        self.heightsamples_virtual = self.height_field_virtual
        
        print(f"[Terrain] Initialized plane + virtual terrain mode")
        print(f"  Physical: plane (infinite flat ground)")
        print(f"  Virtual: {self.tot_rows}x{self.tot_cols} heightfield")
    
    def _curiculum_virtual_only(self):
        """Generate virtual terrain curriculum (no physical terrain)."""
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if self.cfg.num_rows > 1:
                    difficulty = i / (self.cfg.num_rows - 1)
                    # print(f"fuck plane difficulty: {difficulty}")
                else:
                    difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001
                
                terrain, terrain_type = self.make_terrain(choice, difficulty)
                
                # Only extract virtual heightfield
                if hasattr(terrain, 'height_field_virtual') and terrain.height_field_virtual is not None:
                    start_x = self.border + i * self.length_per_env_pixels
                    end_x = self.border + (i + 1) * self.length_per_env_pixels
                    start_y = self.border + j * self.width_per_env_pixels
                    end_y = self.border + (j + 1) * self.width_per_env_pixels
                    self.height_field_virtual[start_x: end_x, start_y:end_y] = terrain.height_field_virtual.T
                
                self.terrain_type_map[i, j] = terrain_type
                
                # Set env origins (all at z=0 since physical is plane)
                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                self.env_origins[i, j] = [env_origin_x, env_origin_y, 0.0]
    
    def _randomized_virtual_only(self):
        """Generate randomized virtual terrain (no physical terrain)."""
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            terrain, terrain_type = self.make_terrain(choice, difficulty)
            
            # Only extract virtual heightfield
            if hasattr(terrain, 'height_field_virtual') and terrain.height_field_virtual is not None:
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_virtual[start_x: end_x, start_y:end_y] = terrain.height_field_virtual.T
            
            self.terrain_type_map[i, j] = terrain_type
            
            # Set env origins (all at z=0 since physical is plane)
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            self.env_origins[i, j] = [env_origin_x, env_origin_y, 0.0]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def balancing_beams_terrain(terrain, difficulty=1):
    """
    Generate balancing beams terrain according to paper specifications.
    
    Coordinate system:
    - y-direction: forward direction (from start platform to end platform)
    - x-direction: lateral direction (left-right), with periodic oscillation
    - z-direction: height (±0.05m variation)
    
    Parameters from paper (for discrete difficulty levels l ∈ {0,1,2,3,4,5,6,7,8}):
    - stone_size: 0.3 - 0.05[l/3] meters (width in x-direction)
    - stone_distance (x-direction): 0.4 - 0.05l meters (lateral offset amplitude)
    - y_distance: [0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.4, 0.2] meters (forward step)
    
    Terrain layout (in 8.5m × 8.5m terrain):
    - Effective area: 2m (width) × 8m (length) in the center
    - Border around the terrain: 0.25m on all sides (height = 0)
    - Start platform: 1.5m (width) × 1m (length)
    - End platform: 1.5m (width) × 1m (length)
    - Sides beyond 2m width: deep pit (carved out)
    
    Args:
        terrain: SubTerrain object
        difficulty: float from 0 to 1, will be mapped to discrete level l from 0 to 8
    """
    # Map difficulty 0..1 to discrete level 0..8 (round to nearest integer)
    l = int(np.round(difficulty * 8))
    l = np.clip(l, 0, 8)
    
    # Paper formula: stone_size (beam width in x-direction)
    stone_size = 0.3 - 0.05 * (l // 3)
    
    # Paper formula: stone_distance (lateral offset amplitude in x-direction)
    x_offset = 0.4 - 0.05 * l
    
    # Paper formula: y_distance (forward step in y-direction)
    y_steps = [0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.35, 0.4, 0.2]
    y_step = y_steps[l]
    
    # Set entire terrain to pit depth (1.0m below ground)
    depth = 1.0
    depth_int = int(depth / terrain.vertical_scale)
    terrain.height_field_raw[:, :] = -depth_int
    
    # Border configuration
    border_width = 0.25  # 0.25m border on all sides
    border_width_pixels = int(border_width / terrain.horizontal_scale)
    
    # Create border frame around the entire terrain (height = 0)
    # Top and bottom borders (full width)
    terrain.height_field_raw[:, 0:border_width_pixels] = 0
    terrain.height_field_raw[:, terrain.length - border_width_pixels:terrain.length] = 0
    # Left and right borders (full length)
    terrain.height_field_raw[0:border_width_pixels, :] = 0
    terrain.height_field_raw[terrain.width - border_width_pixels:terrain.width, :] = 0
    
    # Define effective beam area: 2m width centered in the terrain
    # The rest remains as deep pit
    effective_width = 2.0
    effective_width_pixels = int(effective_width / terrain.horizontal_scale)
    mid_x = terrain.width // 2
    
    # Boundaries for the effective beam area (in x-direction)
    beam_area_x1 = mid_x - effective_width_pixels // 2
    beam_area_x2 = mid_x + effective_width_pixels // 2
    
    # Platform dimensions: 1.5m (width) × 1m (length)
    platform_width = 1.5
    platform_length = 1.0
    platform_width_pixels = int(platform_width / terrain.horizontal_scale)
    platform_length_pixels = int(platform_length / terrain.horizontal_scale)
    
    # Start platform: 1.5m × 1m at the beginning (after border)
    start_y = border_width_pixels
    terrain.height_field_raw[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                            start_y : start_y + platform_length_pixels] = 0
    
    # End platform: 1.5m × 1m at the end (before border)
    end_y = terrain.length - border_width_pixels - platform_length_pixels
    terrain.height_field_raw[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                            end_y : end_y + platform_length_pixels] = 0
    
    # Convert parameters to pixels
    stone_size_pixels = int(stone_size / terrain.horizontal_scale)
    y_step_pixels = int(y_step / terrain.horizontal_scale)
    x_offset_pixels = int(x_offset / terrain.horizontal_scale)
    
    # Start generating stones after the start platform
    current_y = start_y + platform_length_pixels
    
    step = 0
    
    print(f"[Balancing Beams] difficulty={difficulty:.2f}, l={l}, stone_size={stone_size:.3f}m, "
          f"x_offset={x_offset:.3f}m, y_step={y_step:.3f}m, effective_width=2.0m")

    # Generate stones until we reach the end platform
    while current_y + stone_size_pixels <= end_y:
        # Periodic oscillation in x-direction with period 2 (zigzag pattern)
        # step % 2 == 0: offset to one side
        # step % 2 == 1: offset to the other side
        if step % 2 == 0:
            current_x_center = mid_x + x_offset_pixels//2
        else:
            current_x_center = mid_x - x_offset_pixels//2
        
        # Calculate stone boundaries in x-direction (lateral)
        x1 = current_x_center - stone_size_pixels // 2
        x2 = current_x_center + stone_size_pixels // 2
        
        # Calculate stone boundaries in y-direction (forward)
        y1 = current_y
        y2 = current_y + stone_size_pixels  # stone is square (stone_size × stone_size)
        
        # Don't overlap with end platform
        if y2 > end_y:
            break
        
        # Height variation ±0.05m as specified in paper
        height_var = np.random.uniform(-0.05, 0.05)
        height_int = int(height_var / terrain.vertical_scale)
        
        # Clip to terrain boundaries and effective beam area
        x1 = max(beam_area_x1, x1)
        x2 = min(beam_area_x2, x2)
        y1 = max(border_width_pixels, y1)
        y2 = min(terrain.length - border_width_pixels, y2)
        
        # Place the stone only if it's within the effective beam area
        if x2 > x1 and y2 > y1:
            terrain.height_field_raw[x1:x2, y1:y2] = height_int
        
        # Advance to next stone position (forward by y_step)
        current_y = y1 + y_step_pixels
        step += 1

def stones_everywhere_terrain(terrain, difficulty=1):
    """
    Generate stones everywhere terrain according to paper specifications.
    
    This creates a 2D grid of stepping stones distributed across the entire terrain.
    From the image (a) in the paper, stones are arranged in a regular grid pattern
    with gaps between them, increasing in difficulty as stones become smaller and
    more separated.
    
    Coordinate system:
    - x-direction: lateral direction (left-right)
    - y-direction: forward direction (from start to end)
    - z-direction: height (±0.05m variation)
    
    Parameters from paper (for discrete difficulty levels l ∈ {0,1,2,3,4,5,6,7,8}):
    - stone_size: max{0.25, 1.5(1 - 0.1l)} meters (square stones)
    - stone_distance: 0.05⌈l/2⌉ meters (gap between stones in both x and y)
    
    Terrain layout (in 8.5m × 8.5m terrain):
    - Effective area: 8m × 8m in the center
    - Border around the terrain: 0.25m on all sides (height = 0)
    - Central platform: 4m × 4m at the center of terrain (merged start/end)
    - Stones distributed in 2D grid across effective area, then platform placed on top
    - Corners beyond 8m effective area: deep pit (carved out)
    
    Args:
        terrain: SubTerrain object
        difficulty: float from 0 to 1, will be mapped to discrete level l from 0 to 8
    """
    # Map difficulty 0..1 to discrete level 0..8 (round to nearest integer)
    l = int(np.round(difficulty * 8))
    l = np.clip(l, 0, 8)
    
    # Paper formula: stone_size (square stone dimensions)
    stone_size = max(0.25, 1.5 * (1 - 0.1 * l))
    
    # Paper formula: stone_distance (gap between stones)
    stone_distance = 0.05 * np.ceil(l / 2)
    
    # Set entire terrain to pit depth (1.0m below ground)
    depth = 1.0
    depth_int = int(depth / terrain.vertical_scale)
    terrain.height_field_raw[:, :] = -depth_int
    
    # Border configuration
    border_width = 0.25  # 0.25m border on all sides
    border_width_pixels = int(border_width / terrain.horizontal_scale)
    
    # Create border frame around the entire terrain (height = 0)
    # Top and bottom borders (full width)
    terrain.height_field_raw[:, 0:border_width_pixels] = 0
    terrain.height_field_raw[:, terrain.length - border_width_pixels:terrain.length] = 0
    # Left and right borders (full length)
    terrain.height_field_raw[0:border_width_pixels, :] = 0
    terrain.height_field_raw[terrain.width - border_width_pixels:terrain.width, :] = 0
    
    # Center position in both x and y directions
    mid_x = terrain.width // 2
    mid_y = terrain.length // 2
    
    # Central platform dimensions (read from config or use default)
    platform_width = getattr(terrain.cfg, 'platform_width', 4.0)  # meters (default: 4.0)
    platform_length = getattr(terrain.cfg, 'platform_length', 4.0)  # meters (default: 4.0)
    
    # Convert parameters to pixels
    stone_size_pixels = int(stone_size / terrain.horizontal_scale)
    stone_distance_pixels = int(stone_distance / terrain.horizontal_scale)
    
    # Grid spacing (stone size + gap)
    grid_spacing = stone_size_pixels + stone_distance_pixels
    
    # Define effective stone area: 8m × 8m in the center
    effective_area = 8.0
    effective_area_pixels = int(effective_area / terrain.horizontal_scale)
    
    # Calculate effective area boundaries (centered)
    effective_x1 = mid_x - effective_area_pixels // 2
    effective_x2 = mid_x + effective_area_pixels // 2
    effective_y1 = mid_y - effective_area_pixels // 2
    effective_y2 = mid_y + effective_area_pixels // 2
    
    # Usable area for stones within the effective area
    usable_width = effective_area_pixels
    usable_length = effective_area_pixels
    
    print(f"[Stones Everywhere] difficulty={difficulty:.2f}, l={l}, stone_size={stone_size:.3f}m, "
          f"stone_distance={stone_distance:.3f}m, effective_area={effective_area}m×{effective_area}m, "
          f"platform={platform_width}m×{platform_length}m")
    
    # Step 1: Generate stones in a 2D grid pattern within the effective area
    # We generate all stones first, ignoring the platform
    num_stones_x = int(usable_width / grid_spacing)
    num_stones_y = int(usable_length / grid_spacing)
    
    grid_width = num_stones_x * grid_spacing
    grid_length = num_stones_y * grid_spacing
    
    start_x = effective_x1 + (usable_width - grid_width) // 2
    start_y = effective_y1 + (usable_length - grid_length) // 2
    
    # Generate all stones without checking platform overlap
    for i in range(num_stones_x):
        current_x = start_x + i * grid_spacing
        
        for j in range(num_stones_y):
            current_y = start_y + j * grid_spacing
            
            # Calculate stone boundaries
            x1 = current_x
            x2 = current_x + stone_size_pixels
            y1 = current_y
            y2 = current_y + stone_size_pixels
            
            # Height variation ±0.05m as specified in paper
            height_var = np.random.uniform(-0.05, 0.05)
            height_int = int(height_var / terrain.vertical_scale)
            
            # Clip to effective area boundaries
            x1 = max(effective_x1, x1)
            x2 = min(effective_x2, x2)
            y1 = max(effective_y1, y1)
            y2 = min(effective_y2, y2)
            
            # Place the stone if within bounds
            if x2 > x1 and y2 > y1:
                terrain.height_field_raw[x1:x2, y1:y2] = height_int
    
    # Step 2: Place central platform on top, directly overwriting any stones underneath
    platform_width_pixels = int(platform_width / terrain.horizontal_scale)
    platform_length_pixels = int(platform_length / terrain.horizontal_scale)
    
    # Place the platform at the center, overwriting stones
    terrain.height_field_raw[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                            mid_y - platform_length_pixels//2 : mid_y + platform_length_pixels//2] = 0

def stones_everywhere_stage1_terrain(terrain, difficulty=1):
    """
    Stage 1 training terrain: Physical flat ground + Virtual stones perception.
    
    This implements the "Stage 1: Soft Terrain Dynamics" from the paper where:
    - Robot walks on FLAT physical terrain (safe, no termination from falls)
    - Robot perceives VIRTUAL stones terrain (elevation map for critic/actor)
    - Foothold reward computed from VIRTUAL stones heights (sparse reward signal)
    - Locomotion rewards computed from FLAT terrain (dense reward signal)
    
    
    This allows the robot to "imagine" walking on stones while actually on safe ground,
    learning terrain-aware behaviors without the risk of early termination.
    
    Args:
        terrain: SubTerrain object
        difficulty: float from 0 to 1, will be mapped to discrete level l from 0 to 8
    
    Returns:
        The terrain object will have:
        - terrain.height_field_raw: flat terrain (for physics)
        - terrain.height_field_virtual: stones pattern (for perception and foothold reward)
    """
    # Map difficulty 0..1 to discrete level 0..8 (round to nearest integer)
    l = int(np.round(difficulty * 8))
    l = np.clip(l, 0, 8)
    
    # Paper formula: stone_size (square stone dimensions)
    stone_size = max(0.25, 1.5 * (1 - 0.1 * l))
    
    # Paper formula: stone_distance (gap between stones)
    stone_distance = 0.05 * np.ceil(l / 2)
    
    # ========== PART 1: Physical Terrain (FLAT) ==========
    # Set entire physical terrain to flat ground (height = 0)
    terrain.height_field_raw[:, :] = 0
    
    # ========== PART 2: Virtual Terrain (STONES) ==========
    # Create virtual height field with same shape as physical terrain
    terrain.height_field_virtual = np.zeros_like(terrain.height_field_raw)
    
    # Set virtual terrain to pit depth initially (1.0m below ground)
    depth = 1.0
    depth_int = int(depth / terrain.vertical_scale)
    terrain.height_field_virtual[:, :] = -depth_int
    
    # Border configuration for virtual terrain
    border_width = 0.25  # 0.25m border on all sides
    border_width_pixels = int(border_width / terrain.horizontal_scale)
    
    # Create border frame around the virtual terrain (height = 0)
    terrain.height_field_virtual[:, 0:border_width_pixels] = 0
    terrain.height_field_virtual[:, terrain.length - border_width_pixels:terrain.length] = 0
    terrain.height_field_virtual[0:border_width_pixels, :] = 0
    terrain.height_field_virtual[terrain.width - border_width_pixels:terrain.width, :] = 0
    
    # Center position in both x and y directions
    mid_x = terrain.width // 2
    mid_y = terrain.length // 2
    
    # Central platform dimensions (can be overridden via command line)
    platform_width = getattr(terrain.cfg, 'platform_width', 1.0)  # meters (default: 1.0)
    platform_length = getattr(terrain.cfg, 'platform_length', 1.0)  # meters (default: 1.0)
    
    # Convert parameters to pixels
    stone_size_pixels = int(stone_size / terrain.horizontal_scale)
    stone_distance_pixels = int(stone_distance / terrain.horizontal_scale)
    
    # Grid spacing (stone size + gap)
    grid_spacing = stone_size_pixels + stone_distance_pixels
    
    # Define effective stone area: 8m × 8m in the center
    effective_area = 8.0
    effective_area_pixels = int(effective_area / terrain.horizontal_scale)
    
    # Calculate effective area boundaries (centered)
    effective_x1 = mid_x - effective_area_pixels // 2
    effective_x2 = mid_x + effective_area_pixels // 2
    effective_y1 = mid_y - effective_area_pixels // 2
    effective_y2 = mid_y + effective_area_pixels // 2
    
    # Usable area for stones within the effective area
    usable_width = effective_area_pixels
    usable_length = effective_area_pixels
    
    print(f"[Stage1 Virtual Stones] difficulty={difficulty:.2f}, l={l}, stone_size={stone_size:.3f}m, "
          f"stone_distance={stone_distance:.3f}m, physical=FLAT, virtual=STONES")
    
    # Generate virtual stones in a 2D grid pattern
    num_stones_x = int(usable_width / grid_spacing)
    num_stones_y = int(usable_length / grid_spacing)
    
    grid_width = num_stones_x * grid_spacing
    grid_length = num_stones_y * grid_spacing
    
    start_x = effective_x1 + (usable_width - grid_width) // 2
    start_y = effective_y1 + (usable_length - grid_length) // 2
    
    # Generate all virtual stones
    for i in range(num_stones_x):
        current_x = start_x + i * grid_spacing
        
        for j in range(num_stones_y):
            current_y = start_y + j * grid_spacing
            
            # Calculate stone boundaries
            x1 = current_x
            x2 = current_x + stone_size_pixels
            y1 = current_y
            y2 = current_y + stone_size_pixels
            
            # Height variation ±0.05m as specified in paper
            height_var = np.random.uniform(-0.05, 0.05)
            height_int = int(height_var / terrain.vertical_scale)
            
            # Clip to effective area boundaries
            x1 = max(effective_x1, x1)
            x2 = min(effective_x2, x2)
            y1 = max(effective_y1, y1)
            y2 = min(effective_y2, y2)
            
            # Place the virtual stone if within bounds
            if x2 > x1 and y2 > y1:
                terrain.height_field_virtual[x1:x2, y1:y2] = height_int
    
    # Place virtual central platform on top, overwriting virtual stones
    platform_width_pixels = int(platform_width / terrain.horizontal_scale)
    platform_length_pixels = int(platform_length / terrain.horizontal_scale)
    
    terrain.height_field_virtual[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                                mid_y - platform_length_pixels//2 : mid_y + platform_length_pixels//2] = 0

def stepping_stones_terrain(terrain, difficulty=1):
    """
    Generate stepping stones terrain with alternating left-right pattern.
    
    This creates discrete stepping stones arranged in two alternating rows,
    requiring precise foot placement to traverse from start to end platform.
    
    Coordinate system:
    - y-direction: forward direction (from start platform to end platform)
    - x-direction: lateral direction (left-right), stones alternate sides
    - z-direction: height (±0.05m variation)
    
    Parameters from specification (for discrete difficulty levels l ∈ {0,1,2,3,4,5,6,7,8}):
    - stone_size: [0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2] meters (square stones)
    - stone_distance (forward spacing): 0.1 + 0.05l meters
    
    Terrain layout (in 8.5m × 8.5m terrain):
    - Effective area: 2m (width) × 8m (length) in the center
    - Border around the terrain: 0.25m on all sides (height = 0)
    - Start platform: 1.5m (width) × 1m (length)
    - End platform: 1.5m (width) × 1m (length)
    - Sides beyond 2m width: deep pit (carved out)
    - Stones: Two rows alternating left-right
    
    Args:
        terrain: SubTerrain object
        difficulty: float from 0 to 1, will be mapped to discrete level l from 0 to 8
    """
    # Map difficulty 0..1 to discrete level 0..8 (round to nearest integer)
    l = int(np.round(difficulty * 8))
    l = np.clip(l, 0, 8)
    
    # Stone size sequence from specification
    stone_sizes = [0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]
    stone_size = stone_sizes[l]
    
    # Stone distance (forward spacing) formula from specification
    stone_distance = 0.1 + 0.05 * l
    
    # Lateral offset for left/right alternation (fixed at 0.4m from center)
    lateral_offset = 0.4
    
    # Set entire terrain to pit depth (1.0m below ground)
    depth = 1.0
    depth_int = int(depth / terrain.vertical_scale)
    terrain.height_field_raw[:, :] = -depth_int
    
    # Border configuration
    border_width = 0.25  # 0.25m border on all sides
    border_width_pixels = int(border_width / terrain.horizontal_scale)
    
    # Create border frame around the entire terrain (height = 0)
    # Top and bottom borders (full width)
    terrain.height_field_raw[:, 0:border_width_pixels] = 0
    terrain.height_field_raw[:, terrain.length - border_width_pixels:terrain.length] = 0
    # Left and right borders (full length)
    terrain.height_field_raw[0:border_width_pixels, :] = 0
    terrain.height_field_raw[terrain.width - border_width_pixels:terrain.width, :] = 0
    
    # Define effective area: 2m width centered in the terrain
    effective_width = 2.0
    effective_width_pixels = int(effective_width / terrain.horizontal_scale)
    mid_x = terrain.width // 2
    
    # Boundaries for the effective area (in x-direction)
    area_x1 = mid_x - effective_width_pixels // 2
    area_x2 = mid_x + effective_width_pixels // 2
    
    # Platform dimensions: 1.5m (width) × 1m (length)
    platform_width = 1.5
    platform_length = 1.0
    platform_width_pixels = int(platform_width / terrain.horizontal_scale)
    platform_length_pixels = int(platform_length / terrain.horizontal_scale)
    
    # Start platform: 1.5m × 1m at the beginning (after border)
    start_y = border_width_pixels
    terrain.height_field_raw[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                            start_y : start_y + platform_length_pixels] = 0
    
    # End platform: 1.5m × 1m at the end (before border)
    end_y = terrain.length - border_width_pixels - platform_length_pixels
    terrain.height_field_raw[mid_x - platform_width_pixels//2 : mid_x + platform_width_pixels//2,
                            end_y : end_y + platform_length_pixels] = 0
    
    # Convert parameters to pixels
    stone_size_pixels = int(stone_size / terrain.horizontal_scale)
    stone_distance_pixels = int(stone_distance / terrain.horizontal_scale)
    lateral_offset_pixels = int(lateral_offset / terrain.horizontal_scale)
    
    # Start generating stones after the start platform
    stone_start_y = start_y + platform_length_pixels
    
    print(f"[Stepping Stones] difficulty={difficulty:.2f}, l={l}, stone_size={stone_size:.3f}m, "
          f"stone_distance={stone_distance:.3f}m, lateral_offset={lateral_offset:.3f}m, effective_width=2.0m")
    
    # Generate two separate rows of stones: left row and right row
    for row_side in ['left', 'right']:
        # Set x-position based on which row we're generating
        if row_side == 'left':
            current_x_center = mid_x - lateral_offset_pixels
        else:
            current_x_center = mid_x + lateral_offset_pixels
        
        # Start from the beginning for this row
        current_y = stone_start_y
        stone_count = 0
        
        # Generate stones in this row until we reach the end platform
        while current_y < end_y:
            # Calculate stone boundaries in x-direction (lateral)
            x1 = current_x_center - stone_size_pixels // 2
            x2 = current_x_center + stone_size_pixels // 2
            
            # Calculate stone boundaries in y-direction (forward)
            y1 = current_y
            y2 = current_y + stone_size_pixels  # stone is square
            
            # # Don't overlap with end platform
            if y2 > end_y:y2 = end_y
            
            # Height variation ±0.05m as specified
            height_var = np.random.uniform(-0.05, 0.05)
            height_int = int(height_var / terrain.vertical_scale)
            
            # 保证xvar是整数，且取值区间为[stone_size_pixels //2-lateral_offset_pixels, lateral_offset_pixels - stone_size_pixels //2]
            xvar_low = stone_size_pixels // 2 - lateral_offset_pixels
            xvar_high = lateral_offset_pixels - stone_size_pixels // 2
            if xvar_high < xvar_low:
                xvar_high = xvar_low  # 如果区间非法，则degenerate为单点
            xvar = np.random.randint(xvar_low, xvar_high + 1)

            x1 = x1 + xvar
            x2 = x2 + xvar

            # Clip to terrain boundaries and effective area
            x1 = max(area_x1, x1)
            x2 = min(area_x2, x2)
            y1 = max(border_width_pixels, y1)
            y2 = min(terrain.length - border_width_pixels, y2)
            
            # Place the stone only if it's within the effective area
            if x2 > x1 and y2 > y1:
                terrain.height_field_raw[x1:x2, y1:y2] = height_int
                stone_count += 1
            
            # Advance to next stone position: current stone start + stone size + gap
            # This ensures stones don't overlap and have the specified gap between them
            current_y = y1 + stone_size_pixels + stone_distance_pixels
        
        print(f"  Generated {stone_count} stones in {row_side} row (x_offset={lateral_offset if row_side=='right' else -lateral_offset:.2f}m)")

class HumanoidTerrain(Terrain):
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        super().__init__(cfg, num_robots)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            terrain, terrain_type = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            self.terrain_type_map[i, j] = terrain_type  # 记录地形类型

    def make_terrain(self, choice, difficulty):
        """
        生成地形并返回地形类型
        
        Returns:
            terrain: SubTerrain 对象
            terrain_type: int, 地形类型编号
                0: flat
                1: discrete obstacles
                2: random uniform
                3: pyramid slope (positive)
                4: pyramid slope (negative)
                5: pyramid stairs (up)
                6: pyramid stairs (down)
                7: balancing beams
                8: stones everywhere
                9: stepping stones
        """
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # Add cfg reference for terrain generation functions that need it
        terrain.cfg = self.cfg
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        
        terrain_type = -1  # 默认未知
        
        # Terrain type selection based on proportions
        if choice < self.proportions[0]:
            # Flat terrain
            terrain_type = 0
            pass
        elif choice < self.proportions[1]:
            # Discrete Obstacles
            terrain_type = 1
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[2]:
            # Random Uniform
            terrain_type = 2
            terrain_utils.random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # Pyramid Slope (positive)
            terrain_type = 3
            terrain_utils.pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
        elif choice < self.proportions[4]:
            # Pyramid Slope (negative)
            terrain_type = 4
            terrain_utils.pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
        elif choice < self.proportions[5]:
            # Pyramid Stairs (up)
            terrain_type = 5
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height, platform_size=1.)
        elif choice < self.proportions[6]:
            # Pyramid Stairs (down)
            terrain_type = 6
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-discrete_obstacles_height, platform_size=1.)
        elif choice < self.proportions[7]:
            # Balancing Beams
            terrain_type = 7
            balancing_beams_terrain(terrain, difficulty)
        elif choice < self.proportions[8]:
            # Stones Everywhere (real physical stones)
            terrain_type = 8
            stones_everywhere_terrain(terrain, difficulty)
        elif choice < self.proportions[9]:
            # Stepping Stones
            terrain_type = 9
            stepping_stones_terrain(terrain, difficulty)
        else:
            # Stage 1: Flat with Virtual Stones (imaginary training)
            terrain_type = 10
            stones_everywhere_stage1_terrain(terrain, difficulty)
        
        return terrain, terrain_type

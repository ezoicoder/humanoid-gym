# 地形个性化评估功能

## 核心功能

### ✅ 根据地形类型个性化配置

**不同地形，不同起始位置，不同成功标准！**

| 地形类型 | 起始位置 | 成功标准 |
|---------|---------|---------|
| **Balancing Beams** | 起始平台<br>(y = -3.25m, x ± 0.3m) | 到达终点平台<br>(y > +3.0m) |
| **Stones Everywhere** | 中心平台<br>(xy = center ± 1.5m) | 距中心 > 3.75m<br>AND 行走 >= 8m |

### ✅ 成功不终止！

```python
if 达到成功标准:
    self.episode_success_flags = True  # 只记录
    继续行走                            # 不终止！

if 碰撞失败:
    reset()  # 终止

if 超时:
    reset()  # 终止
```

## 实现细节

### 1. 地形类型识别（鲁棒！不硬编码！）

**关键理解：Curriculum 模式下的地形分配**

```python
# 父类 LeggedRobot._get_env_origins() 中：
terrain_levels = torch.randint(0, max_init_level+1, (num_envs,))  # 随机行索引
terrain_types = floor(env_id / (num_envs / num_cols))  # 列索引 (0, 1, 2, ...)
env_origins = terrain_origins[terrain_levels, terrain_types]

# 关键：
# - terrain_levels 是动态的（curriculum 会调整）
# - terrain_types 是静态的列索引
# - 同一列的所有行都是同一种地形类型！
```

**实现步骤：**

#### Step 1: 地形生成时记录类型

```python
class Terrain:
    def __init__(self, cfg, num_robots):
        # 创建类型映射表 (num_rows × num_cols)
        self.terrain_type_map = np.full((num_rows, num_cols), -1, dtype=np.int32)
        
    def make_terrain(self, choice, difficulty):
        # 根据 choice 和 proportions 确定地形类型
        terrain_type = -1
        if choice < proportions[0]:
            terrain_type = 0  # flat
        elif choice < proportions[1]:
            terrain_type = 1  # obstacles
        ...
        elif choice < proportions[7]:
            terrain_type = 7  # balancing_beams
        else:
            terrain_type = 8  # stones_everywhere
        return terrain, terrain_type
        
    def curiculum(self):
        for row in range(num_rows):
            for col in range(num_cols):
                choice = col / num_cols + 0.001
                terrain, terrain_type = self.make_terrain(choice, difficulty)
                self.terrain_type_map[row, col] = terrain_type  # ✅ 记录！
```

#### Step 2: 环境初始化时读取并转换

```python
def _init_terrain_types(self):
    """从地形对象读取类型（不硬编码！）"""
    
    # ⚠️ 关键：父类的 terrain_types 必须保留！
    # 因为 _update_terrain_curriculum() 需要用它索引 terrain_origins:
    #   self.env_origins[env_ids] = self.terrain_origins[terrain_levels, terrain_types]
    
    # 创建新变量存储实际的地形类型编号
    self.actual_terrain_types = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    
    # 由于同一列的所有行都是同一类型，查第 0 行即可
    for i in range(num_envs):
        col_idx = self.terrain_types[i].item()  # 父类的列索引
        actual_type = self.terrain.terrain_type_map[0, col_idx]  # ✅ 查第 0 行
        self.actual_terrain_types[i] = actual_type
    
    # ✅ 不覆盖 self.terrain_types，使用新变量 self.actual_terrain_types
```

**地形类型编号：**
- 0: flat, 1: obstacles, 2: random, 3: slope+, 4: slope-
- 5: stairs+, 6: stairs-, 7: balancing_beams, 8: stones_everywhere

### 2. 个性化起始位置

```python
def _reset_root_states(self, env_ids):
    """重写父类方法，根据地形类型设置起始位置"""
    
    super()._reset_root_states(env_ids)  # 先调用父类
    
    if self.actual_terrain_types is None:
        return
    
    for env_id in env_ids:
        terrain_type = self.actual_terrain_types[env_id].item()  # ✅ 使用 actual_terrain_types
        
        if terrain_type == 7:  # Balancing Beams
            # 起始平台：y = -3.25m, x ± 0.3m
            self.root_states[env_id, 0] = origin_x + random(-0.3, 0.3)
            self.root_states[env_id, 1] = origin_y - 3.25
            
        elif terrain_type == 8:  # Stones Everywhere
            # 中心平台：xy = center ± 1.5m
            self.root_states[env_id, 0] = origin_x + random(-1.5, 1.5)
            self.root_states[env_id, 1] = origin_y + random(-1.5, 1.5)
```

### 3. 个性化成功标准

```python
def _check_success_criteria(self):
    """根据地形类型检测成功 - 但不终止！"""
    
    if self.actual_terrain_types is None:
        return
    
    # Type 7: Balancing Beams
    beams_success = (self.actual_terrain_types == 7) & (rel_pos_y > 3.0)  # ✅ 使用 actual_terrain_types
    
    # Type 8: Stones Everywhere
    stones_success = (self.actual_terrain_types == 8) &  # ✅ 使用 actual_terrain_types 
                     (dist_from_center > 3.75) & 
                     (travel_distance >= 8.0)
    
    # 只记录，不终止！
    self.episode_success_flags |= (beams_success | stones_success)
```

## 增强的 Curriculum 机制

### 原始 Curriculum（父类 LeggedRobot）

基于**相对距离**动态调整难度：

```python
# 升级条件：走得远（虽然失败了，但走了超过地形一半）
move_up = distance > terrain_length / 2  # > 4m

# 降级条件：走得近（远小于期望距离）
expected_dist = command_vel * max_episode_length * 0.5
move_down = (distance < expected_dist) & ~move_up

# 更新难度
terrain_levels += move_up - move_down
```

### 增强 Curriculum（XBotLFreeEnv）

结合**成功标志**和**相对距离**：

```python
def _update_terrain_curriculum(self, env_ids):
    """Override: 增强版 curriculum"""
    
    # ✅ 优先级 1: 成功 → 直接升级（最高优先级）
    if episode_success_flags[env_ids]:
        level_adjustment += 1
    
    # ✅ 优先级 2: 失败 → 使用原始距离逻辑
    else:
        if distance > terrain_length / 2:
            level_adjustment += 1  # 走得远 → 升级
        elif distance < expected_dist:
            level_adjustment -= 1  # 走得近 → 降级
    
    terrain_levels[env_ids] += level_adjustment
    env_origins[env_ids] = terrain_origins[terrain_levels, terrain_types]
```

**优势**：
- ✅ 成功的机器人快速升级到更难地形
- ✅ 失败但表现好的机器人也能升级
- ✅ 失败且表现差的机器人降级到更简单地形
- ✅ 自适应调整，快速收敛到合适难度

## 评估指标

### Success Rate (Rsucc)

$$R_{succ} = \frac{\text{成功次数}}{\text{总尝试次数}}$$

通过 `episode_success_flags` 统计

### Traverse Rate (Rtrav)

$$R_{trav} = \frac{\text{摔倒前行走距离}}{\text{地形长度 (8m)}}$$

通过 `episode_distance_traveled` 统计

### Foothold Error (Efoot)

*待实现* - 需要添加脚部位置跟踪

## 使用方法

### 运行评估

```bash
cd humanoid-gym
python humanoid/scripts/play_terrain_curriculum.py --task=humanoid_ppo
```

### 地形配置

在 `play_terrain_curriculum.py` 中：

```python
# 地形布局
num_rows = 5  # 难度级别 (0.0 ~ 1.0)
num_cols = 2  # 地形类型数量

# 地形比例（最后两个是 beams 和 stones）
terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]
#                                           ↑    ↑
#                                        col0  col1
#                                       beams stones
```

### 查看统计

环境会在 `extras` 中记录：

```python
extras['episode_success_count']      # 本步成功的环境总数
extras['beams_success_count']        # Balancing Beams 成功数
extras['stones_success_count']       # Stones Everywhere 成功数
extras['episode_avg_distance']       # 成功环境的平均距离
```

## 地形坐标系统

### Balancing Beams (8.5m × 8.5m)

```
       ┌─────────────────────────────┐
       │  Border (0.25m)             │
       │  ┌───────────────────────┐  │
       │  │ Start Platform        │  │  ← y = -3.25m
       │  │ (1.5m × 1m)          │  │
       │  ├───────────────────────┤  │
       │  │                       │  │
       │  │   Beams (2m width)    │  │  ← 8m length
       │  │                       │  │
       │  ├───────────────────────┤  │
       │  │ End Platform          │  │  ← y = +3.25m
       │  │ (1.5m × 1m)          │  │
       │  └───────────────────────┘  │
       │  Pit (sides)                │
       └─────────────────────────────┘
```

**成功标准**: `y > +3.0m` (到达终点平台)

### Stones Everywhere (8.5m × 8.5m)

```
       ┌─────────────────────────────┐
       │  Border (0.25m)             │
       │  ┌───────────────────────┐  │
       │  │                       │  │
       │  │   ┌─────────────┐     │  │
       │  │   │   Central   │     │  │
       │  │   │  Platform   │     │  │  ← 4m × 4m
       │  │   │  (4m × 4m)  │     │  │
       │  │   └─────────────┘     │  │
       │  │                       │  │
       │  │  Stones (8m × 8m)     │  │
       │  └───────────────────────┘  │
       │  Pit (corners)              │
       └─────────────────────────────┘
```

**成功标准**: `距中心 > 3.75m` AND `行走 >= 8m`

## 扩展到其他地形

添加新地形类型（例如 type 2 = stairs）：

```python
# 1. 在 _reset_root_states 中添加起始位置
elif terrain_type == 2:  # Stairs
    # 楼梯底部
    self.root_states[env_id, 1] = origin_y - 3.0

# 2. 在 _check_success_criteria 中添加成功标准
stairs_mask = (self.terrain_types == 2)
stairs_success = stairs_mask & (rel_pos[:, 2] > height_threshold)
newly_success |= stairs_success
```

## 设计原则

1. **直接扩展** - 在 `XBotLFreeEnv` 中添加，不创建新类
2. **成功不终止** - 只记录标志，让机器人继续走
3. **个性化配置** - 不同地形不同起始位置和成功标准
4. **最小侵入** - 重写必要方法，保持与原代码兼容

## 文件结构

```
humanoid-gym/
├── humanoid/
│   ├── envs/
│   │   └── custom/
│   │       └── humanoid_env.py         # XBotLFreeEnv (添加个性化功能)
│   └── scripts/
│       └── play_terrain_curriculum.py  # 评估脚本
└── STONEENV_README.md                  # 本文件
```

## 许可证

BSD-3-Clause License  
Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD.

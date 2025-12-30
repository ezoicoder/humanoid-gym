# Stage 1 Virtual Terrain Implementation

## 概述

实现了论文中的 **Stage 1: Soft Terrain Dynamics** 训练方法，允许机器人在安全的平地上行走，同时感知虚拟的石头地形。

### 核心思想

- **物理地形**: 完全平坦（height = 0），机器人不会因为失足而终止
- **虚拟地形**: stones_everywhere 石头模式，用于感知和奖励计算
- **Foothold Reward**: 基于虚拟地形的高度计算（稀疏奖励）
- **Locomotion Rewards**: 基于物理平地计算（密集奖励）
- **Elevation Map**: Actor和Critic感知到的是虚拟地形的15×15高度图

## 实现细节

### 1. 新增地形类型

**文件**: `humanoid/utils/terrain.py`

新增函数 `stones_everywhere_stage1_terrain(terrain, difficulty)`:
- 生成 `terrain.height_field_raw` (物理地形 = 平地)
- 生成 `terrain.height_field_virtual` (虚拟地形 = 石头)
- 地形类型编号: **10**

```python
def stones_everywhere_stage1_terrain(terrain, difficulty=1):
    """
    Stage 1 training terrain: Physical flat ground + Virtual stones perception.
    
    - terrain.height_field_raw: flat terrain (for physics)
    - terrain.height_field_virtual: stones pattern (for perception and reward)
    """
    # 物理地形设为平地
    terrain.height_field_raw[:, :] = 0
    
    # 虚拟地形生成石头模式
    terrain.height_field_virtual = generate_stones_pattern(...)
```

### 2. Terrain类增强

**文件**: `humanoid/utils/terrain.py`

- 添加 `height_field_virtual` 属性存储虚拟高度图
- 添加 `heightsamples_virtual` 暴露给环境使用
- 修改 `add_terrain_to_map()` 支持虚拟地形拼接

### 3. 环境增强

**文件**: `humanoid/envs/base/legged_robot.py`

- `_create_heightfield()` 和 `_create_trimesh()`: 加载 `height_samples_virtual`
- `_get_heights(use_virtual_terrain=False)`: 支持虚拟地形采样

**文件**: `humanoid/envs/custom/humanoid_env.py`

- `_should_use_virtual_terrain()`: 判断是否使用虚拟地形
- `compute_observations()`: 使用虚拟地形生成elevation map
- `_reward_foothold()`: 使用虚拟地形计算foothold reward

```python
def _should_use_virtual_terrain(self):
    """检查是否使用虚拟地形（terrain type 10）"""
    if self.actual_terrain_types is None:
        return False
    return (self.actual_terrain_types == 10).any().item()
```

### 4. 配置文件

**文件**: `humanoid/envs/custom/humanoid_config.py`

新增配置类:
- `XBotLStoneStage1Cfg`: Stage 1 训练环境配置
- `XBotLStoneStage1CfgPPO`: Stage 1 PPO算法配置

```python
class XBotLStoneStage1Cfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        num_rows = 9  # 9个难度级别
        num_cols = 1  # 1种地形类型
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        # 只使用 terrain type 10 (stones_everywhere_stage1)
```

## 使用方法

### 训练 Stage 1

```bash
cd humanoid-gym
python humanoid/scripts/train.py --task=xbotl_stone_stage1
```

### Curriculum 混合训练

如果想在一个curriculum中同时包含真实石头和虚拟石头:

```python
class MixedCurriculumCfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        num_rows = 9
        num_cols = 2  # 2列：一列真实石头，一列虚拟石头
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5]
        # 50% stones_everywhere (type 8), 50% stones_everywhere_stage1 (type 10)
```

## 验证测试

运行测试脚本验证实现:

```bash
cd /workspace_robot
python test_stage1_direct.py
```

测试结果示例:
```
✅ ALL TESTS PASSED!

--- Testing difficulty = 0.0 ---
  Physical terrain: Mean: 0.00, Std: 0.00 (FLAT)
  Virtual terrain: Mean: -22.55, Std: 63.03 (STONES)
  
--- Testing difficulty = 1.0 ---
  Physical terrain: Mean: 0.00, Std: 0.00 (FLAT)
  Virtual terrain: Mean: -92.94, Std: 99.94 (STONES)
```

## 训练流程建议

### Stage 1 → Stage 2 渐进式训练

1. **Stage 1** (安全训练):
   ```bash
   python humanoid/scripts/train.py --task=xbotl_stone_stage1
   ```
   - 在平地上行走，无终止风险
   - 学习感知虚拟石头地形
   - 学习foothold placement策略
   - 训练至收敛（约5-10M steps）

2. **Stage 2** (真实地形):
   ```bash
   python humanoid/scripts/train.py --task=xbotl_stone --load_run=<stage1_run>
   ```
   - 加载Stage 1的policy
   - 在真实石头地形上fine-tune
   - 已有terrain-aware能力，收敛更快

## 技术细节

### 地形类型映射

| Type | Name | Physical | Virtual | Use Case |
|------|------|----------|---------|----------|
| 0-7  | 其他地形 | 真实 | N/A | 正常训练 |
| 8    | stones_everywhere | 真实石头 | N/A | Stage 2 |
| 9    | stepping_stones | 真实石头 | N/A | 正常训练 |
| **10** | **stones_everywhere_stage1** | **平地** | **石头** | **Stage 1** |

### 奖励计算

- **Dense Rewards** (locomotion): 基于物理地形（平地）
  - tracking_lin_vel, tracking_ang_vel
  - orientation, base_height
  - joint_pos, action_smoothness
  - 等等...

- **Sparse Reward** (foothold): 基于虚拟地形（石头）
  - 采样脚底9个点的虚拟地形高度
  - 惩罚落在虚拟pit上的情况
  - 鼓励踩在虚拟石头上

### 感知

- **Elevation Map** (15×15): 基于虚拟地形
  - Actor观察: 虚拟石头的高度图
  - Critic观察: 虚拟石头的高度图
  - 机器人"想象"自己在石头上行走

## 代码修改总结

### 新增文件
- `test_stage1_direct.py`: 测试脚本
- `STAGE1_IMPLEMENTATION.md`: 本文档

### 修改文件
1. `humanoid/utils/terrain.py`
   - 新增 `stones_everywhere_stage1_terrain()` 函数
   - `Terrain.__init__()`: 初始化 `height_field_virtual`
   - `add_terrain_to_map()`: 支持虚拟地形拼接
   - `HumanoidTerrain.make_terrain()`: 添加 type 10

2. `humanoid/envs/base/legged_robot.py`
   - `_create_heightfield()`: 加载 `height_samples_virtual`
   - `_create_trimesh()`: 加载 `height_samples_virtual`
   - `_get_heights()`: 添加 `use_virtual_terrain` 参数

3. `humanoid/envs/custom/humanoid_env.py`
   - `_init_terrain_types()`: 添加 type 10 名称
   - `_should_use_virtual_terrain()`: 新增方法
   - `compute_observations()`: 使用虚拟地形
   - `_reward_foothold()`: 使用虚拟地形

4. `humanoid/envs/custom/humanoid_config.py`
   - 新增 `XBotLStoneStage1Cfg`
   - 新增 `XBotLStoneStage1CfgPPO`

## 常见问题

### Q: 为什么要分Stage 1和Stage 2？

A: Stage 1在平地上训练，避免早期训练时机器人频繁摔倒导致的训练不稳定。机器人可以安全地学习terrain-aware的行为模式，然后在Stage 2迁移到真实地形。

### Q: 虚拟地形会影响物理模拟吗？

A: 不会。物理引擎只使用 `height_field_raw`（平地），虚拟地形仅用于observation和reward计算。

### Q: 可以在一个curriculum中混合使用吗？

A: 可以！设置 `terrain_proportions` 来控制不同地形类型的比例，例如前期多用Stage 1，后期多用真实地形。

### Q: 如何确认使用了虚拟地形？

A: 查看训练日志中的 "TERRAIN TYPE DISTRIBUTION"，应该看到 "Type 10 (stones_everywhere_stage1 (virtual))"。

## 下一步

1. ✅ 实现完成
2. ✅ 测试通过
3. ⏳ 运行完整训练验证收敛性
4. ⏳ 对比Stage 1 → Stage 2 vs 直接Stage 2的训练效率
5. ⏳ 调优超参数（reward scales, curriculum难度等）

---

**实现日期**: 2025-12-30  
**作者**: AI Assistant  
**状态**: ✅ 完成并测试通过


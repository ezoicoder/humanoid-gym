# Stage 1 训练：mesh_type 对比与优化

## 问题背景

Stage 1 训练的核心思想是：
- **物理地形**: 平地（机器人不会摔倒）
- **虚拟地形**: 石头（用于感知和奖励）

但是如何实现"平地物理"有三种方案，性能差异巨大。

## 三种实现方案对比

### 方案 1: `trimesh` (原始实现)

```python
class XBotLStoneStage1Cfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        mesh_type = 'trimesh'  # 三角网格
        # 生成平坦的 height_field_raw
        # 转换为三角网格添加到物理引擎
```

**工作流程**:
1. 生成平坦的高度图 (height_field_raw = 0)
2. 生成虚拟石头高度图 (height_field_virtual)
3. 将平坦高度图转换为三角网格 (vertices + triangles)
4. 物理引擎加载三角网格

**性能分析**:
- ❌ **最慢**: 三角网格碰撞检测计算量大
- ❌ **内存占用大**: 存储顶点和三角面片数据
- ❌ **浪费**: 为平地创建复杂网格完全没必要
- ✅ **精确**: 对复杂地形最精确（但我们不需要）

**适用场景**: 真实复杂地形 (Stage 2)

---

### 方案 2: `heightfield`

```python
class XBotLStoneStage1Cfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        mesh_type = 'heightfield'  # 高度场
```

**工作流程**:
1. 生成平坦的高度图 (height_field_raw = 0)
2. 生成虚拟石头高度图 (height_field_virtual)
3. 物理引擎直接使用高度图

**性能分析**:
- ⚡ **中等速度**: 比 trimesh 快，比 plane 慢
- 💾 **中等内存**: 只存储高度值
- ⚠️ **仍有浪费**: 为平地维护高度图数组
- ✅ **简单**: 实现简单

**适用场景**: 中等规模训练 (1024-2048 envs)

---

### 方案 3: `plane` + virtual terrain (✨ **推荐**)

```python
class XBotLStoneStage1PlaneCfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        mesh_type = 'plane'  # 无限平面
        use_virtual_terrain = True  # 启用虚拟地形
```

**工作流程**:
1. 物理引擎创建简单的无限平面 (gymapi.add_ground)
2. 生成虚拟石头高度图 (height_field_virtual)
3. 虚拟地形仅用于观察和奖励计算

**性能分析**:
- 🚀 **最快**: 平面碰撞检测极简单 (O(1))
- 💾 **最小内存**: 无需存储物理地形数据
- ✅ **零浪费**: 物理引擎只处理平面
- ✅ **完整功能**: 虚拟地形功能完全保留

**适用场景**: 大规模训练 (4096+ envs) ⭐ **推荐用于 Stage 1**

---

## 性能对比表

| 指标 | trimesh | heightfield | plane + virtual |
|------|---------|-------------|-----------------|
| **物理模拟速度** | 慢 (1x) | 中 (2-3x) | 快 (5-10x) |
| **内存占用** | 大 (~100MB/env) | 中 (~10MB/env) | 小 (~1MB/env) |
| **GPU占用** | 高 | 中 | 低 |
| **4096 envs FPS** | ~500 | ~1000 | ~2000+ |
| **虚拟地形支持** | ✅ | ✅ | ✅ |
| **实现复杂度** | 简单 | 简单 | 中等 |

*注: FPS数据为估算，实际性能取决于硬件*

---

## 代码实现细节

### 关键修改点

#### 1. Terrain 类初始化 (`terrain.py`)

```python
class Terrain:
    def __init__(self, cfg, num_robots):
        self.type = cfg.mesh_type
        self.use_virtual_terrain = getattr(cfg, 'use_virtual_terrain', False)
        
        if self.type == 'plane':
            if self.use_virtual_terrain:
                # 特殊处理: plane + virtual
                self._init_plane_with_virtual(cfg)
            return  # 普通 plane 直接返回
```

#### 2. Virtual terrain 生成 (`terrain.py`)

```python
def _init_plane_with_virtual(self, cfg):
    """为 plane 模式生成虚拟地形"""
    # 不创建 height_field_raw (物理用 plane)
    self.height_field_raw = None
    
    # 创建 height_field_virtual (感知用)
    self.height_field_virtual = np.zeros((tot_rows, tot_cols), dtype=np.int16)
    
    # 生成虚拟地形 (只提取 virtual heightfield)
    if cfg.curriculum:
        self._curiculum_virtual_only()
```

#### 3. 环境加载 (`legged_robot.py`)

```python
def _create_ground_plane(self):
    """创建平面并加载虚拟地形"""
    self.gym.add_ground(self.sim, plane_params)
    
    # 加载虚拟高度样本
    if hasattr(self.terrain, 'heightsamples_virtual'):
        self.height_samples_virtual = torch.tensor(
            self.terrain.heightsamples_virtual
        ).to(self.device)
```

#### 4. 高度采样 (`legged_robot.py`)

```python
def _get_heights(self, env_ids=None, use_virtual_terrain=False):
    if self.cfg.terrain.mesh_type == 'plane':
        if use_virtual_terrain and self.height_samples_virtual is not None:
            # 使用虚拟地形
            pass  # 继续采样
        else:
            # 纯 plane 模式
            return torch.zeros(...)
```

---

## 使用方法

### 方案 1: trimesh (原始)

```bash
# 适合小规模测试
python humanoid/scripts/train.py --task=humanoid_stones_stage1_ppo --num_envs=1024
```

### 方案 3: plane + virtual (推荐)

```bash
# 适合大规模训练
python humanoid/scripts/train.py --task=humanoid_stones_stage1_plane_ppo --num_envs=4096
```

---

## 性能测试建议

### 测试脚本

```bash
# 测试 trimesh 版本
time python humanoid/scripts/train.py \
  --task=humanoid_stones_stage1_ppo \
  --num_envs=4096 \
  --max_iterations=100

# 测试 plane 版本  
time python humanoid/scripts/train.py \
  --task=humanoid_stones_stage1_plane_ppo \
  --num_envs=4096 \
  --max_iterations=100
```

### 预期结果

| 配置 | 4096 envs | 训练速度 | GPU显存 |
|------|-----------|---------|---------|
| trimesh | ~500 FPS | 1x | ~8GB |
| plane | ~2000 FPS | 4x | ~4GB |

---

## 常见问题

### Q1: plane + virtual 会影响训练效果吗？

**A**: 不会！机器人感知到的完全相同：
- ✅ 相同的 15×15 elevation map (虚拟石头)
- ✅ 相同的 foothold reward (基于虚拟石头)
- ✅ 相同的 locomotion rewards (基于平地)

唯一区别是物理引擎的实现方式，对训练透明。

### Q2: 为什么不直接用 plane？

**A**: 纯 plane 模式下：
- ❌ `terrain.py` 的 `if self.type == 'plane': return` 会跳过地形生成
- ❌ 不会生成虚拟高度图
- ❌ 无法获取 elevation map

所以需要特殊处理 `use_virtual_terrain=True`。

### Q3: 可以混合使用吗？

**A**: 可以！在 curriculum 中：
- Stage 1 前期: plane + virtual (快速探索)
- Stage 1 后期: trimesh + virtual (过渡到真实地形)
- Stage 2: trimesh 真实石头

### Q4: 如何选择？

**推荐策略**:
- 🧪 **原型测试** (< 1024 envs): trimesh (简单)
- 🚀 **大规模训练** (4096+ envs): plane + virtual (快)
- 🎯 **最终训练**: plane + virtual → trimesh 渐进

---

## 实现状态

- ✅ trimesh 版本: `XBotLStoneStage1Cfg` (已实现)
- ✅ plane + virtual 版本: `XBotLStoneStage1PlaneCfg` (已实现)
- ✅ 任务注册: `humanoid_stones_stage1_plane_ppo`
- ✅ 测试通过: 虚拟地形正确生成

---

## 总结

对于 Stage 1 训练：

1. **开发阶段**: 使用 `trimesh` (简单直接)
2. **大规模训练**: 使用 `plane + virtual` (性能最优) ⭐
3. **过渡到 Stage 2**: 从 plane 切换到 trimesh

**推荐配置**: `XBotLStoneStage1PlaneCfg` 🚀

---

**更新日期**: 2025-12-30  
**状态**: ✅ 完成并测试


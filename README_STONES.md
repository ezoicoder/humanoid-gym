# Stones Terrain 训练与演示指南

## 重要提示
⚠️ **所有命令必须在 `humanoid-gym/humanoid` 目录下运行**

## 1. 训练脚本

### 基本用法

```bash
script -c "python scripts/train.py --task=humanoid_stones_stage1_plane_ppo --run_name v3p --double_critic --headless --platform_width=3.0 --platform_length=3.0 --resume --load_run=Jan01_03-22-16_v3 --checkpoint={xxx}" -f ../logs/stage1_plane_v3p.txt
```

### 参数详解

| 参数 | 说明 |
|------|------|
| `script -c "..." -f` | Linux 命令，将终端输出保存到日志文件 |
| `--task` | 训练任务名称，此处为 `humanoid_stones_stage1_plane_ppo` |
| `--run_name` | 本次运行的名称（用于区分实验），⚠️ **不能包含下划线** |
| `--double_critic` | 启用双评论器：Critic1 处理密集奖励(运动)，Critic2 处理稀疏奖励(落脚点) |
| `--headless` | 无界面模式，训练时不显示 Isaac Gym 窗口，提升速度 |
| `--platform_width` | 中央平台宽度（米），控制虚拟地形难度 |
| `--platform_length` | 中央平台长度（米），控制虚拟地形难度 |
| `--resume` | 从检查点恢复训练 |
| `--load_run` | 指定要加载的运行名称（时间戳+名称格式，如 `Jan01_03-22-16_v3`） |
| `--checkpoint` | 模型检查点编号，替换 `{xxx}` 为具体数字（如 `5000`），`-1` 表示最新 |
| `-f ../logs/xxx.txt` | 日志输出文件路径 |

### 示例说明

上述命令表示：
- 从 `Jan01_03-22-16_v3` 运行的第 `xxx` 个检查点恢复训练
- 新运行命名为 `v3p`
- 使用双评论器架构
- 平台尺寸为 3.0m × 3.0m
- 训练日志保存到 `../logs/stage1_plane_v3p.txt`

---

## 2. 演示脚本（Demo/Play）

### 基本用法

```bash
python scripts/play_terrain_curriculum.py --task=humanoid_stones_stage1_plane_ppo --run_name v3p --platform_width=3.0 --platform_length=3.0
```

### 参数详解

| 参数 | 说明 |
|------|------|
| `--task` | 演示任务名称，需与训练时保持一致 |
| `--run_name` | 运行名称，用于加载对应的训练模型 |
| `--platform_width` | 平台宽度（米），需与训练配置一致 |
| `--platform_length` | 平台长度（米），需与训练配置一致 |

### 可选参数

**训练脚本的许多参数同样适用于演示脚本**，例如：

```bash
python scripts/play_terrain_curriculum.py \
  --task=humanoid_stones_stage1_plane_ppo \
  --run_name v3p \
  --platform_width=3.0 \
  --platform_length=3.0 \
  --load_run=Jan01_03-22-16_v3p \
  --checkpoint=10000
```

| 参数 | 说明 |
|------|------|
| `--load_run` | 指定要加载的模型运行名称 |
| `--checkpoint` | 指定检查点编号，`-1` 为最新 |
| `--num_envs` | 环境（机器人）数量 |

---

## 3. 注意事项

1. ⚠️ `--run_name` 参数**不能包含下划线** `_`，请使用其他分隔符（如数字、字母）
2. 训练时使用 `--headless` 加速，演示时去掉此参数可查看可视化
3. `--platform_width` 和 `--platform_length` 会影响地形难度，训练和演示时可以保持一致
4. 检查点文件位于 `logs/{实验名}/{运行名}/model_{编号}.pt`，其中实验名（experiment name）可在 `humanoid/envs/__init__.py` 中根据 task name查找。

---

## 4. 项目结构解读

### 4.1 核心训练任务（`humanoid/envs/__init__.py`）

项目在 `humanoid/envs/__init__.py` 中注册了多个训练任务：

```python
task_registry.register("humanoid_stones_ppo", ...)
task_registry.register("humanoid_stones_stage1_ppo", ...)
task_registry.register("humanoid_stones_stage1_plane_ppo", ...)
```

#### 任务对比

| 任务名称 | 物理地形 | 虚拟地形 | 训练速度 | 适用场景 |
|---------|---------|---------|---------|---------|
| `humanoid_stones_ppo` | Trimesh（真实石头） | ❌ | 慢 | Stage2 训练 |
| `humanoid_stones_stage1_ppo` | Trimesh（石头） | ✅ | 较慢 | Stage1 训练（完整物理） |
| `humanoid_stones_stage1_plane_ppo` | **Plane（平面）** | ✅ | **快** | ⭐ **推荐：Stage1 大规模训练** |

**关键区别**：
- **`humanoid_stones_stage1_plane_ppo`**（推荐）：
  - 物理模拟运行在**平面**上（plane），计算开销极低
  - 机器人通过**虚拟高度场**（virtual heightfield）感知石头地形
  - 速度比 trimesh 版本快数倍，适合 4096+ 环境并行训练
  
- **`humanoid_stones_stage1_ppo`**：
  - 物理模拟使用 **trimesh**，真实石头碰撞
  - 训练效果与 plane 版本相似，但速度慢得多
  - 适合小规模训练或需要完整物理仿真的场景

### 4.2 双评论器架构（Double Critic）

项目实现了双评论器架构，用于分别处理**密集奖励**和**稀疏奖励**。

#### 修改位置

主要修改在 `algo/ppo/` 目录下：
- **`actor_critic.py`**：定义双 Critic 网络结构
- **`ppo.py`**：实现双 Critic 的训练逻辑和优势函数加权
- **`rollout_storage.py`**：存储两个 Critic 的观测和值

#### 工作原理

```
Critic 1 → 密集奖励（Dense Rewards）
          - 运动控制奖励（速度跟踪、姿态保持等）
          - 权重：w1 = 1.0

Critic 2 → 稀疏奖励（Sparse Rewards）  
          - 落脚点奖励（foothold reward）
          - 权重：w2 = 0.25

最终优势函数 = w1 × A1 + w2 × A2
```

启用方式：在训练命令中添加 `--double_critic` 参数。

### 4.3 落脚点奖励（Foothold Reward）

**定义位置**：`humanoid/envs/custom/humanoid_env.py`

关键方法：`_reward_foothold()` （第 889 行）

#### 奖励逻辑

```
r_foothold = -∑(i=1→2) C_i · ∑(j=1→n) 𝟙{d_ij < ε}
```

其中：
- `C_i`：脚 i 的接触状态（1=接触，0=离地）
- `d_ij`：脚上采样点 j 的地形高度
- `ε`：深度容忍阈值（如 0.03m）

**目标**：惩罚机器人将脚放在不安全位置（例如脚掌采样点明显低于期望高度），鼓励踩在稳固的石头或平台上。

#### 虚拟地形支持

```python
use_virtual = getattr(self.cfg.terrain, 'use_virtual_terrain', False)
if self.cfg.terrain.mesh_type == 'plane' and use_virtual:
    # 使用虚拟地形高度场计算 foothold reward
```

在 `humanoid_stones_stage1_plane_ppo` 任务中，虽然物理地形是平面，但 foothold reward 依然基于**虚拟石头地形高度场**计算，确保训练效果。

### 4.4 地形生成（`humanoid/utils/terrain.py`）

项目实现了多种复杂地形生成函数：

| 地形类型 | 函数名称 | 特点 |
|---------|---------|------|
| 平衡木 | `balancing_beams_terrain()` | 窄木条，两侧深坑，考验平衡能力 |
| 石头遍布 | `stones_everywhere_terrain()` | 随机分布的石头，密集落脚点挑战 |
| 跳石 | `stepping_stones_terrain()` | 交错排列的石头，需精准落脚 |

每种地形都支持**难度参数**（difficulty），通过课程学习（curriculum）逐步提升挑战。

### 4.5 环境定义（`humanoid/envs/custom/humanoid_env.py`）

**核心环境类**：`XBotLFreeEnv`（继承自 `LeggedRobot`）

主要实现：
- **奖励函数集合**：包括速度跟踪、姿态稳定、能量消耗、foothold 等
- **观测计算**：机器人状态、地形感知、高度扫描
- **虚拟地形支持**：在 plane 物理环境中叠加虚拟地形感知

### 4.6 配置文件（`humanoid/envs/custom/humanoid_config.py`）

每个训练任务对应两个配置类：
- **环境配置**（如 `XBotLStoneStage1PlaneCfg`）：定义地形类型、物理参数、奖励权重等
- **PPO 配置**（如 `XBotLStoneStage1PlaneCfgPPO`）：定义网络结构、学习率、训练步数等

**Stage1 Plane 配置关键设置**：
```python
class terrain:
    mesh_type = 'plane'              # 物理地形：平面
    use_virtual_terrain = True       # 启用虚拟地形
    curriculum = True                # 课程学习
    terrain_proportions = [0,0,0,0,0,0,0,0,0,0,1.0]  # 仅使用 stage1_stones
```

### 4.7 文件结构总览

```
humanoid/
├── envs/
│   ├── __init__.py                    # 任务注册
│   ├── custom/
│   │   ├── humanoid_env.py           # 环境实现（包含 foothold reward）
│   │   └── humanoid_config.py        # 配置定义
│   └── base/
│       ├── legged_robot.py           # 基类
│       └── base_task.py
├── algo/
│   └── ppo/
│       ├── actor_critic.py           # 双 Critic 网络
│       ├── ppo.py                    # 双 Critic 训练逻辑
│       └── rollout_storage.py        # 双 Critic 数据存储
├── utils/
│   ├── terrain.py                    # 地形生成函数
│   └── task_registry.py              # 任务注册工具
└── scripts/
    ├── train.py                       # 训练脚本
    └── play_terrain_curriculum.py    # 演示脚本
```

---

## 5. 可视化与录屏

### 5.1 创建虚拟显示屏幕

在无显示器的服务器环境（如远程训练服务器）上运行 Isaac Gym 时，需要创建虚拟显示：

```bash
/workspace_robot/start_vnc_simple.sh
```

**说明**：
- 该脚本会启动 VNC 服务，创建虚拟显示环境
- 启动后可以通过 VNC 客户端连接查看 Isaac Gym 的可视化窗口
- 训练时去掉 `--headless` 参数即可在虚拟屏幕中显示仿真画面

### 5.2 录制演示视频

需要录制演示视频时，使用录屏脚本：

```bash
/workspace_robot/screenshot.sh
```

**使用流程**：
1. 启动 VNC 虚拟屏幕（如果尚未启动）
2. 运行演示脚本（`play_terrain_curriculum.py`），去掉 `--headless`
3. 执行 `screenshot.sh` 开始录屏
4. 停止脚本后保存录屏文件

### 5.3 可视化使用建议

| 场景 | 建议配置 | 说明 |
|------|---------|------|
| **大规模训练** | `--headless` | 无界面模式，最快速度 |
| **监控训练** | VNC + 少量环境 | 可视化查看机器人行为，用于调试 |
| **演示展示** | VNC + 录屏 | 生成演示视频 |
| **本地测试** | 本地 GUI | 有显示器时直接显示 |

**提示**：
- 训练阶段建议使用 `--headless` 以获得最佳性能
- 需要观察机器人行为时，可启动 VNC 并减少环境数量（如 `--num_envs=16`）
- 录屏会消耗额外资源，建议在演示模式下使用

---

## 6. 未来优化与发展方向

### 6.1 待调试问题

#### 1. Heightmap 计算准确性
**当前状态**：虚拟地形使用 heightmap 进行感知和 foothold reward 计算

**待验证**：
- Heightmap 计算是否准确反映地形几何
- 是否包含噪声（观测噪声 vs 奖励计算噪声）
- 采样分辨率是否足够（当前 `horizontal_scale = 0.02`）

**调试建议**：
- 可视化对比真实地形与 heightmap 采样结果
- 检查 `terrain.py` 中地形生成与高度场转换的一致性
- 验证 `_reward_foothold()` 中的高度查询逻辑

#### 2. Foothold Reward 采样密度
**当前状态**：脚部采样点数量可能有限

**改进方向**：
- 增加每只脚的采样点数量（提高检测精度）
- 优化采样点分布（覆盖脚掌关键区域）
- 可能需要权衡计算开销与奖励准确性

**实现位置**：`humanoid_env.py` 的 `_reward_foothold()` 方法

#### 3. 地形生成数量优化
**问题**：当前地形生成数量可能限制了训练多样性和并行效率

**改进方案**：
- 增加 `num_cols`（地形列数），生成更多地形变体
- 配合增加 `num_envs`（环境数量）以充分利用 GPU
- 示例：`num_cols=512` + `num_envs=8192` 可显著加快采样效率

**配置位置**：`humanoid_config.py` 中的 `terrain.num_cols` 和 `env.num_envs`

### 6.2 功能增强方向

#### 4. Viewer 性能优化
**目标**：在保留可视化的同时提升速度

**当前痛点**：
- `--headless` 快但看不到
- 开启可视化慢，且录屏进一步降速

**可能方案**：
- 探索 Isaac Gym 的异步渲染选项
- 降低渲染分辨率/帧率
- 使用 offscreen rendering + 后处理录屏（技术难度较高）
- 分离训练与可视化：快速训练 + 定期加载检查点演示

#### 5. 丰富奖励设计
**当前状态**：主要依赖速度跟踪、姿态稳定、foothold 奖励

**新增奖励建议**：

| 奖励类型 | 目的 | 实现思路 |
|---------|------|---------|
| **抬脚高度奖励** | 鼓励足够的离地间隙 | 检测脚部离地时的最大高度，奖励 > 阈值的抬脚 |
| **跨步跨度奖励** | 鼓励大步幅通过障碍 | 测量相邻脚步之间的水平距离 |
| **落脚精准度** | 精准踩在石头中心 | 基于脚部中心与石头中心的距离 |
| **动态平衡** | 减少摇晃 | 惩罚躯干角速度的高频波动 |
| **能量效率** | 优化步态能耗 | 当前已有，可调整权重 |

**实现位置**：在 `humanoid_env.py` 中添加对应的 `_reward_xxx()` 方法，并在配置文件中设置权重。

### 6.3 训练路线图

```
当前阶段: Stage 1 训练中
         ├─ 使用 humanoid_stones_stage1_plane_ppo
         ├─ 双 Critic 架构
         └─ 虚拟地形 + 平面物理

下一步: Stage 1 评估与调优
        ├─ 验证 heightmap 准确性
        ├─ 调整 foothold reward 采样
        └─ 监控训练指标（成功率、步态质量）

最终目标: Stage 2 真实地形部署
          ├─ 任务: humanoid_stones_ppo (trimesh)
          ├─ 加载 Stage 1 训练的策略
          ├─ 在真实石头地形上微调
          └─ 验证跨越复杂地形的能力
```

### 6.4 Debug 检查清单

在进入 Stage 2 之前，建议完成以下检查：

- [ ] **Stage 1 训练收敛**：奖励曲线稳定，成功率 > 80%
- [ ] **Foothold reward 有效**：观察机器人是否主动避开"危险区域"
- [ ] **步态自然**：可视化检查机器人行走是否流畅
- [ ] **Heightmap 准确**：对比虚拟地形与实际感知
- [ ] **采样数量足够**：foothold 检测不遗漏脚掌边缘
- [ ] **地形多样性**：确保训练覆盖不同难度和布局

**最终期望**：Stage 1 训练出的策略能在虚拟地形上稳定行走，为 Stage 2 的真实地形部署打下坚实基础。

---

## 附录：常见问题

**Q: Stage 1 和 Stage 2 的区别是什么？**
- Stage 1：虚拟地形感知 + 平面物理，快速训练基础策略
- Stage 2：真实 trimesh 地形，完整物理接触，策略微调与验证

**Q: 为什么 Stage 1 训练结果还没体现？**
- 需要足够的训练迭代（通常数千到上万次）
- 调试期可能需要调整奖励权重、网络结构等超参数
- 建议监控 tensorboard 日志，观察奖励分量的变化趋势

**Q: 如何判断训练是否成功？**
- 查看演示脚本中机器人是否能稳定前进
- 检查 foothold reward 是否从负值逐渐提升
- 观察跌倒率是否显著下降


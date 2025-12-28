# 平衡木地形详细文档 (Balancing Beams Terrain)

## 注意，目前没验证过训练脚本，下面的训练脚本有待观察！

Humanoid-Gym的设计思路是训练场景和演示场景解耦，可能强调一个场景训练的能力到另一个场景的展示，所以 play.py 强调的是平面场景的演示 ,play_balancing_beams则是平衡木场景的演示。

Humanoid-Gym 框架中平衡木地形的完整使用指南。

## 目录
- [概述](#概述)
- [地形规格](#地形规格)
- [快速开始](#快速开始)
- [训练](#训练)
- [评估与可视化](#评估与可视化)
- [配置详情](#配置详情)
- [代码结构](#代码结构)
- [实现细节](#实现细节)

## 概述

**平衡木地形**是一个具有挑战性的运动任务，旨在测试人形机器人的平衡能力、精确性和适应能力。该地形包括：

- **窄踩踏石**，按之字形排列
- **可变难度级别**（0-8），调整梁宽、间隙和横向偏移
- **平台镶边**，提供视觉清晰度和安全性
- **起始和结束平台**，用于机器人初始化

该地形受鲁棒人形运动研究的启发，为评估策略的鲁棒性和泛化能力提供了严格的基准。

## 地形规格

### 物理尺寸
- **总地形尺寸**：2.5m（宽）× 8.5m（长）
- **有效穿越区域**：1.5m × ~6m
- **镶边宽度**：四周各 0.25m（高度 = 0，与平台相同）
- **出发和抵达平台尺寸**：1.5m（宽）× 1.0m（长）
- **深坑深度**：地面以下 1.0m

### 难度参数

地形支持 9 个离散难度级别（l = 0, 1, 2, ..., 8），每个级别从连续难度值（0.0 到 1.0）映射而来：

| 难度级别 (l) | 梁宽 | 横向偏移 | 前进步长 |
|-------------|------|---------|---------|
| 0 | 0.30m | 0.40m | 0.20m |
| 1 | 0.30m | 0.35m | 0.20m |
| 2 | 0.30m | 0.30m | 0.20m |
| 3 | 0.25m | 0.25m | 0.25m |
| 4 | 0.25m | 0.20m | 0.30m |
| 5 | 0.25m | 0.15m | 0.35m |
| 6 | 0.20m | 0.10m | 0.35m |
| 7 | 0.20m | 0.05m | 0.40m |
| 8 | 0.20m | 0.00m | 0.20m |

**参数说明：**
- **梁宽**：每块踩踏石在横向（x）方向的宽度
- **横向偏移**：从中心线的最大之字形振幅
- **前进步长**：连续石头之间在前进（y）方向的距离
- **高度变化**：每块石头随机 ±0.05m

### 坐标系统
- **x 方向**：横向（左右），宽度 = 2.5m
- **y 方向**：前进（起点到终点），长度 = 8.5m
- **z 方向**：高度（垂直），深坑在 -1.0m，地面/平台在 0.0m

## 快速开始

### 1. 训练专门策略

专门针对平衡木训练策略：

```bash
cd humanoid-gym/humanoid
python scripts/train.py --task=humanoid_balancing_beams_ppo --headless --num_envs 4096
```

**配置说明**：使用 `XBotLBalancingBeamsCfg`，包括：
- 单个地形（1 行 × 1 列）
- 固定难度 = 0.8
- 地形分辨率：0.02m（2cm）

### 2. 多难度级别可视化

```bash
python scripts/play_balancing_beams.py --task=humanoid_balancing_beams_ppo --run_name your_run_name
```

同时显示 5 个机器人，每个在不同难度级别（0.0, 0.25, 0.5, 0.75, 1.0）。

### 3. 测试零样本泛化

评估通用行走策略在平衡木上的迁移效果：

```bash
python scripts/play_balancing_beams.py --task=humanoid_ppo --run_name your_general_policy
```

测试策略在未见过的挑战性地形上的鲁棒性。

## 训练

### 任务注册

平衡木任务在 `humanoid/envs/__init__.py` 中注册：

```python
task_registry.register(
    "humanoid_balancing_beams_ppo",
    XBotLFreeEnv,
    XBotLBalancingBeamsCfg(),
    XBotLBalancingBeamsCfgPPO()
)
```

### 训练配置

在 `humanoid/envs/custom/humanoid_config.py` 中定义：

```python
class XBotLBalancingBeamsCfg(XBotLCfg):
    class terrain(XBotLCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False
        selected = True
        terrain_kwargs = {
            'type': 'balancing_beams_terrain',
            'terrain_kwargs': {'difficulty': 0.8}
        }
        horizontal_scale = 0.02  # 2cm 分辨率
        terrain_width = 2.5      # 包含镶边的总宽度
        terrain_length = 8.5     # 包含镶边的总长度
        num_rows = 1             # 单个地形
        num_cols = 1
```

### 训练参数

- **分辨率**：0.02m（125 × 425 像素）
- **难度**：固定为 0.8（对应级别 6）
- **课程学习**：禁用，以保持训练一致性
- **环境数量**：推荐 4096 以获得最佳采样

### 训练命令选项

```bash
# 基础训练
python scripts/train.py --task=humanoid_balancing_beams_ppo --headless

# 自定义环境数量
python scripts/train.py --task=humanoid_balancing_beams_ppo --headless --num_envs 8192

# 从检查点恢复
python scripts/train.py --task=humanoid_balancing_beams_ppo --resume --load_run <run_id>

# 自定义运行名称
python scripts/train.py --task=humanoid_balancing_beams_ppo --run_name my_experiment --headless
```

## 评估与可视化

### 脚本：`play_balancing_beams.py`

该脚本提供可视化和评估功能：

**功能特性：**
- 同时显示 5 个机器人，跨 5 个难度级别
- 导出策略为 JIT 模块
- 录制视频输出
- 支持专门策略和通用策略

**关键配置覆盖：**
```python
env_cfg.env.num_envs = 5           # 每个难度级别一个
env_cfg.terrain.num_rows = 5       # 5 个难度级别
env_cfg.terrain.num_cols = 1       # 单一地形类型
```

### 可视化命令

```bash
# 专门策略
python scripts/play_balancing_beams.py --task=humanoid_balancing_beams_ppo --run_name v1

# 通用策略（零样本）
python scripts/play_balancing_beams.py --task=humanoid_ppo --run_name baseline

# 加载特定检查点
python scripts/play_balancing_beams.py --task=humanoid_balancing_beams_ppo --checkpoint 1000
```

### 输出

- **JIT 策略导出**：`logs/XBot_balancing_beams_ppo/exported/policies/`
- **视频录制**：`videos/XBot_balancing_beams_ppo/<timestamp>_<run_name>.mp4`
- **控制台日志**：回合奖励和统计信息

## 配置详情

### 文件位置

| 组件 | 文件路径 | 说明 |
|------|----------|------|
| 地形生成 | `humanoid/utils/terrain.py` | `balancing_beams_terrain()` 函数 |
| 环境配置 | `humanoid/envs/custom/humanoid_config.py` | `XBotLBalancingBeamsCfg` 类 |
| 训练配置 | `humanoid/envs/custom/humanoid_config.py` | `XBotLBalancingBeamsCfgPPO` 类 |
| 演示脚本 | `humanoid/scripts/play_balancing_beams.py` | 可视化入口点 |
| 任务注册 | `humanoid/envs/__init__.py` | 任务注册表入口 |

## 代码结构

### 目录组织

```
humanoid-gym/
├── humanoid/
│   ├── algo/              # 强化学习算法（如 PPO 实现）
│   ├── envs/              # 环境配置和类
│   │   ├── base/          # 足式机器人基类
│   │   │   └── legged_robot_config.py  # 基础配置
│   │   ├── custom/        # 特定机器人实现
│   │   │   └── humanoid_config.py      # ⭐ 包含 XBotLBalancingBeamsCfg
│   │   └── __init__.py                  # ⭐ 任务注册
│   ├── scripts/           # 可执行脚本
│   │   ├── train.py                     # 训练入口
│   │   ├── play.py                      # 标准可视化
│   │   └── play_balancing_beams.py      # ⭐ 平衡木可视化
│   └── utils/             # 工具模块
│       └── terrain.py                   # ⭐ 包含 balancing_beams_terrain()
├── resources/             # 机器人资产（URDF, MJCF）
└── logs/                  # 训练日志和导出模型
```

**⭐ 标记的文件是平衡木任务的核心组件**

### 地形生成参数

位于 `terrain.py::balancing_beams_terrain()`：

```python
def balancing_beams_terrain(terrain, difficulty=1):
    """
    参数:
        terrain: SubTerrain 对象，包含像素网格
        difficulty: float [0.0, 1.0] → 离散级别 [0, 8]
    
    布局:
        - 镶边: 0.25m (高度=0)
        - 起始平台: 2m × 1m (高度=0)
        - 踩踏石: 之字形排列，参数可变
        - 结束平台: 2m × 1m (高度=0)
        - 深坑: 其他区域 (深度=-1m)
    """
```

### 设计约定

遵循 Humanoid-Gym 的约定：
- **num_rows**：表示难度递进
- **num_cols**：表示地形类型变体
- **curriculum**：humanoid_ppo的训练方式，每个环境的机器人会根据当前难度的完成情况动态调整难度
- **selected**：使用指定参数的单一地形类型

## 实现细节

### 地形生成算法

1. **初始化深坑**：将整个地形设置为 -1.0m 深度
2. **创建镶边框架**：地面水平（0.0m）0.25m 宽的镶边
3. **放置起始平台**：开始处 2m × 1m
4. **生成踩踏石**：
   - 根据难度级别计算参数
   - 按之字形排列石头（左右交替）
   - 应用随机高度变化（±0.05m）
   - 继续直到到达结束平台区域
5. **放置结束平台**：末端 2m × 1m

### 关键实现特性

**动态缩放：**
```python
# 所有尺寸根据地形属性计算
stone_size_pixels = int(stone_size / terrain.horizontal_scale)
border_width_pixels = int(border_width / terrain.horizontal_scale)
```

**自动难度映射：**
```python
# 在多行的 selected_terrain 模式中
if self.cfg.num_rows > 1:
    kwargs['difficulty'] = i / (self.cfg.num_rows - 1)
```

**坐标系统处理：**
```python
# SubTerrain 使用 [width, length] 索引
terrain.height_field_raw[x1:x2, y1:y2] = height_value
# x 轴: width（横向）
# y 轴: length（前进）
```

### 像素分辨率权衡

| 分辨率 | 像素数 | 视觉质量 | 内存占用 | 生成时间 |
|--------|--------|---------|---------|---------|
| 0.05m | 50 × 170 | 低 | 低 | 快 |
| 0.02m | 125 × 425 | 好 | 中等 | 中等 |
| 0.01m | 250 × 850 | 高 | 高 | 慢 |
| 0.004m | 625 × 2125 | 很高 | 很高 | 很慢 |

**推荐**：0.02m（当前默认）提供良好平衡。

## 高级用法

### 自定义难度级别

在 `humanoid_config.py` 中修改难度：

```python
terrain_kwargs = {
    'type': 'balancing_beams_terrain',
    'terrain_kwargs': {'difficulty': 0.5}  # 调整 0.0-1.0
}
```

### 课程学习

启用渐进难度：

```python
class terrain(XBotLCfg.terrain):
    curriculum = True
    num_rows = 10  # 10 个难度级别
```

### 多种地形类型

在随机模式中组合其他地形：

```python
class terrain(XBotLCfg.terrain):
    curriculum = False
    selected = False  # 启用随机模式
    terrain_proportions = [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2]
    # 最后一个元素 (0.2) = 20% 平衡木
```

### 自定义地形尺寸

调整地形大小：

```python
terrain_width = 3.0   # 更宽的地形
terrain_length = 10.0 # 更长的地形
# 注意: 如需要，在 terrain.py 中调整 border_width
```

## 故障排除

### 常见问题

**问题**：地形显示为平面/空白
- **原因**：错误的任务或 mesh_type='plane'
- **解决**：确保使用 `--task=humanoid_balancing_beams_ppo` 或在 play 脚本中覆盖地形

**问题**：机器人立即摔倒
- **原因**：策略未针对此地形训练
- **解决**：专门训练或调整零样本迁移预期

**问题**：石头太窄/太宽
- **原因**：分辨率或难度不匹配
- **解决**：调整 `horizontal_scale` 或 `difficulty` 参数

**问题**：视觉伪影/间隙
- **原因**：分辨率太低
- **解决**：减小 `horizontal_scale`（例如 0.02 → 0.01）

### 性能提示

1. **训练**：使用 4096+ 环境以获得更好的样本多样性
2. **可视化**：将 `num_envs` 降至 5 或更少以实现流畅渲染
3. **内存**：更高分辨率需要更多 GPU 内存
4. **调试**：暂时使用 `mesh_type='plane'` 来隔离问题

## 引用

如果您在研究中使用平衡木地形，请引用：

```bibtex
@article{gu2024humanoid,
  title={Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer},
  author={Gu, Xinyang and Wang, Yen-Jen and Chen, Jianyu},
  journal={arXiv preprint arXiv:2404.05695},
  year={2024}
}
```

## 技术支持

如有问题或建议：
- 在此仓库中创建 Issue
- 联系邮箱：support@robotera.com


# 🎮 使用命令行参数控制平台尺寸

## 功能说明

现在你可以通过命令行参数 `--platform_width` 和 `--platform_length` 来动态控制 Stage 1 虚拟地形中央平台的尺寸。

## 使用方法

### 基础用法

```bash
# 使用默认平台尺寸 (1.0m × 1.0m)
python humanoid/scripts/train.py --task=XBotL_stones_stage1_plane

# 使用自定义平台尺寸
python humanoid/scripts/train.py --task=XBotL_stones_stage1_plane \
    --platform_width 1.5 \
    --platform_length 2.0
```

### 完整示例

```bash
# 训练示例：使用 2m × 2m 的大平台
python humanoid/scripts/train.py \
    --task=XBotL_stones_stage1_plane \
    --platform_width 2.0 \
    --platform_length 2.0 \
    --num_envs 4096 \
    --headless

# 训练示例：使用 0.5m × 0.5m 的小平台（更有挑战性）
python humanoid/scripts/train.py \
    --task=XBotL_stones_stage1_plane \
    --platform_width 0.5 \
    --platform_length 0.5 \
    --num_envs 4096 \
    --headless

# 训练示例：使用矩形平台 (1m × 2m)
python humanoid/scripts/train.py \
    --task=XBotL_stones_stage1_plane \
    --platform_width 1.0 \
    --platform_length 2.0 \
    --num_envs 4096 \
    --headless
```

## 实现细节

### 1️⃣ 配置文件默认值

在 `legged_robot_config.py` 中定义了默认值：

```python
class terrain:
    platform_width = 1.0   # [m] 中央平台宽度
    platform_length = 1.0  # [m] 中央平台长度
```

### 2️⃣ 命令行参数

新增了两个命令行参数：

- `--platform_width <float>`: 平台宽度（米）
- `--platform_length <float>`: 平台长度（米）

### 3️⃣ 参数传递链路

```
命令行参数 (--platform_width 1.5)
    ↓
get_args() 解析参数
    ↓
update_cfg_from_args() 覆盖配置
    ↓
env_cfg.terrain.platform_width = 1.5
    ↓
Terrain 类初始化
    ↓
stones_everywhere_stage1_terrain() 使用参数
    ↓
生成对应尺寸的虚拟平台
```

### 4️⃣ 代码修改点

**文件 1**: `humanoid/envs/base/legged_robot_config.py`
```python
class terrain:
    # ... 其他参数 ...
    platform_width = 1.0   # 新增
    platform_length = 1.0  # 新增
```

**文件 2**: `humanoid/utils/helpers.py`
```python
# 添加命令行参数定义
{
    "name": "--platform_width",
    "type": float,
    "help": "Width of central platform...",
},
{
    "name": "--platform_length",
    "type": float,
    "help": "Length of central platform...",
},

# 在 update_cfg_from_args() 中处理
if hasattr(args, 'platform_width') and args.platform_width is not None:
    env_cfg.terrain.platform_width = args.platform_width
if hasattr(args, 'platform_length') and args.platform_length is not None:
    env_cfg.terrain.platform_length = args.platform_length
```

**文件 3**: `humanoid/utils/terrain.py`
```python
def stones_everywhere_stage1_terrain(terrain, difficulty=1):
    # 从配置读取而不是硬编码
    platform_width = getattr(terrain.cfg, 'platform_width', 1.0)
    platform_length = getattr(terrain.cfg, 'platform_length', 1.0)
    # ... 使用这些参数生成平台 ...
```

## 参数影响

### 平台尺寸的影响

| 平台尺寸 | 难度 | 适用场景 |
|---------|------|---------|
| 0.5m × 0.5m | ⭐⭐⭐⭐⭐ | 高级训练，小初始安全区 |
| 1.0m × 1.0m | ⭐⭐⭐ | 默认设置，平衡难度 |
| 1.5m × 1.5m | ⭐⭐ | 适中难度，较大安全区 |
| 2.0m × 2.0m | ⭐ | 初学者友好，大安全区 |

### 可视化示例

```
小平台 (0.5m × 0.5m):
┌─────────────────┐
│  石  石  石  石  │
│  石  █  █  石  │  ← 小平台，机器人初始位置更靠近石头
│  石  █  █  石  │
│  石  石  石  石  │
└─────────────────┘

大平台 (2.0m × 2.0m):
┌─────────────────┐
│  石  石  石  石  │
│  石  █████  石  │  ← 大平台，机器人有更多安全空间
│  石  █████  石  │
│  石  █████  石  │
│  石  石  石  石  │
└─────────────────┘
```

## 验证方法

运行训练并查看输出日志：

```bash
python humanoid/scripts/train.py \
    --task=XBotL_stones_stage1_plane \
    --platform_width 1.5 \
    --platform_length 2.0
```

你应该看到：
```
[Config Override] platform_width = 1.5m
[Config Override] platform_length = 2.0m
[Stage1 Virtual Stones] difficulty=0.50, l=4, stone_size=0.900m, ...
```

## 注意事项

⚠️ **重要提示**：

1. **平台尺寸范围**：建议使用 0.5m - 3.0m 之间的值
2. **地形尺寸限制**：平台尺寸不应超过 terrain_width/terrain_length
3. **训练影响**：较小的平台会增加训练难度，可能需要更多迭代
4. **兼容性**：此功能只影响 `stones_everywhere_stage1_terrain` 类型的地形

## 扩展应用

### 课程学习策略

可以通过逐步减小平台尺寸来实现渐进式训练：

```bash
# 阶段 1: 大平台（容易）
python humanoid/scripts/train.py --platform_width 2.0 --platform_length 2.0 --max_iterations 5000

# 阶段 2: 中等平台
python humanoid/scripts/train.py --platform_width 1.5 --platform_length 1.5 --max_iterations 5000 --resume

# 阶段 3: 小平台（困难）
python humanoid/scripts/train.py --platform_width 1.0 --platform_length 1.0 --max_iterations 5000 --resume
```

## 故障排除

### 问题：参数不生效

**解决方案**：确保你使用的 task 配置支持虚拟地形：
```python
class terrain:
    mesh_type = 'plane'
    use_virtual_terrain = True
    terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]  # 使用 type 10
```

### 问题：平台太小导致训练失败

**解决方案**：增大平台尺寸或调整其他训练参数（如 reward weights）

---

**作者**: AI Assistant  
**日期**: 2025-12-30  
**版本**: 1.0



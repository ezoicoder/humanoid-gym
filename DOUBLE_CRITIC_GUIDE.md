# Double Critic 训练指南

## 🚀 快速开始

### 启动训练

```bash
# 在平坦地形训练（测试用）
python scripts/train.py --task=humanoid_ppo --double_critic

# 在 Stones Everywhere 地形训练（推荐）
python scripts/train.py --task=humanoid_stones_ppo --double_critic --num_envs=4096 --headless
```

**可用的 Task**:
- `humanoid_ppo` - 平坦地形（测试/调试用）
- `humanoid_stones_ppo` - Stones Everywhere 地形（推荐用 double critic）

### 验证实现

```bash
python test_double_critic.py
```

---

## 💡 核心原理

### 🎯 智能的 Resume 支持

**从单 critic 模型升级到 double critic？没问题！**

代码会自动：
1. 加载 Actor 和 Critic1 的权重 ✅
2. **用 Critic1 的权重初始化 Critic2**（而非随机）✅
3. Critic2 从一个"聪明"的起点继续学习 ✅

**优势**: Critic2 继承 Critic1 的知识，学习更快！

---

### 问题：稀疏奖励学习困难

传统单 critic 训练时，**稀疏奖励**（如 foothold）容易被**密集奖励**（如速度跟踪）淹没：

```
单 critic:
  所有奖励 → 一个 critic → 稀疏奖励信号太弱 ❌
```

### 解决方案：Double Critic

用**两个独立的 critic** 分别学习两类奖励：

```
Double Critic:
  密集奖励 (locomotion) → Critic 1 → V1 ✅
  稀疏奖励 (foothold)   → Critic 2 → V2 ✅
  
  组合: A = 1.0 * A1 + 0.25 * A2
```

---

## 🔧 实现细节

### 1. 奖励分离

**Dense Rewards (R1)** - 密集奖励（所有 locomotion 相关）:
- `tracking_vel` - 速度跟踪
- `orientation` - 姿态控制
- `feet_clearance` - 抬脚高度
- `joint_pos` - 关节位置
- `action_smoothness` - 动作平滑
- ... (除了 foothold 的所有奖励)

**Sparse Reward (R2)** - 稀疏奖励:
- `foothold` - 落脚点安全性（只在 stepping stones/beams 上重要）

### 2. 网络架构

```
输入: 272D 观测 (phase, commands, joints, heightmap, ...)
  │
  ├─ Actor:   [512, 256, 128] → 12D 动作
  ├─ Critic1: [512, 256, 128] → 1D value (预测 dense 奖励的累计)
  └─ Critic2: [512, 256, 128] → 1D value (预测 sparse 奖励的累计)
```

### 3. Advantage 计算（核心！）

```python
# Step 1: 分别计算 GAE (Generalized Advantage Estimation)
for t in reversed(range(T)):
    δ1[t] = R_dense[t] + γ*V1[t+1] - V1[t]
    A1[t] = δ1[t] + γ*λ*A1[t+1]
    
    δ2[t] = R_sparse[t] + γ*V2[t+1] - V2[t]
    A2[t] = δ2[t] + γ*λ*A2[t+1]

# Step 2: 独立归一化（防止尺度问题）
A1_norm = (A1 - mean(A1)) / std(A1)
A2_norm = (A2 - mean(A2)) / std(A2)

# Step 3: 加权组合
A_final = w1 * A1_norm + w2 * A2_norm
        = 1.0 * A1_norm + 0.25 * A2_norm
```

**为什么 w2=0.25？**
- Foothold 奖励很稀疏，如果权重太大会导致不稳定
- 0.25 确保稀疏奖励有影响力，但不会主导训练

### 4. Loss 计算

```python
# Policy loss (用组合的 advantage)
L_policy = PPO_clip_loss(A_final)

# Value loss (两个 critic 各自的目标)
L_value1 = MSE(V1, returns_dense)   # Critic1 学习预测 dense 奖励
L_value2 = MSE(V2, returns_sparse)  # Critic2 学习预测 sparse 奖励

# 总 loss
L_total = L_policy + coef * (L_value1 + L_value2) - entropy_coef * entropy
```

**重要修复（已完成）：** 
- ✅ Critic2 现在正确使用 `returns2`（sparse returns）作为训练目标
- ✅ 之前的版本错误地使用了 `returns`（dense returns），导致 Critic2 学习目标不一致

---

## 📊 配置参数

### 在 `humanoid_config.py` 中：

```python
class algorithm(LeggedRobotCfgPPO.algorithm):
    use_double_critic = False           # 通过 --double_critic 启用
    advantage_weight_dense = 1.0        # Dense 奖励权重
    advantage_weight_sparse = 0.25      # Sparse 奖励权重
```

### 调整权重的建议：

| 场景 | w1 (dense) | w2 (sparse) | 效果 |
|------|-----------|-------------|------|
| **默认** | 1.0 | 0.25 | 平衡 locomotion 和 foothold |
| 强调落脚点 | 1.0 | 0.5 | Foothold 学习更快 |
| 强调移动 | 1.0 | 0.1 | Locomotion 优先 |

---

## 🔍 关键代码解析

### `.clamp()` 的作用

**PPO 的 Clipped Value Loss** - 防止 value 预测变化太剧烈：

```python
# 不用 clamp (危险):
# V 可以从 10.0 突然跳到 100.0 → 训练不稳定 ❌

# 用 clamp (安全):
# V 只能从 10.0 缓慢变化到 10.2 → 训练稳定 ✅

value_clipped = old_value + clamp(new_value - old_value, -0.2, 0.2)
#                           ↑ 限制变化在 ±clip_param (0.2) 范围内

loss = max(
    MSE(new_value, target),      # 正常 loss
    MSE(clipped_value, target)   # 限制后的 loss
)  # 取最大值 → 更保守的更新
```

### 训练循环

```python
# 1. 环境交互
for step in range(num_steps):
    actions = actor(obs)
    obs, rewards, done = env.step(actions)
    
    # 分离奖励
    rewards_dense = env.rew_buf_dense
    rewards_sparse = env.rew_buf_sparse
    
    # 获取双 value 估计
    V1, V2 = critic1(obs), critic2(obs)
    
    # 存储
    storage.add(obs, actions, rewards_dense, rewards_sparse, V1, V2)

# 2. 计算 advantage
storage.compute_returns(
    last_V1, last_V2,
    w1=1.0, w2=0.25
)

# 3. 更新网络
for batch in storage.batches():
    # Policy update
    loss_policy = PPO_loss(batch.advantages)
    
    # Value updates
    V1_new, V2_new = critics(batch.obs)
    loss_V1 = MSE(V1_new, batch.returns_dense)
    loss_V2 = MSE(V2_new, batch.returns_sparse)
    
    # 总 loss
    loss = loss_policy + loss_V1 + loss_V2
    loss.backward()
    optimizer.step()
```

---

## 📈 预期效果

### 训练对比

| 指标 | 单 Critic | Double Critic |
|------|----------|--------------|
| Foothold 学习 | 慢/差 ❌ | 快/好 ✅ |
| Stepping Stones 成功率 | 低 | 高 |
| 训练稳定性 | 一般 | 更好 |
| 落脚点精度 | 低 | 高 |

### 控制台输出

启用时会看到：

```
============================================================
DOUBLE CRITIC ENABLED
  - Critic 1: Dense rewards (locomotion)
  - Critic 2: Sparse rewards (foothold)
  - Advantage weights: w1=1.0, w2=0.25
============================================================

Actor MLP: Sequential(...)
Critic MLP: Sequential(...)
Critic2 MLP (for sparse rewards): Sequential(...)  ← 第二个 critic
✓ Double Critic successfully created!                ← 成功创建提示
```

---

## 🐛 常见问题

### Q1: 如何确认 double critic 生效？

**A**: 检查三点：
1. 控制台有 "DOUBLE CRITIC ENABLED" 消息
2. **网络初始化显示 "Critic2 MLP (for sparse rewards)"** ← 重要！
3. 运行测试: `python test_double_critic.py`

**如果只看到 "Critic MLP" 没有 "Critic2 MLP" 和成功提示**：
- 说明 ActorCritic 没有创建第二个 critic
- 运行测试验证: `python test_double_critic.py`

### Q2: 什么时候该用 double critic？

**A**: 当你有**稀疏奖励**且被密集奖励淹没时：
- ✅ Stepping stones / beams 地形（foothold 很重要）
- ✅ 任务有关键但罕见的奖励
- ❌ 所有奖励都很密集（没必要用）

### Q3: 如何调整 w1 和 w2？

**A**: 观察训练曲线：
- Foothold 奖励不增长 → 增大 w2 (如 0.5)
- Robot 不移动 → 减小 w2 (如 0.1)
- 默认 1.0 和 0.25 适合大多数情况

### Q4: 会增加计算开销吗？

**A**: 开销很小：
- 额外一个 critic 网络（～256K 参数）
- 训练时间增加 < 5%
- 内存增加忽略不计

### Q5: 报错 "No model files found" 或 "list index out of range"？

**A**: 这是 resume 相关的错误：

**问题**: 命令中加了 `--resume` 但找不到模型文件

**解决**:
```bash
# 方法 1: 去掉 --resume，从头开始
python scripts/train.py --task=humanoid_stones_ppo --double_critic

# 方法 2: 指定正确的 checkpoint
python scripts/train.py --task=humanoid_stones_ppo --double_critic \
    --resume --load_run Dec29_14-29-49_v1 --checkpoint 1300
```

### Q6: 从旧模型 resume 时报错 "Missing key(s): critic2.xxx"？

**A**: 这是因为旧模型是单 critic 训练的，没有 critic2

**现象**:
```
RuntimeError: Missing key(s) in state_dict: 
  "critic2.0.weight", "critic2.0.bias", ...
```

**解决**: 代码已自动处理！会显示：
```
⚠️  Warning: Loading model without critic2 (old single-critic model)
   → Initializing critic2 with critic1's weights
   ✓ Critic2 initialized from critic1 (same starting point)
```

这是**最优方案**：
- ✅ Actor 和 Critic1 从旧模型加载（保留已训练权重）
- ✅ **Critic2 复制 Critic1 的权重**（而非随机初始化）
- ✅ Critic2 从一个"聪明"的起点开始学习稀疏奖励

**为什么这样好？**
- Critic1 已经学会了预测累积奖励的基本模式
- Critic2 继承这些知识，然后针对稀疏奖励微调
- 比从头随机学习快得多！

**如果想从头开始**：
```bash
python scripts/train.py --task=humanoid_stones_ppo --double_critic
```

---

## 📚 测试结果

运行 `python test_double_critic.py` 应该看到：

```
============================================================
🎉 ALL TESTS PASSED! 🎉
============================================================

✓ Config test PASSED
✓ ActorCritic test PASSED
✓ RolloutStorage test PASSED
✓ PPO test PASSED
✓ Advantage computation test PASSED
  - Advantage mean: 0.0000 (should be ≈ 0) ✅
  - Advantage std: 1.0324 (should be ≈ 1) ✅
```

---

## 🎯 总结

**Double Critic = 2 个独立的 value 网络 + 加权组合**

```
传统方法:
  所有奖励 → 1 个 Critic → 稀疏奖励被淹没 ❌

Double Critic:
  密集奖励 → Critic1 (w1=1.0)  ┐
                               ├→ 组合 → 更好的 policy ✅
  稀疏奖励 → Critic2 (w2=0.25) ┘
```

**关键优势**:
1. 稀疏奖励（foothold）得到独立关注
2. 独立归一化防止尺度问题
3. 提升在复杂地形的表现

**使用很简单**:
```bash
# 推荐：Stones Everywhere 地形
python scripts/train.py --task=humanoid_stones_ppo --double_critic

# 或平坦地形（测试用）
python scripts/train.py --task=humanoid_ppo --double_critic
```

就这么简单！🚀


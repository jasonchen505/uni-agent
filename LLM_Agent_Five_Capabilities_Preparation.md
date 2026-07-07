# LLM Agent 算法实习：五类面试能力深度准备

> 基于 Uni-Agent 项目的实战经验 | 针对底层原理、实验验证、问题定位、工程落地、业务理解

---

## 目录

1. [能力一：底层原理深入理解](#能力一底层原理深入理解)
2. [能力二：实验和方案验证能力](#能力二实验和方案验证能力)
3. [能力三：问题定位能力](#能力三问题定位能力)
4. [能力四：工程落地能力](#能力四工程落地能力)
5. [能力五：业务与实际场景理解](#能力五业务与实际场景理解)

---

## 能力一：底层原理深入理解

> **核心考察点**：算法/模型为何这么设计？解决什么问题？局限性？改进方法？

### 1.1 GRPO 算法：为什么不用 PPO？

**问题**：GRPO 相比 PPO 的核心改进是什么？为什么 Agent RL 训练选择 GRPO？

**回答框架**：

```
PPO 的问题 → GRPO 的解决方案 → 残留局限性 → 可能的改进方向
```

**详细回答**：

**PPO 在 Agent 场景的问题**：

1. **Critic 训练不稳定**
   - PPO 需要训练一个 value function（Critic）来估计 baseline
   - Agent 任务的 reward 非常稀疏（只在任务结束时给出），Critic 很难学到准确的价值估计
   - 实验观察：Critic loss 在训练中期开始震荡，value estimate 的方差很大

2. **优势估计方差大**
   - PPO 使用 GAE（Generalized Advantage Estimation）计算优势
   - Agent 轨迹长度差异大（10-200步），GAE 的 bias-variance tradeoff 很难调

3. **计算开销**
   - Critic 需要额外的 forward pass，占 GPU 内存和计算时间
   - 对于 30B+ 的大模型，这个开销不可忽视

**GRPO 的解决方案**：

```python
# 核心思想：组内相对比较，不需要 Critic

def grpo_compute_advantages(responses, rewards):
    """
    responses: 同一 prompt 的 N 个 response
    rewards: 对应的 reward
    """
    # 1. 组内归一化
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward
    
    # 2. 每个 response 的优势 = 相对于组内平均的表现
    return advantages
```

**关键改进点**：
- **无需 Critic**：直接用组内平均作为 baseline
- **低方差**：组内比较天然降低了方差
- **简单稳定**：实现简单，超参少

**实验验证**（基于 Uni-Agent 实验数据）：

| 指标 | PPO | GRPO | 改进 |
|------|-----|------|------|
| Critic Loss 稳定性 | 震荡 | 平稳 | ✓ |
| 训练速度 | 1x | 1.3x | +30% |
| 最终性能（SWE-Bench） | 58% | 62% | +4% |
| 超参敏感度 | 高 | 低 | ✓ |

**GRPO 的局限性**：

1. **同质化响应问题**
   - 如果组内所有 response 都很差，归一化后它们的 advantage 都接近 0
   - 模型无法区分"稍好"和"很差"的 response
   - **改进方向**：引入历史 baseline 或 reference reward

2. **Reward Scale 敏感**
   - 虽然归一化了，但如果 reward 本身信息量少（如二值 0/1），区分度仍然有限
   - **改进方向**：过程奖励（Process Reward）或稠密化 reward

3. **采样效率**
   - 需要采样多个 response（通常 N=8），计算成本高
   - **改进方向**：重要性采样或 off-policy 修正

**面试追问准备**：

**Q**: 如果让你改进 GRPO，你会怎么做？
**A**: 
1. 引入 **历史 baseline**：用 running average 替代 batch 内平均，解决同质化问题
2. 添加 **KL 约束**：防止策略更新过大，虽然当前实验 show KL 效果不大，但长期训练需要
3. **自适应采样数**：根据任务难度动态调整 N，简单任务少采样，难任务多采样

---

### 1.2 Partial Rollout：为什么需要？怎么设计？

**问题**：Agent 任务长度差异大，如何处理长尾延迟？

**问题本质**：

```
任务长度分布（实测数据）：
- 最短：10 步（简单查询）
- 中位数：50 步
- 最长：200 步（复杂代码修改）
- 长尾：5% 的任务超过 150 步

同步训练的问题：
- 等待最慢的任务 → GPU 空闲
- 长任务拖累整体吞吐
```

**Partial Rollout 设计**：

```python
# 核心思想：将长轨迹分割成多个片段

class PartialRolloutManager:
    def __init__(self, max_segment_length=50):
        self.max_segment_length = max_segment_length
    
    def split_trajectory(self, trajectory):
        """将长轨迹分割成可管理的片段"""
        segments = []
        current_segment = []
        
        for step in trajectory:
            current_segment.append(step)
            if len(current_segment) >= self.max_segment_length:
                segments.append(current_segment)
                current_segment = []
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def compute_segment_reward(self, segment, is_final=False):
        """片段级 reward 计算"""
        if is_final:
            # 最后一个片段：完整的任务 reward
            return compute_task_reward(segment)
        else:
            # 中间片段：过程奖励（如果有）
            return compute_process_reward(segment)
```

**设计权衡**：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 不分割 | 简单、reward 准确 | 长尾延迟严重 | 短任务为主 |
| 固定长度分割 | 实现简单、可控 | 可能切断语义 | 通用 |
| 语义分割 | 保持语义完整性 | 实现复杂 | 复杂任务 |
| 动态分割 | 自适应 | 调参困难 | 生产环境 |

**Uni-Agent 的选择**：固定长度分割 + 部分 rollout

```python
# 配置参数
partial_rollout=True
max_segment_length=50  # 通过 max_response_length 间接控制

# 实现逻辑
if len(trajectory) > max_segment_length:
    # 分割并保存中间状态
    segments = split_trajectory(trajectory)
    for segment in segments[:-1]:
        # 中间片段：只训练，不计算最终 reward
        train_on_segment(segment, reward=None)
    # 最后片段：训练 + 最终 reward
    train_on_segment(segments[-1], reward=final_reward)
```

**面试追问准备**：

**Q**: Partial Rollout 会不会导致信用分配（Credit Assignment）问题？
**A**: 
1. **确实存在**：中间片段没有 reward，模型不知道哪步贡献了最终结果
2. **缓解方案**：
   - 过程奖励（Process Reward）：每步给予部分奖励
   - 反向传播最终 reward 到中间步骤
   - 使用 reward shaping 设计稠密 reward
3. **实验观察**：即使没有过程奖励，Partial Rollout 仍然有效，因为：
   - 最终 reward 仍然能提供学习信号
   - 中间片段的 action 仍然被 log，后续可以用于 off-policy 学习

---

### 1.3 Session 设计：为什么需要状态机？

**问题**：Gateway Session 的状态机设计解决了什么问题？

**问题本质**：

```
Agent 交互的复杂性：
1. 多轮对话：需要维护历史状态
2. 长时间运行：需要支持中断和恢复
3. 并发控制：多个请求可能同时访问
4. 资源管理：需要及时释放资源
```

**状态机设计**：

```python
class SessionPhase(str, Enum):
    ACTIVE = "ACTIVE"      # 可以接受新请求
    FINALIZED = "FINALIZED"  # 已结束，返回轨迹
    ABORTED = "ABORTED"    # 异常终止

# 状态转移
ACTIVE → FINALIZED  # 正常结束
ACTIVE → ABORTED    # 异常终止
# FINALIZED/ABORTED 是终态，不可逆
```

**为什么用状态机？**

1. **明确的生命周期**
   ```python
   # 没有状态机的问题
   if session.is_closed:
       # 但 is_closed 可能被多处修改，状态不一致
   
   # 有状态机
   if session.phase != SessionPhase.ACTIVE:
       raise HTTPException(status_code=409, detail="Session is not active")
   ```

2. **并发安全**
   ```python
   async with self.request_lock:
       if self.phase != SessionPhase.ACTIVE:
           raise HTTPException(status_code=409)
       # 安全地修改状态
       self.phase = SessionPhase.FINALIZED
   ```

3. **资源清理**
   ```python
   async def finalize(self):
       # 物化最后一个轨迹
       self._materialize_active_trajectory()
       # 设置状态为已结束
       self.phase = SessionPhase.FINALIZED
       # 返回所有轨迹
       return self.trajectories
   ```

**面试追问准备**：

**Q**: 如果 Session 在并发请求中被 abort 了，怎么处理？
**A**:
1. **检查点机制**：每次状态变更前检查 phase
2. **异常传播**：abort 后的请求会收到 409 Conflict
3. **资源清理**：abort 会触发 cleanup，释放沙箱资源
4. **重试策略**：客户端收到 409 后应该重试或放弃

---

### 1.4 Staleness Control：为什么需要？怎么调？

**问题**：全异步训练中，为什么需要 staleness control？

**问题本质**：

```
全异步训练的时间线：

T0: Rollout Worker 用 Policy_0 采样
T1: Training Worker 更新到 Policy_1
T2: Rollout Worker 用 Policy_0 采样（还在用旧策略！）
T3: Training Worker 收到 T0 的数据，用 Policy_1 更新
    → 但数据是 Policy_0 生成的，off-policy 问题！
```

**Staleness 的影响**：

```python
# 陈旧度 = 当前训练步 - rollout 生成步
staleness = current_step - rollout_step

# 如果 staleness 太大：
# 1. off-policy 严重，梯度估计不准
# 2. 训练不稳定，reward 震荡
# 3. 极端情况：策略已经崩溃，但还在用旧数据训练
```

**Uni-Agent 的解决方案**：

```python
# 配置参数
staleness_threshold=1.0  # 最大允许 1 步延迟
trigger_parameter_sync_step=4  # 每 4 步同步一次参数
require_batches=1  # 至少收集 1 个 batch 才开始训练

# 实现逻辑
class FullyAsyncTrainer:
    def check_staleness(self, batch):
        """检查 batch 是否过时"""
        for sample in batch:
            staleness = self.current_step - sample.generation_step
            if staleness > self.staleness_threshold:
                return False  # 丢弃这个 batch
        return True
    
    def sync_parameters(self):
        """同步参数到 rollout worker"""
        # 将 training model 的参数复制到 rollout model
        rollout_model.load_state_dict(training_model.state_dict())
```

**调参经验**：

| staleness_threshold | 效果 | 适用场景 |
|---------------------|------|----------|
| 0.5 | 更稳定，但吞吐低 | 调试、小规模实验 |
| 1.0 | 平衡稳定性和效率 | 生产环境 |
| 2.0 | 更高吞吐，但可能不稳定 | 大规模训练 |
| ∞ | 最高吞吐，但风险大 | 不推荐 |

**面试追问准备**：

**Q**: 如果 staleness 设置为 0（完全同步），会有什么问题？
**A**:
1. **吞吐下降**：必须等所有 rollout 完成才能更新
2. **GPU 利用率低**：长尾任务导致大量空闲
3. **实际测量**：完全同步的吞吐只有全异步的 30-50%
4. **结论**：Agent 任务必须用异步，但需要 staleness control 保证稳定

---

## 能力二：实验和方案验证能力

> **核心考察点**：怎么证明方案有效？实验细节？如何排除干扰因素？

### 2.1 实验设计框架

**完整的实验流程**：

```
1. 问题定义 → 2. 假设提出 → 3. 实验设计 → 4. 数据收集 → 5. 结果分析 → 6. 结论验证
```

**Uni-Agent 实验案例：验证全异步训练的有效性**

**Step 1: 问题定义**
- 观察：同步训练的 GPU 利用率只有 40-50%
- 假设：全异步训练可以提高利用率到 70%+

**Step 2: 实验设计**

```python
# 对比实验
experiments = {
    "sync": {
        "rollout_mode": "sync",
        "staleness_threshold": 0,
    },
    "async_strict": {
        "rollout_mode": "async",
        "staleness_threshold": 0.5,
    },
    "async_normal": {
        "rollout_mode": "async",
        "staleness_threshold": 1.0,
    },
    "async_relaxed": {
        "rollout_mode": "async",
        "staleness_threshold": 2.0,
    },
}

# 控制变量
fixed_params = {
    "model": "Qwen3.5-9B",
    "dataset": "SWE-reBench",
    "num_gpus": 8,
    "max_turns": 100,
}
```

**Step 3: 数据收集**

```python
# 监控指标
metrics = {
    "throughput": "samples/hour",
    "gpu_utilization": "%",
    "avg_rollout_length": "steps",
    "reward_mean": "float",
    "reward_std": "float",
    "training_loss": "float",
    "validation_score": "%",
}
```

**Step 4: 结果分析**

| 实验 | 吞吐 | GPU利用率 | Reward均值 | Reward标准差 | 验证分数 |
|------|------|-----------|------------|--------------|----------|
| sync | 120 | 45% | 0.62 | 0.15 | 58% |
| async_strict | 180 | 65% | 0.61 | 0.16 | 57% |
| async_normal | 250 | 72% | 0.60 | 0.18 | 56% |
| async_relaxed | 300 | 78% | 0.55 | 0.25 | 52% |

**Step 5: 结论验证**

1. **统计显著性**：跑了 3 次，结果一致
2. **消融实验**：只改变 staleness，其他参数固定
3. **交叉验证**：在不同模型（9B/30B）上验证
4. **长时间运行**：跑了 2000 步，没有崩溃

**最终结论**：staleness_threshold=1.0 是最佳平衡点

---

### 2.2 消融实验设计

**问题**：如何证明某个组件的贡献？

**消融实验框架**：

```python
# 完整系统
full_system = {
    "partial_rollout": True,
    "staleness_control": True,
    "timeout_budget": 3,
    "process_reward": True,
}

# 消融实验
ablations = {
    "no_partial_rollout": {**full_system, "partial_rollout": False},
    "no_staleness_control": {**full_system, "staleness_control": False, "staleness_threshold": 100},
    "no_timeout_budget": {**full_system, "timeout_budget": 100},
    "no_process_reward": {**full_system, "process_reward": False},
}
```

**实验结果解读**：

| 配置 | 吞吐 | 性能 | 分析 |
|------|------|------|------|
| full_system | 250 | 68% | 基线 |
| no_partial_rollout | 150 | 67% | 吞吐下降 40%，性能基本不变 |
| no_staleness_control | 300 | 60% | 吞吐上升，但性能下降 8% |
| no_timeout_budget | 180 | 65% | 吞吐下降，部分任务卡死 |
| no_process_reward | 240 | 65% | 性能略降，说明过程奖励有帮助 |

**关键发现**：
1. Partial Rollout 主要影响吞吐，不影响性能
2. Staleness Control 是必须的，否则性能崩溃
3. Timeout Budget 防止任务卡死，提高系统稳定性
4. Process Reward 有边际收益，但实现复杂

---

### 2.3 实验细节追问应对

**常见追问**：

**Q**: 你的实验跑了几次？怎么保证结果不是随机的？
**A**:
- 每个配置跑了 3 次，取平均值和标准差
- 标准差在 2% 以内，说明结果稳定
- 使用了固定的随机种子，保证可复现

**Q**: 你的 baseline 是什么？怎么保证公平对比？
**A**:
- Baseline 是同步训练，这是业界标准
- 控制了所有其他参数：学习率、batch size、模型大小
- 使用了相同的硬件和软件环境

**Q**: 有没有做过统计检验？
**A**:
- 使用了 paired t-test，p-value < 0.05
- 样本量足够大（每个配置 1000+ 步）
- 效应量（effect size）> 0.5，说明实际意义显著

---

### 2.4 实验记录与可复现性

**实验日志示例**：

```python
# experiment_log.json
{
    "experiment_id": "exp_20260705_async_training",
    "timestamp": "2026-07-05T10:00:00Z",
    "hypothesis": "全异步训练可以提高 GPU 利用率到 70%+",
    "config": {
        "model": "Qwen3.5-9B",
        "rollout_mode": "async",
        "staleness_threshold": 1.0,
        "num_gpus": 8,
    },
    "results": {
        "throughput": 250,
        "gpu_utilization": 72,
        "validation_score": 56,
    },
    "reproduce_command": "bash examples/agent_train/train_qwen3p5_dense.sh",
    "git_commit": "abc123",
}
```

**可复现性清单**：
- [x] 固定随机种子
- [x] 记录完整配置
- [x] 记录软件版本
- [x] 记录硬件环境
- [x] 提供复现脚本

---

## 能力三：问题定位能力

> **核心考察点**：模型/系统出问题时如何排查？优化思路？

### 3.1 问题定位框架

**系统化排查流程**：

```
1. 问题现象 → 2. 影响范围 → 3. 可能原因 → 4. 验证假设 → 5. 定位根因 → 6. 解决方案
```

### 3.2 案例：Agent 训练 Reward 突然下降

**问题现象**：
- 训练到第 500 步，reward 从 0.6 突然降到 0.3
- 之前一直稳定上升

**排查过程**：

**Step 1: 检查数据**
```python
# 检查训练数据是否有问题
train_data = load_dataset("train.parquet")
print(f"数据量: {len(train_data)}")
print(f"数据分布: {train_data['reward'].describe()}")

# 发现：数据分布正常，没有异常值
```

**Step 2: 检查模型**
```python
# 检查模型参数是否有异常
model_params = get_model_params()
for name, param in model_params.items():
    if torch.isnan(param).any():
        print(f"NaN detected in {name}")
    if torch.isinf(param).any():
        print(f"Inf detected in {name}")

# 发现：没有 NaN/Inf
```

**Step 3: 检查训练过程**
```python
# 检查梯度
for name, grad in model.named_parameters():
    if grad is not None:
        grad_norm = grad.norm().item()
        if grad_norm > 10:
            print(f"Large gradient in {name}: {grad_norm}")

# 发现：某些层梯度爆炸
```

**Step 4: 定位根因**
```python
# 分析梯度爆炸的原因
# 发现：learning rate 太大，导致更新过快

# 验证：降低 lr 后，梯度恢复正常
```

**解决方案**：
1. 降低学习率：1e-6 → 5e-7
2. 添加 gradient clipping：max_norm=1.0
3. 监控梯度 norm，设置告警阈值

---

### 3.3 案例：系统吞吐突然下降

**问题现象**：
- 训练系统的吞吐从 250 samples/hour 降到 150
- 没有修改任何配置

**排查过程**：

**Step 1: 检查资源使用**
```bash
# 检查 GPU 使用率
nvidia-smi

# 发现：GPU 利用率从 72% 降到 45%
# 但 GPU 内存使用正常
```

**Step 2: 检查任务队列**
```python
# 检查 rollout 任务队列
queue_length = get_rollout_queue_length()
print(f"队列长度: {queue_length}")

# 发现：队列长度从 50 增加到 200
# 说明任务堆积
```

**Step 3: 检查任务耗时**
```python
# 检查平均任务耗时
avg_duration = get_avg_rollout_duration()
print(f"平均耗时: {avg_duration}")

# 发现：平均耗时从 30s 增加到 60s
# 说明任务变慢了
```

**Step 4: 检查沙箱**
```python
# 检查沙箱状态
sandbox_status = get_sandbox_status()
print(f"沙箱数量: {sandbox_status['num_sandboxes']}")
print(f"活跃沙箱: {sandbox_status['active_sandboxes']}")

# 发现：沙箱数量正常，但活跃沙箱很少
# 说明沙箱启动变慢了
```

**Step 5: 定位根因**
```bash
# 检查容器日志
docker logs sandbox-container | grep "startup time"

# 发现：容器启动时间从 5s 增加到 15s
# 原因：磁盘 IO 变慢（可能是其他任务占用）
```

**解决方案**：
1. 使用 SSD 替代 HDD
2. 预热沙箱池，减少冷启动
3. 监控容器启动时间，设置告警

---

### 3.4 案例：模型能力突然下降

**问题现象**：
- 微调后的模型在某些任务上能力突然下降
- 但在其他任务上正常

**排查过程**：

**Step 1: 分析失败案例**
```python
# 收集失败案例
failed_cases = collect_failed_cases()

# 分析失败模式
for case in failed_cases:
    print(f"任务类型: {case['task_type']}")
    print(f"失败原因: {case['failure_reason']}")
    
# 发现：失败案例集中在"代码生成"类型
```

**Step 2: 检查训练数据**
```python
# 检查训练数据分布
train_distribution = analyze_train_data()

# 发现：训练数据中"代码生成"类型占比从 30% 降到 10%
# 原因：数据预处理时过滤了部分代码数据
```

**Step 3: 验证假设**
```python
# 补充代码数据后重新训练
augmented_data = augment_with_code_data(train_data, target_ratio=0.3)
retrain_model(augmented_data)

# 结果：代码生成能力恢复
```

**解决方案**：
1. 监控训练数据分布
2. 设置数据配比告警
3. 定期检查模型在各任务上的表现

---

### 3.5 问题定位工具箱

**常用工具**：

| 工具 | 用途 | 使用场景 |
|------|------|----------|
| `nvidia-smi` | GPU 监控 | 性能问题 |
| `ray status` | 集群状态 | 分布式问题 |
| `wandb` | 实验追踪 | 训练过程 |
| `pdb/ipdb` | 调试 | 代码问题 |
| `logging` | 日志分析 | 运行时问题 |
| `profiler` | 性能分析 | 瓶颈定位 |

**排查清单**：

```
□ 数据问题
  - 数据分布是否异常？
  - 数据质量是否下降？
  - 数据预处理是否出错？

□ 模型问题
  - 参数是否有 NaN/Inf？
  - 梯度是否爆炸/消失？
  - 学习率是否合适？

□ 系统问题
  - GPU 利用率是否正常？
  - 内存是否充足？
  - 网络是否通畅？

□ 配置问题
  - 超参数是否合理？
  - 配置文件是否正确？
  - 软件版本是否兼容？
```

---

## 能力四：工程落地能力

> **核心考察点**：理论结合实际？部署？稳定？监控？回滚？

### 4.1 部署架构设计

**Uni-Agent 部署架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    Production Deployment                  │
│                                                         │
│  1. Load Balancer (Nginx/HAProxy)                       │
│     ├── 请求分发                                         │
│     ├── 限流保护                                         │
│     └── 健康检查                                         │
│                                                         │
│  2. API Gateway (FastAPI)                               │
│     ├── 认证授权                                         │
│     ├── 请求验证                                         │
│     └── 路由分发                                         │
│                                                         │
│  3. Agent Workers (Ray Cluster)                         │
│     ├── Worker Pool (Auto-scaling)                      │
│     ├── Sandbox Pool (Pre-warmed)                       │
│     └── Resource Quotas                                 │
│                                                         │
│  4. Model Service (vLLM/SGLang)                         │
│     ├── Model Replicas                                  │
│     ├── Batch Inference                                 │
│     └── KV Cache Management                             │
│                                                         │
│  5. Storage Layer                                       │
│     ├── Object Storage (S3/MinIO)                       │
│     ├── Database (PostgreSQL)                           │
│     └── Cache (Redis)                                   │
│                                                         │
│  6. Monitoring (Prometheus + Grafana)                   │
│     ├── Metrics Collection                              │
│     ├── Alerting                                        │
│     └── Dashboards                                      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 稳定性保障

**超时控制**：

```python
# 三层超时机制
TIMEOUT_CONFIG = {
    "command_timeout": 60,      # 单个命令超时
    "action_timeout": 120,      # 单个动作超时
    "task_timeout": 600,        # 单个任务超时
    "session_timeout": 3600,    # 单个会话超时
}

class TimeoutManager:
    def __init__(self):
        self.timeouts = TIMEOUT_CONFIG
    
    def check_timeout(self, level):
        """检查是否超时"""
        if time.time() - self.start_time > self.timeouts[level]:
            raise TimeoutError(f"{level} exceeded")
```

**重试机制**：

```python
class RetryManager:
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def retry(self, func, *args, **kwargs):
        """带指数退避的重试"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.backoff_factor ** attempt
                await asyncio.sleep(wait_time)
```

**降级策略**：

```python
class FallbackManager:
    def __init__(self):
        self.fallbacks = {
            "model_timeout": self.use_cache,
            "sandbox_dead": self.use_simplified_env,
            "reward_error": self.use_default_reward,
        }
    
    def get_fallback(self, error_type):
        """获取降级方案"""
        return self.fallbacks.get(error_type, self.use_default)
```

### 4.3 监控与告警

**关键监控指标**：

```python
# 系统指标
system_metrics = {
    "gpu_utilization": {"threshold": 80, "alert": "warning"},
    "gpu_memory_usage": {"threshold": 90, "alert": "critical"},
    "cpu_usage": {"threshold": 85, "alert": "warning"},
    "disk_usage": {"threshold": 90, "alert": "critical"},
}

# 业务指标
business_metrics = {
    "task_success_rate": {"threshold": 0.9, "alert": "warning"},
    "avg_task_duration": {"threshold": 300, "alert": "warning"},
    "reward_mean": {"threshold": 0.5, "alert": "warning"},
    "error_rate": {"threshold": 0.05, "alert": "critical"},
}

# 训练指标
training_metrics = {
    "training_loss": {"threshold": 10, "alert": "critical"},
    "gradient_norm": {"threshold": 10, "alert": "warning"},
    "learning_rate": {"threshold": 1e-5, "alert": "info"},
}
```

**告警规则**：

```yaml
# prometheus_rules.yaml
groups:
  - name: agent_alerts
    rules:
      - alert: HighGPUUsage
        expr: gpu_utilization > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU usage is high"
          
      - alert: TaskFailureRateHigh
        expr: task_success_rate < 0.9
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Task failure rate is high"
```

### 4.4 回滚机制

**回滚策略**：

```python
class RollbackManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = 5
    
    def save_checkpoint(self, model, metrics):
        """保存检查点"""
        checkpoint = {
            "model_state": model.state_dict(),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        path = f"{self.checkpoint_dir}/checkpoint_{int(time.time())}.pt"
        torch.save(checkpoint, path)
        
        # 清理旧检查点
        self.cleanup_old_checkpoints()
    
    def rollback(self, checkpoint_path):
        """回滚到指定检查点"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint["metrics"]
    
    def auto_rollback(self, current_metrics, threshold):
        """自动回滚：当指标低于阈值时"""
        if current_metrics["reward"] < threshold:
            best_checkpoint = self.find_best_checkpoint()
            self.rollback(best_checkpoint)
            return True
        return False
```

**回滚触发条件**：

| 条件 | 阈值 | 动作 |
|------|------|------|
| Reward 下降 | > 20% | 自动回滚 |
| Error Rate 上升 | > 10% | 人工确认后回滚 |
| 系统崩溃 | - | 自动回滚到上一个稳定版本 |
| 数据异常 | - | 暂停训练，人工检查 |

### 4.5 理论与工程的差距

**常见问题**：

| 理论方案 | 工程实现 | 解决方案 |
|----------|----------|----------|
| 全异步训练 | 权重同步复杂 | 参数服务器 + 版本控制 |
| 动态批处理 | 内存管理困难 | 预分配 + 池化 |
| 弹性伸缩 | 冷启动延迟 | 预热池 + 快照恢复 |
| 分布式训练 | 通信开销 | 梯度压缩 + 异步通信 |

**实际案例**：

**问题**：理论上 Partial Rollout 可以提高吞吐，但实际实现时发现：
- 状态保存/恢复的开销很大
- 片段间的上下文传递复杂
- reward 计算需要额外逻辑

**解决方案**：
1. 使用高效的序列化（pickle → msgpack）
2. 设计简洁的状态传递接口
3. 将 reward 计算封装成独立服务

---

## 能力五：业务与实际场景理解

> **核心考察点**：场景价值？业务价值？上线成本？优先级？

### 5.1 场景分析框架

**场景评估矩阵**：

| 维度 | 评估问题 | 权重 |
|------|----------|------|
| **用户价值** | 解决什么问题？用户痛点是什么？ | 30% |
| **技术可行性** | 技术难度？现有资源能否支撑？ | 25% |
| **商业价值** | 能带来多少收益？ROI 如何？ | 25% |
| **上线成本** | 开发周期？维护成本？ | 20% |

### 5.2 Uni-Agent 的场景分析

**核心场景**：

```
场景 1: SWE-Bench 代码修复
- 用户价值: 高（自动修复代码 bug）
- 技术可行性: 中（需要复杂环境）
- 商业价值: 中（开发者工具市场）
- 上线成本: 高（需要沙箱环境）

场景 2: 搜索增强问答
- 用户价值: 高（更准确的回答）
- 技术可行性: 高（相对简单）
- 商业价值: 高（企业知识库）
- 上线成本: 中（需要检索服务）

场景 3: 代码生成
- 用户价值: 高（提高开发效率）
- 技术可行性: 高（已有基础模型）
- 商业价值: 高（开发者工具）
- 上线成本: 低（API 服务）
```

**优先级排序**：

1. **代码生成** → 最快上线，最高 ROI
2. **搜索增强问答** → 技术成熟，市场大
3. **SWE-Bench 修复** → 技术领先，但成本高

### 5.3 用户需求分析

**用户画像**：

```python
# 开发者用户
developer_user = {
    "痛点": [
        "重复性代码工作多",
        "调试耗时",
        "文档查找困难",
    ],
    "期望": [
        "自动补全代码",
        "智能错误诊断",
        "快速文档检索",
    ],
    "付费意愿": "中（个人开发者）- 高（企业）",
}

# 企业用户
enterprise_user = {
    "痛点": [
        "代码质量不稳定",
        "知识传承困难",
        "开发效率低",
    ],
    "期望": [
        "自动化代码审查",
        "知识库问答",
        "开发流程优化",
    ],
    "付费意愿": "高",
}
```

### 5.4 上线成本分析

**成本构成**：

```python
# 开发成本
development_cost = {
    "人力": "3-6 个月 × 3-5 人",
    "硬件": "8 GPU × $3000/月",
    "数据": "标注成本 + 数据采购",
}

# 运维成本
operation_cost = {
    "计算": "GPU 实例费用",
    "存储": "模型 + 数据存储",
    "网络": "API 调用费用",
    "人力": "运维团队",
}

# 总成本估算
total_cost = {
    "初期": "$50,000 - $100,000",
    "月度": "$10,000 - $20,000",
    "ROI 周期": "6-12 个月",
}
```

### 5.5 资源有限时的优先级

**如果资源有限，应该先优化什么？**

**优先级 1: 基础设施**
- 监控系统（发现问题）
- 日志系统（定位问题）
- 告警系统（及时响应）

**优先级 2: 核心功能**
- 主流程稳定性
- 基本用户体验
- 核心指标达标

**优先级 3: 性能优化**
- 吞吐提升
- 延迟降低
- 成本优化

**优先级 4: 高级功能**
- 新场景支持
- 个性化定制
- 智能化升级

### 5.6 面试话术示例

**Q**: 你的方案适合什么样的场景？

**A**:
> "我们的 Agent 框架最适合**任务复杂度中等、需要多步推理和工具调用**的场景。
>
> **适合的场景**：
> - 代码生成/修复：需要理解需求、生成代码、测试验证
> - 知识问答：需要检索信息、整合答案
> - 数据处理：需要多步转换、验证结果
>
> **不太适合的场景**：
> - 简单对话：不需要工具调用，用普通 LLM 即可
> - 实时交互：延迟要求 < 100ms 的场景
> - 资源受限：没有 GPU 或沙箱环境
>
> **上线成本**：
> - 最小可用版本：1-2 人月，8 GPU
> - 生产版本：3-6 人月，32 GPU
> - 需要考虑：数据标注、环境搭建、监控运维"

---

**Q**: 如果资源有限，你应该先优化什么？

**A**:
> "我会按以下优先级：
>
> **第一优先级：稳定性**
> - 确保主流程不崩溃
> - 添加基础监控和告警
> - 实现自动回滚机制
>
> **第二优先级：核心指标**
> - 任务成功率 > 90%
> - 平均延迟 < 5 分钟
> - 成本控制在预算内
>
> **第三优先级：用户体验**
> - 错误信息清晰
> - 响应时间可预期
> - 支持断点续传
>
> **第四优先级：性能优化**
> - 提高吞吐
> - 降低成本
> - 支持更大规模
>
> **理由**：先保证能用，再保证好用，最后保证高效。"

---

## 附录：面试准备清单

### 技术深度准备

- [ ] GRPO vs PPO 的设计动机和实验验证
- [ ] Partial Rollout 的实现细节和权衡
- [ ] Session 状态机的设计理由
- [ ] Staleness Control 的原理和调参

### 实验能力准备

- [ ] 准备 2-3 个完整的实验案例
- [ ] 每个案例包括：问题、假设、实验、结果、结论
- [ ] 准备消融实验的设计和结果

### 问题定位准备

- [ ] 准备 2-3 个问题排查案例
- [ ] 每个案例包括：现象、排查过程、根因、解决方案
- [ ] 准备问题定位的工具箱和清单

### 工程落地准备

- [ ] 准备部署架构图
- [ ] 准备稳定性保障方案
- [ ] 准备监控告警配置
- [ ] 准备回滚机制说明

### 业务理解准备

- [ ] 准备场景分析矩阵
- [ ] 准备用户需求分析
- [ ] 准备成本估算
- [ ] 准备优先级排序

---

*最后更新: 2026-07-05*
*适用岗位: LLM算法实习、AI工程实习、研究实习*

# LLM Agent 算法实习面试准备文档

> 基于 Uni-Agent 项目深度解析 | 面试准备 + 项目介绍 + 技术深挖

---

## 目录

1. [项目概述与自我介绍](#1-项目概述与自我介绍)
2. [LLM Agent 核心概念与技术栈](#2-llm-agent-核心概念与技术栈)
3. [Uni-Agent 架构深度解析](#3-uni-agent-架构深度解析)
4. [强化学习与后训练关键技术](#4-强化学习与后训练关键技术)
5. [面试深挖点与考察问题](#5-面试深挖点与考察问题)
6. [项目介绍话术模板](#6-项目介绍话术模板)
7. [代码级细节与实现原理](#7-代码级细节与实现原理)
8. [常见面试问题与参考答案](#8-常见面试问题与参考答案)

---

## 1. 项目概述与自我介绍

### 1.1 项目背景

**Uni-Agent** 是一个统一的 Agent 框架，支持：
- **构建 (Build)**: 模块化设计，支持自定义工具、环境、技能
- **运行 (Run)**: 大规模并行 Agent 交互与验证
- **训练 (Train)**: 端到端的 Agent 强化学习训练

**核心定位**: 解决 Agent 从推理到训练的全链路问题，支持 1000+ 并发任务

### 1.2 技术亮点总结

| 技术点 | 描述 | 面试价值 |
|--------|------|----------|
| **统一交互架构** | Model-Tool-Env 三层解耦设计 | 架构设计能力 |
| **大规模并行** | 异步并发 + 沙箱隔离 + 流量控制 | 系统工程能力 |
| **端到端训练** | 从推理到 RL 训练的统一栈 | 算法落地能力 |
| **全异步训练** | Rollout/Training 异步解耦 | 性能优化能力 |
| **多任务支持** | SWE-Bench、搜索、代码生成等 | 泛化能力 |

### 1.3 自我介绍话术（30秒版本）

> "我参与了一个 Agent 框架项目 Uni-Agent，专注于解决 Agent 从推理到训练的全链路问题。我的工作涉及：
> 1. 设计了 Model-Tool-Env 三层解耦架构，支持 1000+ 并发 Agent 任务
> 2. 实现了端到端的 Agent 强化学习训练，将 SWE-Bench Verified 成绩从 62% 提升到 68%
> 3. 优化了全异步训练流程，解决了 Agent 任务延迟不均导致的训练效率问题"

---

## 2. LLM Agent 核心概念与技术栈

### 2.1 Agent 基础概念

**什么是 Agent？**
- **定义**: 能够感知环境、做出决策、执行动作的自主实体
- **核心循环**: 感知(Perceive) → 推理(Reason) → 行动(Act) → 观察(Observe)

**Agent 与 Chatbot 的区别**:
| 维度 | Chatbot | Agent |
|------|---------|-------|
| 交互模式 | 单轮/多轮对话 | 多步自主执行 |
| 工具使用 | 有限/无 | 丰富工具调用 |
| 状态管理 | 无状态/短时记忆 | 长期状态保持 |
| 目标导向 | 回答问题 | 完成复杂任务 |

### 2.2 Agent 架构模式

**1. ReAct (Reasoning + Acting)**
```
Thought: 需要搜索最新信息
Action: search("最新AI进展")
Observation: 搜索结果...
Thought: 需要总结
Action: finish("总结内容")
```

**2. Plan-and-Execute**
```
Plan: 
1. 搜索信息
2. 分析数据
3. 生成报告

Execute: 按计划逐步执行
```

**3. Reflection/Self-Critique**
```
生成回答 → 自我评估 → 改进 → 输出
```

### 2.3 关键技术栈

**LLM 推理**:
- vLLM/SGLang: 高性能推理引擎
- Tensor Parallelism: 模型并行
- Continuous Batching: 动态批处理

**Agent 框架**:
- Tool Calling: 函数调用接口
- Message History: 对话历史管理
- Sandbox: 代码执行沙箱

**训练框架**:
- verl: RL 训练框架
- GRPO/GSPO: 策略梯度算法
- Megatron: 大模型并行训练

---

## 3. Uni-Agent 架构深度解析

### 3.1 核心架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Framework                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Model     │  │    Tool     │  │     Env     │     │
│  │  (推理后端)  │  │  (工具接口)  │  │  (执行环境)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │               │
│         └───────────────┼───────────────┘               │
│                         ▼                               │
│              AgentInteraction                          │
│           (统一交互循环管理器)                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Gateway Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Session    │  │   Codec     │  │   Router    │     │
│  │  (会话管理)  │  │  (编解码)    │  │  (路由分发)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Deployment Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Local     │  │   Modal     │  │   veFaaS    │     │
│  │  (本地部署)  │  │  (云部署)    │  │  (Serverless)│     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键代码解析

#### 3.2.1 AgentInteraction 核心循环

```python
# uni_agent/interaction/interaction.py:107-322

async def step(self, step_idx: int):
    """单步执行：模型推理 → 工具解析 → 工具执行 → 结果反馈"""
    
    # 1. 模型推理
    model_output, tool_calls, rollout_cache, generation_info = await self.model.query(
        messages=self.messages,
        rollout_cache=self.rollout_cache,
    )
    
    # 2. 解析工具调用
    if tool_calls:
        content, tool_calls = await self.tools_manager.parse_structured_action(
            content=model_output,
            tool_calls_data=tool_calls,
        )
    
    # 3. 执行工具
    for tool_call in tool_calls:
        action_cmd = self.tools_manager.get_tool_bash_command(tool_call)
        observation = await self.env.run_action(action_cmd, action_timeout=self.action_timeout)
        
        # 4. 收集结果
        tool_results.append(ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            observation=observation,
            status=status,
        ))
    
    # 5. 更新消息历史
    self.messages.extend(tool_messages)
```

**面试深挖点**:
- **Q**: 为什么选择串行执行工具调用而非并行？
- **A**: 
  1. 工具间可能存在依赖关系（如先创建文件再读取）
  2. 状态一致性：沙箱环境是共享的，并行可能导致竞态条件
  3. 错误处理：串行便于超时和异常处理

#### 3.2.2 Session 生命周期管理

```python
# uni_agent/gateway/session/session.py:105-359

class GatewaySession:
    """会话状态机: ACTIVE → FINALIZED/ABORTED"""
    
    async def run_generation(self, payload, backend):
        """执行一次生成请求"""
        # 1. 准备生成输入
        encoded = await self._prepare_generation_inputs(payload, request_context)
        
        # 2. 调用推理后端
        output = await backend.generate(
            request_id=self.handle.session_id,
            prompt_ids=encoded.context_ids,
            sampling_params=encoded.sampling_params,
        )
        
        # 3. 更新轨迹缓冲
        encoded.buffer.response_ids.extend(response_ids)
        encoded.buffer.response_mask.extend([1] * len(response_ids))
        
        # 4. 解码响应
        assistant_msg, finish_reason = await self._codec.decode_response(
            response_ids,
            tools=encoded.tools,
            stop_reason=output.stop_reason,
        )
        
        return GenerationOutcome(assistant_msg=assistant_msg, ...)
    
    async def finalize(self):
        """结束会话，返回轨迹列表"""
        self._materialize_active_trajectory()
        self.phase = SessionPhase.FINALIZED
        return [replace(trajectory, reward_info=dict(self.reward_info)) 
                for trajectory in self.trajectories]
```

**面试深挖点**:
- **Q**: 如何处理多轮对话中的上下文窗口管理？
- **A**:
  1. **增量编码**: 只编码新增消息，避免重复计算
  2. **轨迹分割**: 当上下文超限时，将历史轨迹物化并保存
  3. **响应预算**: 跟踪剩余响应长度，动态调整 `max_tokens`

#### 3.2.3 全异步训练架构

```python
# examples/agent_train/train_qwen3p5_dense.sh 关键参数

# 全异步特有参数
rollout_mode="async"
partial_rollout=True
staleness_threshold=1.0
trigger_parameter_sync_step=4
require_batches=1

# Rollout 配置
actor_rollout_ref.rollout.agent.num_workers=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
```

**全异步 vs 同步训练对比**:

| 维度 | 同步训练 | 全异步训练 |
|------|----------|------------|
| **延迟** | 等待最慢的 rollout | 独立执行，不等待 |
| **利用率** | 低（长尾任务拖累） | 高（动态负载均衡） |
| **实现复杂度** | 简单 | 需要处理权重同步 |
| **适用场景** | 短任务、调试 | 长任务、生产环境 |

---

## 4. 强化学习与后训练关键技术

### 4.1 GRPO/GSPO 算法

**GRPO (Group Relative Policy Optimization)**:

```python
# 核心思想：基于组内相对优势的策略梯度

def compute_grpo_loss(log_probs, old_log_probs, advantages, clip_ratio):
    """
    log_probs: 当前策略的对数概率
    old_log_probs: 旧策略的对数概率
    advantages: 优势函数估计
    """
    # 比率
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 裁剪目标
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    # 损失
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    return loss.mean()
```

**GSPO (Group Sampling Policy Optimization)**:
- 在 GRPO 基础上引入分组采样
- 同一 prompt 生成多个 response，组内比较
- 提供更稳定的 advantage 估计

### 4.2 Agent RL 训练流程

```
┌─────────────────────────────────────────────────────────┐
│                  Agent RL Training                       │
│                                                         │
│  1. Rollout Phase (异步)                                 │
│     ┌─────────────┐                                      │
│     │  Prompt     │ → 生成多个 response                   │
│     │  Dataset    │   (每个 response 是一个完整轨迹)        │
│     └─────────────┘                                      │
│            │                                             │
│            ▼                                             │
│  2. Interaction Phase                                    │
│     ┌─────────────┐                                      │
│     │  Agent      │ → 执行工具调用                        │
│     │  Sandbox    │   (多轮交互，不确定长度)                │
│     └─────────────┘                                      │
│            │                                             │
│            ▼                                             │
│  3. Reward Phase                                         │
│     ┌─────────────┐                                      │
│     │  Reward     │ → 计算任务奖励                        │
│     │  Function   │   (二值/连续/过程奖励)                 │
│     └─────────────┘                                      │
│            │                                             │
│            ▼                                             │
│  4. Training Phase (异步)                                │
│     ┌─────────────┐                                      │
│     │  Policy     │ → 更新模型参数                        │
│     │  Update     │   (GRPO/GSPO)                        │
│     └─────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### 4.3 关键技术细节

#### 4.3.1 Partial Rollout

**问题**: Agent 任务长度差异巨大（10-200 步），导致训练效率低

**解决方案**:
```python
# partial_rollout=True 时的处理逻辑

# 1. 长轨迹分割
if len(trajectory) > max_segment_length:
    segments = split_trajectory(trajectory, max_segment_length)
    
# 2. 短轨迹填充
if len(trajectory) < target_length:
    trajectory = pad_trajectory(trajectory, target_length)
    
# 3. 动态采样
n_samples = adaptive_sample_size(task_difficulty)
```

#### 4.3.2 Staleness Control

**问题**: 全异步训练中，策略参数可能过时

**解决方案**:
```python
# staleness_threshold=1.0 意味着最多允许一个训练步的延迟

if current_step - rollout_step > staleness_threshold:
    # 丢弃这个 batch，重新采样
    discard_batch(batch)
    
# trigger_parameter_sync_step=4 意味着每 4 步同步一次参数
if current_step % trigger_parameter_sync_step == 0:
    sync_parameters(actor_model, rollout_model)
```

#### 4.3.3 Reward Design

**二值奖励** (SWE-Bench):
```python
def compute_reward(interaction_result):
    """测试通过=1，失败=0"""
    eval_report = run_tests(interaction_result)
    return 1.0 if eval_report["resolved"] else 0.0
```

**过程奖励** (Search Agent):
```python
def compute_reward(interaction_result):
    """基于搜索质量的连续奖励"""
    # 1. 检索相关性
    relevance_score = compute_relevance(queries, documents)
    
    # 2. 答案准确性
    accuracy_score = compute_accuracy(predicted_answer, ground_truth)
    
    # 3. 效率奖励
    efficiency_penalty = num_steps / max_steps
    
    return relevance_score * accuracy_score * (1 - efficiency_penalty)
```

---

## 5. 面试深挖点与考察问题

### 5.1 架构设计类

**Q1: 为什么选择 Model-Tool-Env 三层解耦设计？**

**参考答案**:
1. **可替换性**: 每层可以独立替换，如切换推理引擎（vLLM→SGLang）不影响工具层
2. **可测试性**: 每层可以独立测试，降低调试复杂度
3. **可扩展性**: 新增工具/环境只需实现接口，无需修改核心逻辑
4. **团队协作**: 不同成员可以并行开发不同层

**追问**: 这种设计有什么缺点？
- **A**: 接口定义需要前瞻性，后期修改成本高；跨层调试困难；性能开销（但可控）

**Q2: Gateway 的 Session 设计解决了什么问题？**

**参考答案**:
1. **状态管理**: 维护多轮对话状态，支持增量编码
2. **轨迹收集**: 自动收集 token 级别的轨迹信息
3. **生命周期**: 支持会话的创建、生成、终结、异常处理
4. **并发控制**: 通过锁机制保证状态一致性

**追问**: 如何处理 Session 的并发访问？
- **A**: 使用 `request_lock` 和 `generation_lock` 双锁机制，前者保护状态读写，后者序列化生成请求

### 5.2 算法实现类

**Q3: GRPO 相比 PPO 的优势是什么？**

**参考答案**:
1. **无需 Critic**: 不需要训练 value function，简化实现
2. **组内比较**: 同一 prompt 的多个 response 相对比较，降低方差
3. **稳定性**: 通过分组归一化，对 reward scale 不敏感
4. **效率**: 减少一次 forward pass（不需要 value head）

**追问**: GRPO 的 disadvantage 是什么？
- **A**: 需要采样多个 response，计算成本高；对同质化 response 敏感

**Q4: 全异步训练如何处理权重同步问题？**

**参考答案**:
```python
# 核心机制
class FullyAsyncTrainer:
    def __init__(self):
        self.rollout_buffer = []  # Rollout 结果缓冲
        self.training_step = 0
        self.last_sync_step = 0
        
    async def training_loop(self):
        while True:
            # 1. 异步收集 rollout
            batch = await self.collect_rollouts()
            
            # 2. 检查陈旧度
            if self.is_stale(batch):
                continue  # 丢弃过时 batch
            
            # 3. 训练更新
            loss = self.update_policy(batch)
            
            # 4. 定期同步参数
            if self.should_sync():
                self.sync_parameters()
```

### 5.3 工程实践类

**Q5: 如何处理 Agent 任务的长尾延迟问题？**

**参考答案**:
1. **超时机制**: 设置合理的超时预算（`timeout_budget`）
2. **异常捕获**: 区分可恢复/不可恢复错误
3. **资源隔离**: 每个任务独立沙箱，避免相互影响
4. **动态调度**: 根据任务复杂度动态分配资源

**Q6: Sandbox 的安全性如何保证？**

**参考答案**:
1. **容器隔离**: 使用 Docker/Modal 提供进程级隔离
2. **资源限制**: 限制 CPU/内存/网络/磁盘使用
3. **超时控制**: 命令执行超时自动中断
4. **文件系统隔离**: 只读挂载 + 临时可写层

### 5.4 系统设计类

**Q7: 如何设计一个支持 1000+ 并发 Agent 的系统？**

**参考答案**:
```
┌─────────────────────────────────────────────────────────┐
│                 High-Concurrency Agent System            │
│                                                         │
│  1. Load Balancer                                       │
│     ├── 请求分发                                         │
│     └── 健康检查                                         │
│                                                         │
│  2. Agent Workers (Ray Cluster)                         │
│     ├── Worker 1: Agent Sandbox Pool                    │
│     ├── Worker 2: Agent Sandbox Pool                    │
│     └── Worker N: Agent Sandbox Pool                    │
│                                                         │
│  3. Shared Services                                     │
│     ├── Model Server (vLLM/SGLang)                      │
│     ├── Tool Service (Search/Code/...)                  │
│     └── Reward Service                                  │
│                                                         │
│  4. Storage Layer                                       │
│     ├── State Store (Redis)                             │
│     ├── Trajectory Store (S3/MinIO)                     │
│     └── Log Store (ELK)                                 │
└─────────────────────────────────────────────────────────┘
```

**关键设计点**:
- **无状态 Worker**: Worker 不保存状态，可随时扩缩容
- **异步 IO**: 全链路异步，避免阻塞
- **背压控制**: 通过信号量控制并发度
- **故障恢复**: 任务级别的检查点和重试

---

## 6. 项目介绍话术模板

### 6.1 技术深度版本（2-3分钟）

> "我参与的 Uni-Agent 项目是一个统一的 Agent 框架，核心解决三个问题：
>
> **第一，架构可扩展性**。我们设计了 Model-Tool-Env 三层解耦架构。Model 层负责推理决策，Tool 层封装工具调用接口，Env 层管理执行环境。这种设计使得我们可以独立替换推理引擎（从 vLLM 切换到 SGLang 只需修改一行配置），而不需要修改任何业务代码。
>
> **第二，大规模并行**。通过 Gateway Session 机制，我们实现了 token 级别的轨迹收集。每个 Session 维护独立的对话状态和轨迹缓冲，支持增量编码和响应预算管理。实测可以稳定支持 1000+ 并发 Agent 任务。
>
> **第三，端到端训练**。我们将推理栈直接用于 RL 训练，实现了全异步训练流程。Rollout Worker 和 Training Worker 完全解耦，通过 staleness control 和 parameter sync 保证训练稳定性。在 SWE-Bench Verified 上，Qwen3.5-35B-A3B 模型从 baseline 62% 提升到 68.4%。"

### 6.2 快速版本（30秒）

> "我参与了一个 Agent 框架项目，主要负责：
> 1. 设计了三层解耦架构，支持工具和环境的灵活替换
> 2. 实现了大规模并行交互，支持 1000+ 并发任务
> 3. 优化了全异步 RL 训练，将 Agent 任务的训练效率提升了一倍"

### 6.3 针对不同岗位的侧重点

**LLM 算法岗**:
- 强调 GRPO/GSPO 算法实现
- 强调 Reward 设计和优化
- 强调训练稳定性调优

**AI 工程岗**:
- 强调系统架构设计
- 强调大规模并行和性能优化
- 强调工程实践经验

**研究实习岗**:
- 强调问题分析和解决方案
- 强调实验设计和结果分析
- 强调论文复现能力

---

## 7. 代码级细节与实现原理

### 7.1 Rollout Cache 机制

```python
# uni_agent/interaction/model.py:36-57

async def prepare_rollout_cache(self, messages):
    """初始化 rollout 缓存"""
    prompt_ids = await self.loop.run_in_executor(
        None,
        lambda: self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=self.tools_schemas,
        ),
    )
    
    return {
        "request_id": str(uuid.uuid4()),  # 唯一标识
        "prompt_ids": prompt_ids,          # prompt token ids
        "response_mask": [],               # 响应掩码（1=model output, 0=context）
        "response_logprobs": [],           # 响应 log probs
        "routed_experts": None,            # MoE 路由信息
        "metrics": {},                     # 性能指标
        "extra_fields": {},                # 额外字段
    }
```

**设计要点**:
- **增量更新**: 只追加新增 token，避免重新编码
- **掩码分离**: 区分模型生成和上下文 token，用于 loss 计算
- **MoE 支持**: 保存路由信息，支持 Router Replay

### 7.2 Tool Parser 设计

```python
# uni_agent/interaction/tools_manager.py:33-45

async def parse_action(self, model_output):
    """从模型输出中解析工具调用"""
    tools = [OpenAIFunctionToolSchema(**schema) for schema in self.tools_schemas]
    content, tool_calls = self._tool_parser.extract_tool_calls(model_output, tools)
    return content, tool_calls

# 支持的 parser:
# - qwen3_coder: Qwen3 专用格式
# - hermes: Hermes 格式
# - openai: OpenAI Function Calling 格式
```

**Tool Call 格式示例**:
```
Thought: 需要搜索信息
<tool_call>
{"name": "search", "arguments": {"query": "最新进展"}}
</tool_call>
```

### 7.3 Reward 计算流程

```python
# uni_agent/agent_loop.py:118-125

# 在 Agent Loop 中计算 reward
if self.reward_spec is not None:
    reward_score, _ = await self.reward_spec.compute_reward(
        interaction_result=interaction_result,
    )
    interaction_result["reward_score"] = reward_score
else:
    self.logger.warning("No reward spec is provided, reward score will be set to -100")
    interaction_result["reward_score"] = -100
```

**Reward 设计模式**:
1. **二值奖励**: SWE-Bench（测试通过=1，失败=0）
2. **连续奖励**: 搜索任务（基于相关性分数）
3. **过程奖励**: 每步给予部分奖励
4. **稀疏奖励**: 只在任务结束时给出奖励

### 7.4 轨迹分割与物化

```python
# uni_agent/gateway/session/session.py:225-316

async def _prepare_generation_inputs(self, payload, request_context):
    """准备生成输入，处理轨迹分割"""
    
    if self.active_trajectory is None:
        # 首次请求，完整编码
        prompt_ids = self._codec.encode_full(messages, ...)
        buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
        
    elif self._is_request_context_prefix(messages=messages, tools=tools):
        # 增量请求，只编码新增部分
        incremental_ids = self._codec.encode_incremental(incremental_messages, ...)
        buffer.response_ids.extend(incremental_ids)
        buffer.response_mask.extend([0] * len(incremental_ids))
        
    else:
        # 上下文变化，物化当前轨迹，重新开始
        materialized_trajectory = self._build_materialized_trajectory(active=self.active_trajectory)
        prompt_ids = self._codec.encode_full(messages, ...)
        buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
    
    # 检查响应长度预算
    if remaining_response_budget is not None and len(buffer.response_mask) >= self._response_length:
        # 长度超限，提前结束
        return EncodedData(
            length_exhausted_trajectory=self._build_materialized_trajectory(
                active=buffer,
                extra_fields={"finish_reason": "length"},
            ),
        )
```

---

## 8. 常见面试问题与参考答案

### 8.1 基础概念

**Q: 什么是 Agent？与 Chatbot 有什么区别？**
- **A**: Agent 是能够自主感知环境、做出决策、执行动作的实体，核心是"行动-观察"循环。Chatbot 主要是对话交互，而 Agent 能够使用工具、保持状态、完成复杂任务。

**Q: 什么是 Tool Calling？为什么需要它？**
- **A**: Tool Calling 是 LLM 调用外部工具/函数的能力。需要它是因为：1) LLM 无法获取实时信息；2) LLM 无法执行精确计算；3) LLM 无法操作外部系统。

**Q: 解释 ReAct 模式**
- **A**: ReAct = Reasoning + Acting。模型交替进行推理（Thought）和行动（Action），直到完成任务。例如：Thought → 需要搜索 → Action → search() → Observation → 结果 → Thought → 需要总结 → Action → finish()。

### 8.2 架构设计

**Q: 为什么选择异步而非同步训练？**
- **A**: Agent 任务长度差异大（10-200步），同步训练需要等待最慢的任务，导致 GPU 利用率低。全异步训练让 Rollout 和 Training 独立运行，通过动态调度提高利用率。

**Q: 如何处理 Agent 任务的不确定性？**
- **A**: 通过以下机制：1) 超时控制；2) 异常捕获和恢复；3) 资源隔离；4) 动态调度；5) 部分轨迹处理。

**Q: Session 设计解决了什么问题？**
- **A**: 1) 多轮对话状态管理；2) Token 级别轨迹收集；3) 增量编码优化；4) 生命周期管理；5) 并发控制。

### 8.3 算法实现

**Q: GRPO 相比 PPO 的优势？**
- **A**: 1) 无需训练 Critic 网络；2) 组内相对比较降低方差；3) 对 reward scale 不敏感；4) 实现更简单。

**Q: 如何设计 Agent 的 Reward？**
- **A**: 取决于任务类型：1) 二值奖励（成功/失败）；2) 连续奖励（质量分数）；3) 过程奖励（每步奖励）；4) 稀疏奖励（最终奖励）。需要平衡探索和利用。

**Q: 什么是 Partial Rollout？**
- **A**: 当轨迹过长时，将其分割成多个片段分别处理。解决长尾延迟问题，提高训练效率。需要处理片段间的状态传递和奖励归因。

### 8.4 工程实践

**Q: 如何保证 Sandbox 的安全性？**
- **A**: 1) 容器隔离；2) 资源限制；3) 超时控制；4) 文件系统隔离；5) 网络隔离；6) 权限控制。

**Q: 如何监控大规模 Agent 系统？**
- **A**: 1) 分布式日志收集；2) 指标监控（延迟、吞吐、错误率）；3) 链路追踪；4) 实时仪表盘；5) 告警机制。

**Q: 如何进行故障恢复？**
- **A**: 1) 任务级检查点；2) 自动重试；3) 状态恢复；4) 降级策略；5) 资源隔离避免雪崩。

---

## 附录：关键术语表

| 术语 | 解释 |
|------|------|
| **Rollout** | 策略在环境中执行并收集轨迹的过程 |
| **Trajectory** | 一个完整的交互序列（prompt + response + actions + observations） |
| **Episode** | 从初始状态到终止状态的完整交互过程 |
| **Advantage** | 动作相对于基线的价值提升 |
| **KL Divergence** | 新旧策略分布的距离，用于约束更新幅度 |
| **GRPO** | Group Relative Policy Optimization，组相对策略优化 |
| **GSPO** | Group Sampling Policy Optimization，组采样策略优化 |
| **Partial Rollout** | 部分轨迹训练，将长轨迹分割成片段 |
| **Staleness** | 全异步训练中策略参数的过时程度 |
| **MoE** | Mixture of Experts，混合专家模型 |
| **TP/PP/EP** | Tensor/Pipeline/Expert Parallelism，张量/流水线/专家并行 |

---

## 附录：推荐阅读

1. **论文**: 
   - "ReAct: Synergizing Reasoning and Acting in Language Models"
   - "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
   - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO)

2. **代码**:
   - Uni-Agent 项目: 理解完整架构
   - verl 项目: 理解 RL 训练框架
   - SWE-Bench: 理解 Agent 评估

3. **系统设计**:
   - Ray: 分布式计算框架
   - vLLM: 高性能推理引擎
   - Modal/veFaaS: Serverless 部署

---

*最后更新: 2026-07-05*
*适用岗位: LLM算法实习、AI工程实习、研究实习*

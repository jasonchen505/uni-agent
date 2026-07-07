# Uni-Agent 8卡4090全流程复现计划

> 基于实际硬件资源的可行性评估与详细执行方案

---

## 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [可行性分析](#2-可行性分析)
3. [复现路线图](#3-复现路线图)
4. [分阶段详细计划](#4-分阶段详细计划)
5. [配置适配方案](#5-配置适配方案)
6. [常见问题与解决方案](#6-常见问题与解决方案)
7. [学习路径与里程碑](#7-学习路径与里程碑)

---

## 1. 硬件资源评估

### 1.1 8卡4090规格

| 参数 | 数值 |
|------|------|
| GPU型号 | RTX 4090 |
| 显存 | 24GB GDDR6X × 8 = **192GB 总显存** |
| FP16算力 | ~82 TFLOPS × 8 = **656 TFLOPS** |
| INT8算力 | ~164 TOPS × 8 = **1312 TOPS** |
| NVLink | 无（4090不支持NVLink） |
| PCIe | PCIe 4.0 x16 |

### 1.2 与原项目硬件对比

| 维度 | 原项目配置 | 8卡4090 | 差距 |
|------|------------|---------|------|
| GPU显存 | 80GB A100/H100 | 24GB 4090 | 3.3x差距 |
| GPU数量 | 32-64卡（多节点） | 8卡（单节点） | 4-8x差距 |
| 互联带宽 | NVLink 600GB/s | PCIe 32GB/s | 19x差距 |
| 总算力 | 2500+ TFLOPS | 656 TFLOPS | 4x差距 |

### 1.3 显存需求分析

**模型显存需求估算**：

| 模型大小 | FP16显存 | INT8显存 | 4090可否运行 |
|----------|----------|----------|--------------|
| 1.5B | 3GB | 1.5GB | ✓ 单卡 |
| 4B | 8GB | 4GB | ✓ 单卡 |
| 8B | 16GB | 8GB | ✓ 单卡/TP2 |
| 14B | 28GB | 14GB | ⚠️ 需TP2 |
| 30B | 60GB | 30GB | ⚠️ 需TP4 |
| 70B | 140GB | 70GB | ❌ 需多节点 |

**训练显存需求**（含优化器状态）：

| 模型大小 | 训练显存（AdamW） | 4090配置 |
|----------|-------------------|----------|
| 1.5B | ~12GB | 单卡可行 |
| 4B | ~32GB | 需TP2 + Offload |
| 8B | ~64GB | 需TP4 + Offload |
| 14B | ~112GB | 需TP4 + 重度Offload |

---

## 2. 可行性分析

### 2.1 可行的复现范围

**✓ 完全可行**：
- 项目代码理解和架构学习
- 小模型（1.5B-4B）的推理和Agent交互
- 搜索Agent（Search Agent）的完整流程
- 本地沙箱环境搭建
- 单节点训练流程验证

**⚠️ 部分可行（需适配）**：
- SWE-Bench任务（需简化环境）
- RL训练（需小模型 + Offload）
- 大规模并行推理（需降低并发）

**✗ 不可行**：
- 原项目的大规模训练（30B+模型）
- 多节点分布式训练
- 完整的SWE-Bench评测（资源不足）

### 2.2 推荐复现策略

**策略：渐进式复现，从小到大**

```
阶段1: 环境搭建 + 代码理解（1-2周）
  ↓
阶段2: 小模型推理验证（1周）
  ↓
阶段3: 搜索Agent复现（2周）
  ↓
阶段4: 简化RL训练（2周）
  ↓
阶段5: 进阶实验与优化（可选）
```

---

## 3. 复现路线图

### 3.1 总体时间线

```
Week 1-2: 环境准备与代码学习
├── Day 1-3: 硬件环境配置
├── Day 4-7: 项目代码阅读
└── Day 8-14: 核心模块理解

Week 3: 推理验证
├── Day 15-17: 下载小模型
├── Day 18-19: 推理环境搭建
└── Day 20-21: Agent交互验证

Week 4-5: 搜索Agent复现
├── Day 22-25: LocalWiki搭建
├── Day 26-28: 搜索Agent配置
└── Day 29-35: 端到端测试

Week 6-7: RL训练复现
├── Day 36-40: 训练环境配置
├── Day 41-42: 数据准备
└── Day 43-49: 训练运行与调优

Week 8: 总结与文档
├── Day 50-52: 实验记录
├── Day 53-54: 问题总结
└── Day 55-56: 文档整理
```

### 3.2 里程碑定义

| 里程碑 | 目标 | 验证标准 |
|--------|------|----------|
| M1 | 环境搭建完成 | 所有依赖安装成功 |
| M2 | 代码理解完成 | 能画出架构图并讲解 |
| M3 | 推理验证完成 | 成功运行Agent交互 |
| M4 | 搜索Agent完成 | 端到端搜索问答 |
| M5 | RL训练完成 | 训练loss收敛 |
| M6 | 实验报告完成 | 文档和代码提交 |

---

## 4. 分阶段详细计划

### 阶段1: 环境准备与代码学习（Week 1-2）

#### 4.1.1 硬件环境配置

**Day 1-3: 基础环境**

```bash
# 1. 检查GPU状态
nvidia-smi

# 2. 安装CUDA（如果未安装）
# 推荐CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 3. 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 4. 创建虚拟环境
conda create -n uni-agent python=3.10 -y
conda activate uni-agent
```

**验证命令**：
```bash
python -c "import torch; print(torch.cuda.device_count())"  # 应输出8
python -c "import torch; print(torch.cuda.get_device_name(0))"  # 应输出RTX 4090
```

#### 4.1.2 项目代码阅读计划

**Day 4-7: 核心模块阅读**

按以下顺序阅读代码：

| 优先级 | 模块 | 文件 | 目标 |
|--------|------|------|------|
| P0 | 入口 | `uni_agent/agent_loop.py` | 理解整体流程 |
| P0 | 交互 | `uni_agent/interaction/interaction.py` | 理解Agent循环 |
| P0 | 模型 | `uni_agent/interaction/model.py` | 理解模型调用 |
| P1 | 工具 | `uni_agent/interaction/tools_manager.py` | 理解工具管理 |
| P1 | 环境 | `uni_agent/interaction/env.py` | 理解沙箱交互 |
| P1 | 会话 | `uni_agent/gateway/session/session.py` | 理解状态管理 |
| P2 | 奖励 | `uni_agent/reward/` | 理解奖励计算 |
| P2 | 框架 | `uni_agent/framework/framework.py` | 理解训练集成 |

**Day 8-14: 深入理解**

```markdown
每日学习计划：

Day 8: Agent交互循环
- 重点：interaction.py 的 step() 方法
- 输出：画出单步执行流程图

Day 9: 工具调用机制
- 重点：tools_manager.py 的 parse_action()
- 输出：理解tool call格式

Day 10: 沙箱环境
- 重点：env.py 的 run_action()
- 输出：理解命令执行流程

Day 11: 会话管理
- 重点：session.py 的状态机
- 输出：画出状态转移图

Day 12: 奖励系统
- 重点：reward/base.py 和 reward/search.py
- 输出：理解奖励计算逻辑

Day 13: 训练集成
- 重点：framework.py 的 generate_sequences()
- 输出：理解推理到训练的数据流

Day 14: 整体回顾
- 输出：完成架构图和核心流程文档
```

#### 4.1.3 代码学习输出物

完成阶段1后，你应该能够回答：

1. **Agent交互循环**：模型调用→工具解析→工具执行→结果反馈的完整流程
2. **状态管理**：Session如何维护多轮对话状态
3. **错误处理**：超时、格式错误、沙箱死亡等异常如何处理
4. **数据流**：从prompt到trajectory到reward的完整数据流

---

### 阶段2: 推理验证（Week 3）

#### 4.2.1 模型选择与下载

**推荐模型（4090适配）**：

| 模型 | 参数量 | 显存需求 | 推荐度 |
|------|--------|----------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | 3GB | ⭐⭐⭐ 入门首选 |
| Qwen2.5-3B-Instruct | 3B | 6GB | ⭐⭐⭐ 性价比高 |
| Qwen2.5-7B-Instruct | 7B | 14GB | ⭐⭐ 性能更好 |
| Llama-3.2-3B-Instruct | 3B | 6GB | ⭐⭐ 轻量级 |

**下载命令**：

```bash
# 使用huggingface-cli下载
pip install huggingface_hub

# 下载Qwen2.5-3B（推荐）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir ~/models/Qwen2.5-3B-Instruct

# 或下载Qwen2.5-1.5B（入门）
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir ~/models/Qwen2.5-1.5B-Instruct
```

#### 4.2.2 推理环境搭建

**Day 15-17: 安装依赖**

```bash
# 进入项目目录
cd /data/home/yizhou/uni-agent

# 安装verl
git submodule update --init --recursive
pip install --no-deps -e ./verl

# 安装Uni-Agent依赖
pip install swe-rex loguru pydantic pydantic_settings aiohttp

# 安装推理引擎
pip install vllm  # 或 sglang

# 验证安装
python -c "from uni_agent.agent_loop import UniAgentLoop; print('OK')"
```

**Day 18-19: 推理测试**

```bash
# 创建推理测试脚本
cat > test_inference.py << 'EOF'
import asyncio
from uni_agent.interaction import AgentChatModel, AgentEnv, AgentEnvConfig, ToolsManager, ToolsManagerConfig

async def test_basic_inference():
    """测试基本推理功能"""
    # 1. 初始化模型
    model = AgentChatModel(
        client=None,  # 后续配置
        tokenizer=None,
        max_model_len=4096,
        sampling_params={"temperature": 0.7, "top_p": 0.9}
    )
    print("✓ 模型初始化成功")
    
    # 2. 测试tool schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute"}
                    },
                    "required": ["command"]
                }
            }
        }
    ]
    model.set_tools_schemas(tools)
    print("✓ Tool schema设置成功")
    
    print("✓ 基础推理测试通过")

if __name__ == "__main__":
    asyncio.run(test_basic_inference())
EOF

python test_inference.py
```

#### 4.2.3 Agent交互验证

**Day 20-21: 完整交互测试**

```bash
# 创建本地沙箱配置
cat > examples/agent_interaction/agent_config_local.yaml << 'EOF'
- name: local_agent
  _target_: uni_agent.agent_loop.UniAgentLoop
  concurrency: 4
  log_dir: /tmp/local_agent
  mask_abnormal_exit_traj: false
  
  interaction:
    action_timeout: 60
    max_turns: 10
  
  env:
    deployment:
      type: local  # 使用本地部署
    env_variables:
      PAGER: "cat"
  
  tools:
    - name: execute_bash
    - name: submit
EOF

# 运行简单测试
python examples/agent_interaction/parallel_infer.py \
    --data-path examples/data_preprocess/test_data.parquet \
    --model-path ~/models/Qwen2.5-3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config_local.yaml \
    --tensor-parallel-size 1 \
    --num-workers 2 \
    --max-turns 5 \
    --max-samples 2
```

---

### 阶段3: 搜索Agent复现（Week 4-5）

#### 4.3.1 LocalWiki搭建

**Day 22-25: 检索服务**

```bash
# 1. 下载预构建索引
export DATA_ROOT=${HOME}/uni_agent_data
mkdir -p "$DATA_ROOT/wiki24"

# 下载FAISS索引（约2GB）
huggingface-cli download begunner/wikipedia-2024-06-bge-m3-faiss-ivf \
    --repo-type dataset \
    --local-dir "$DATA_ROOT/wiki24"

# 合并索引文件
cd "$DATA_ROOT/wiki24"
cat wiki24_faiss.index.part?? > wiki24_faiss.index

# 2. 下载嵌入模型
huggingface-cli download BAAI/bge-m3 --local-dir "$DATA_ROOT/model/bge-m3"

# 3. 启动LocalWiki服务
cd /data/home/yizhou/uni-agent
bash uni_agent/tools/search/localwiki/run_localwiki.sh
```

**验证LocalWiki**：

```bash
# 测试检索接口
curl http://127.0.0.1:8001/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query": "What is machine learning?", "top_k": 3}'
```

#### 4.3.2 搜索Agent配置

**Day 26-28: Agent配置**

```bash
# 1. 准备ASearcher数据
python examples/data_preprocess/asearcher.py \
    --input_json ~/data/asearcher.jsonl \
    --local_save_dir $DATA_ROOT/data/asearcher_uni_processed \
    --train_rows 100 \
    --test_rows 20

# 2. 创建适配4090的Agent配置
cat > examples/search_agent/agent_config_4090.yaml << 'EOF'
- name: search_agent
  _target_: uni_agent.agent_loop.UniAgentLoop
  concurrency: 8  # 降低并发
  log_dir: /tmp/search_agent_4090
  mask_abnormal_exit_traj: false
  
  interaction:
    action_timeout: 60
    max_turns: 16  # 减少最大轮次
  
  env:
    deployment:
      type: host  # 使用本地环境
    env_variables:
      RETRIEVAL_SERVICE_URL: "http://127.0.0.1:8001/retrieve"
      CRAWL_SERVICE_URL: "http://127.0.0.1:8001/crawl"
  
  tool_parser: hermes
  
  tools:
    - name: search
    - name: finish
  
  reward:
    name: search
EOF
```

#### 4.3.3 端到端测试

**Day 29-35: 运行验证**

```bash
# 创建4090适配的训练脚本
cat > examples/search_agent/train_4090.sh << 'EOF'
#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# 基础配置
NNODES_ROLLOUT=1
NNODES_TRAIN=1
NGPUS_PER_NODE=8

# 路径配置
DATA_ROOT=${DATA_ROOT:-${HOME}/uni_agent_data}
WORKING_DIR=${PWD}

# 模型配置（使用小模型）
model_path=$DATA_ROOT/model/Qwen2.5-3B-Instruct
max_prompt_length=2048
max_response_length=8192

# 算法配置
adv_estimator=grpo
loss_mode=gspo
rollout_n=4  # 减少采样数

# 并行配置（4090适配）
TP_SIZE=2  # 张量并行
train_tp=2
infer_tp=2
offload=True  # 启用offload

# 训练配置
train_prompt_bsz=0
total_rollout_steps=1000
staleness_threshold=1.0
trigger_parameter_sync_step=4
partial_rollout=True

ray job submit --no-wait \
    --runtime-env examples/search_agent/runtime_env.yaml \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    algorithm.adv_estimator=$adv_estimator \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$train_tp \
    actor_rollout_ref.actor.megatron.param_offload=$offload \
    actor_rollout_ref.actor.megatron.grad_offload=$offload \
    actor_rollout_ref.actor.megatron.optimizer_offload=$offload \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/search_agent/agent_config_4090.yaml \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$NGPUS_PER_NODE \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.partial_rollout=$partial_rollout
EOF

chmod +x examples/search_agent/train_4090.sh
```

---

### 阶段4: RL训练复现（Week 6-7）

#### 4.4.1 训练环境配置

**Day 36-40: 训练脚本**

```bash
# 创建4090适配的训练脚本
cat > examples/agent_train/train_4090_swe.sh << 'EOF'
#!/usr/bin/env bash
set -xeuo pipefail

project_name='Uni-Agent-4090'
exp_name='GRPO-Qwen2.5-3B-4090'

# 路径配置
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_small.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_small.parquet"}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"examples/agent_interaction/agent_config_local.yaml"}

# 算法参数
adv_estimator=grpo
loss_mode=gspo
use_kl_in_reward=False
kl_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# 响应长度（4090适配）
max_prompt_length=2048
max_response_length=8192

# 性能参数
use_dynamic_bsz=True
offload=True
gen_tp=2
train_tp=2

# 全异步参数
rollout_mode="async"
train_prompt_bsz=0
n_resp_per_prompt=4
total_rollout_steps=500
test_freq=5
staleness_threshold=1.0
trigger_parameter_sync_step=4
partial_rollout=True

ray job submit --no-wait \
    --runtime-env $RUNTIME_ENV \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=8 \
    rollout.total_rollout_steps=${total_rollout_steps} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.partial_rollout=${partial_rollout}
EOF
```

#### 4.4.2 数据准备

**Day 41-42: 数据处理**

```bash
# 下载并处理小规模数据
python examples/data_preprocess/r2e_gym_subset_filtered.py \
    --local-save-dir $RAY_DATA_HOME/data/swe_agent \
    --max-samples 100  # 小规模测试

python examples/data_preprocess/swe_bench_verified.py \
    --local-save-dir $RAY_DATA_HOME/data/swe_agent \
    --max-samples 20
```

#### 4.4.3 训练运行

**Day 43-49: 训练执行**

```bash
# 启动训练
bash examples/agent_train/train_4090_swe.sh

# 监控训练
# 查看Ray Dashboard
ray dashboard

# 查看Wandb日志
wandb login
# 访问 https://wandb.ai
```

---

## 5. 配置适配方案

### 5.1 关键配置对比

| 参数 | 原配置（A100） | 4090适配 | 原因 |
|------|----------------|----------|------|
| `max_response_length` | 65536 | 8192 | 显存限制 |
| `n_resp_per_prompt` | 8 | 4 | 减少并发 |
| `num_workers` | 8 | 4 | 减少worker数 |
| `tensor_model_parallel_size` | 4 | 2 | 显存限制 |
| `gpu_memory_utilization` | 0.8 | 0.7 | 预留显存 |
| `offload` | False | True | 显存不足 |
| `concurrency` | 512 | 8 | 降低并发 |

### 5.2 显存优化技巧

```python
# 1. 梯度检查点（Recompute）
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=2

# 2. 优化器Offload
actor_rollout_ref.actor.megatron.param_offload=True
actor_rollout_ref.actor.megatron.grad_offload=True
actor_rollout_ref.actor.megatron.optimizer_offload=True

# 3. 动态批处理
actor_rollout_ref.actor.use_dynamic_bsz=True

# 4. 混合精度
# 默认使用bf16，4090支持
```

### 5.3 性能调优建议

```bash
# 1. 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_DISABLE_COMPILE_CACHE=1

# 2. 使用Flash Attention
pip install flash-attn --no-build-isolation

# 3. 监控GPU使用
watch -n 1 nvidia-smi
```

---

## 6. 常见问题与解决方案

### 6.1 显存不足（OOM）

**问题**：CUDA out of memory

**解决方案**：

```bash
# 1. 减小max_response_length
max_response_length=4096  # 从8192降到4096

# 2. 增加offload
offload=True

# 3. 减小batch size
train_prompt_mini_bsz=8  # 从16降到8

# 4. 使用更小的模型
model_path=~/models/Qwen2.5-1.5B-Instruct
```

### 6.2 NCCL通信错误

**问题**：NCCL error / timeout

**解决方案**：

```bash
# 1. 检查网络
export NCCL_DEBUG=INFO

# 2. 设置超时
export NCCL_TIMEOUT=1800

# 3. 使用gloo后端（fallback）
export NCCL_BACKEND=gloo
```

### 6.3 训练不收敛

**问题**：reward不上升

**解决方案**：

```python
# 1. 检查学习率
actor_lr=5e-7  # 降低学习率

# 2. 调整clip ratio
clip_ratio_low=0.1
clip_ratio_high=0.2

# 3. 增加rollout数量
n_resp_per_prompt=8

# 4. 检查reward函数
# 打印reward分布
print(f"Reward mean: {rewards.mean()}, std: {rewards.std()}")
```

### 6.4 沙箱环境问题

**问题**：容器启动失败

**解决方案**：

```bash
# 1. 使用本地环境替代
env:
  deployment:
    type: host

# 2. 检查Docker
docker ps
docker run --rm hello-world

# 3. 简化环境
env:
  deployment:
    type: local
```

---

## 7. 学习路径与里程碑

### 7.1 知识点清单

**阶段1完成后应掌握**：

- [ ] Agent交互循环的完整流程
- [ ] Model-Tool-Env三层架构设计
- [ ] Session状态机的工作原理
- [ ] 错误处理和超时机制
- [ ] Rollout Cache的数据结构

**阶段2完成后应掌握**：

- [ ] 模型推理引擎的使用（vLLM/SGLang）
- [ ] Tool Calling的格式和解析
- [ ] 本地沙箱的搭建和使用
- [ ] 基本的Agent交互测试

**阶段3完成后应掌握**：

- [ ] 检索增强生成（RAG）的实现
- [ ] 搜索Agent的完整流程
- [ ] LocalWiki服务的部署
- [ ] 多轮对话的状态管理

**阶段4完成后应掌握**：

- [ ] GRPO/GSPO算法的实现
- [ ] 全异步训练的配置
- [ ] 分布式训练的并行策略
- [ ] 训练监控和调优

### 7.2 产出物清单

| 阶段 | 产出物 | 格式 |
|------|--------|------|
| 阶段1 | 架构图 + 核心流程文档 | Markdown |
| 阶段2 | 推理测试报告 | Jupyter Notebook |
| 阶段3 | 搜索Agent演示 | 视频/GIF |
| 阶段4 | 训练实验报告 | Markdown + Wandb |
| 总结 | 完整复现报告 | Markdown |

### 7.3 验证标准

**M1 - 环境搭建完成**：
```bash
python -c "import uni_agent; print('OK')"
python -c "import verl; print('OK')"
nvidia-smi  # 显示8卡4090
```

**M2 - 代码理解完成**：
- 能画出Agent交互循环图
- 能解释Session状态转移
- 能说明Tool Calling流程

**M3 - 推理验证完成**：
- 成功运行Agent交互
- 模型能正确调用工具
- 输出符合预期

**M4 - 搜索Agent完成**：
- LocalWiki服务正常
- 搜索结果相关
- 端到端问答成功

**M5 - RL训练完成**：
- 训练启动无报错
- Reward曲线正常上升
- 模型能力提升

---

## 附录A: 4090与A100对比

| 特性 | RTX 4090 | A100 80GB |
|------|----------|-----------|
| 显存 | 24GB | 80GB |
| FP16算力 | 82 TFLOPS | 312 TFLOPS |
| NVLink | 无 | 600 GB/s |
| 价格 | ~¥12,000 | ~¥80,000 |
| 适用场景 | 推理/小规模训练 | 大规模训练 |
| 性价比 | 高 | 中 |

## 附录B: 推荐学习资源

1. **verl文档**: https://github.com/verl-project/verl
2. **vLLM文档**: https://docs.vllm.ai/
3. **SWE-Bench**: https://www.swebench.com/
4. **GRPO论文**: DeepSeekMath

## 附录C: 联系与支持

- 项目地址: /data/home/yizhou/uni-agent
- 问题反馈: 参考CONTRIBUTING.md

---

*最后更新: 2026-07-05*
*适用硬件: 8×RTX 4090 24GB*

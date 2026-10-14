# Uni-Agent 深挖笔记

> 核验时间：2026-10-14（PDT）。本文描述的是上游 `verl-project/uni-agent` 在该时间点 `main` 分支的状态，上游 HEAD 为 `31a499eb2e4b16bb5fdc609ddf444b9a3b5d11f3`（`[ci] refactor: add gpu ci (#198)`，2026-10-14）。文中星数、issue 数、提交数均为该时间点工具返回的实测值，不是估算。
>
> 本文件只放在 fork `jasonchen505/uni-agent` 的 `notes/learning` 分支，不进入 `main`，不作为上游 PR 内容。

## 1. 同步状态（核验结果）

| 项目 | 实测值 |
| --- | --- |
| 上游 | `verl-project/uni-agent`，默认分支 `main`，Apache-2.0，创建于 2026-01-22 |
| 上游星数 / Fork 数 | 666 / 133 |
| 上游 open 计数 | GitHub `open_issues_count` 为 57，其中 issue 20 个、PR 37 个（对 open 列表逐条分类计数得到） |
| 上游 HEAD | `31a499eb`，2026-10-14 08:13 UTC，最后更新同日 |
| 上游 Release | `v0.1.0`（2026-09-23），`v0.1.0rc1`（2026-08-26，pre-release） |
| Fork `main` | `2181f8a5`，2026-07-07，作者 `jasonchen505` |
| Fork 与上游关系 | `diverged`：落后上游 112 个提交，超前上游 1 个提交 |
| 超前的 1 个提交 | `2181f8a5 Add markdown files`，父提交为上游 2026-06-29 的 `79750f51`，新增 6 个个人笔记 md 文件 |
| 个人笔记文件（只在 fork main 上，绝不进上游 PR） | `8GPU_4090_Reproduction_Plan.md`、`LLM_Agent_Five_Capabilities_Preparation.md`、`LLM_Agent_Interview_Preparation.md`、`Learning_Incremental_Notes.md`、`Three_Projects_Comparative_Analysis.md`、`Three_Projects_Comparison_Analysis.md` |
| Fork 分支 | 写本文之前只有 `main` 一个分支；本文所在的 `notes/learning` 分支从 fork `main`（`2181f8a5`）建出，`main` 本身未改动 |
| 结论 | fork main 停留在 2026 年 6 月底的上游状态加 1 个个人提交，之后上游走了 112 个提交。任何未来的上游贡献分支必须从上游 `main` 建出，不能从 fork `main` 建出，否则 6 个个人笔记文件会混进 PR diff |

## 2. 一句话定位

Uni-Agent 是 verl 生态里训练长程智能体的框架：把任意已有的 agent harness 接进强化学习，用同一套交互栈完成推理、评测、轨迹采集、SFT/RL 训练。

它解决的核心矛盾是：长程 agent 的 rollout 不是单轮问答，而是几十到几百轮工具调用的有状态会话。普通 RL 训练管线按单轮 prompt 处理，会丢失会话、工具、沙箱、轨迹对齐这些东西。Uni-Agent 把这些补齐，同时保证训练用的 token 序列与推理时模型实际看到的完全一致。

官方定位的四个卖点，对应四个可验证的机制：

- **可插拔**：任何能把模型 endpoint 指向 Gateway 的 harness 都能接进 RL。
- **可扩展**：Agent、Task、Tool、Sandbox 各自独立定制，复用同一套评测和训练运行时。
- **可扩展到大规模**：1000 个以上长程有状态会话并发跑，每条轨迹、日志、奖励都绑定到正确的会话。
- **可复现**：recipes 带完整配置、评测设置、结果表和学习曲线。

## 3. 与 verl 生态的关系与分工

### 3.1 verl（`verl-project/verl`）是底座，Uni-Agent 是上层框架

仓库里 `verl/` 是一个 git submodule（`.gitmodules` 指向 `https://github.com/verl-project/verl.git`），安装时要先 `git submodule update --init --recursive` 再 `pip install --no-deps -e ./verl`。Uni-Agent 依赖 verl 提供但不重复实现的部分：

- 训练器：PPO/GRPO/GSPO 等算法、advantage 计算、loss 聚合、checkpoint、LR 调度。
- Rollout 引擎：vLLM/SGLang 的 LLMServer、权重同步、采样。
- TransferQueue：训练与 rollout 之间传轨迹的队列，是 Uni-Agent 向训练器交付数据的契约。
- `agent_loop_manager_class` 扩展点：verl 训练器通过这个扩展点把 rollout 交给 Uni-Agent 的 `AgentFrameworkRolloutAdapter`。
- `RewardLoopWorker` / `NaiveRewardManager`：奖励计算的另一条路径（见第 5 节）。
- `RLInsightLogger` 与 `RolloutTraceConfig`：训练可观测性的底座，Uni-Agent 只做业务 span 的转发（见第 4.8 节）。
- `agent_aware_router` 的接入点：`LLMServerManager` 侧通过 `get_router_handle` 接入 Uni-Agent 的 router。

一句话：verl 管“模型怎么训、token 怎么生成”，Uni-Agent 管“agent 会话怎么跑、轨迹怎么从会话里忠实地抽出来”。

### 3.2 verl-recipe 是配方集，不是框架

`verl-project/verl-recipe`（核验时 338 星，2026-10-13 更新）是一组基于 verl 的端到端 RL 训练示例/配方。它不提供 Gateway、会话管理、沙箱、任务抽象。Uni-Agent 则提供了这些基础设施，并且自己的 `examples/` 里也有可跑的训练配方。两者是“配方”与“框架”的关系，不存在功能重叠到需要二选一。

### 3.3 ProRL-Agent-Server 是同目标、不同栈的竞品

`NVIDIA-NeMo/ProRL-Agent-Server`（核验时 857 星，2026-10-14 更新）目标与 Uni-Agent 高度重合：在任意 harness 上做 agentic RL。但它是 NVIDIA NeMo 生态、server 形态的实现；Uni-Agent 是 verl 生态、Gateway + TransferQueue 形态的实现。选哪一边取决于团队已经在哪个训练栈上，而不是哪个功能列表更长。做贡献时不要把两个项目的接口假设互相套用。

### 3.4 与被包装的 harness 的关系

OpenHands、mini-swe-agent、Claude Code、OpenClaw、Codex、Hermes、DeepSeek harness、DeepEyes 这类项目是“被训练/被评测的对象”，Uni-Agent 是“把它们接进训练的管子”。Uni-Agent 的 black-box 路径（第 4.1 节）就是为这类 harness 存在的：harness 自己拥有 loop 和工具，Uni-Agent 只负责给它一个 Gateway endpoint、一个沙箱、一次评分。

Harbor（`laude-institute/harbor`）则是任务格式：Uni-Agent 把 Harbor 任务当作一种 Task 接入，跑同一套推理管线。

## 4. 架构与数据流

### 4.1 总体结构

顶层目录（上游 main）：

```text
uni_agent/      # 框架本体
verl/           # submodule，verl 底座
examples/       # 推理、训练、black-box recipe、quickstart
docs/           # mkdocs 文档，站点 uni-agent.readthedocs.io
tests/          # tests/uni_agent/*，按模块分目录；另有 tests/special_sanity
assets/         # README 配图
```

`uni_agent/` 下的分层（CONTRIBUTING 明确规定了归属原则，见第 8 节）：

| 层 | 目录 | 归属 |
| --- | --- | --- |
| Agent | `uni_agent/agents/` | 解决策略，拥有或启动 agent loop |
| Tool / Toolbox | `uni_agent/tools/` | 一个模型可见的动作；Toolbox 把 Tool 绑定到 Sandbox |
| Task | `uni_agent/tasks/` | episode 生命周期、样本元数据、配置、奖励、验证 |
| Sandbox | `uni_agent/sandbox/` | 执行环境、生命周期、文件系统、命令数据面 |
| Gateway | `uni_agent/gateway/` | 模型协议适配、会话路由、token 级轨迹捕获 |
| Framework | `uni_agent/framework/` | rollout 编排、失败隔离、训练记录交付 |
| Logging | `uni_agent/logging/` | 共享上下文、handler、产物、敏感数据脱敏 |
| Agent-Aware Router | `uni_agent/agent_aware_router/` | 推理侧负载均衡与 KV 感知路由 |
| RL-Insight | `uni_agent/rl_insight/` | 向 verl RL-Insight 转发业务 span 的唯一适配层 |

全项目用注册表模式组织扩展点：Agent、Task、Sandbox、Tool 各自有 `registry.py`，用 `@register_*` 装饰器注册，`AGENT_MODULES` / `TASK_MODULES` / `SANDBOX_MODULES` 做懒加载。重复注册同名但不同类会直接抛错。懒加载的意义是：某个 agent 的重型依赖没装，不会拖垮整个包的 import，只有真正选中它时才 import。

端到端数据流（verl 托管的训练/推理路径）：

```text
verl LLMServerManager
  -> AgentFrameworkRolloutAdapter（agent_loop_manager_class 扩展点）
  -> Uni-Agent Gateway 会话（每个 rollout 样本一个会话）
  -> Agent Runner（run_task 等）
  -> Task / Agent / Sandbox（Agent 跑 loop，调工具，改沙箱）
  -> Gateway 终结会话，产出 token 级轨迹
  -> Framework 做奖励评分、轨迹后处理、写 TransferQueue
  -> verl 训练器消费 TransferQueue 做 PPO/GRPO/GSPO 更新
```

外部 API 推理路径不走 Gateway：`Task -> Agent -> 外部模型 API`。这条路径能评测，但不产出 Gateway 的 token 级训练轨迹。需要“推理与训练一致”或需要轨迹当训练数据时，必须走 Gateway 路径。

### 4.2 Agent：谁来解题，怎么跑 loop

契约（`uni_agent/agents/base.py`）：

```python
async def run(*, sandbox: Sandbox, messages: list[dict], workdir: str | None = None) -> AgentResult
```

`AgentResult` 有四个字段：`output`（最终结构化输出）、`transcript`（白盒 loop 产生的消息与工具观测）、`info`（实现相关的元数据，如 token 数、进程状态）、`finished`（三态：`True`/`False` 为已知完成状态，`None` 表示 Agent 不上报完成状态；不上报不等于不可训练）。

关键规则：Task 已经启动了 Sandbox，Agent 必须只用、不关闭。`workdir` 可选，指定 Agent 在沙箱内的工作目录。

配置基类 `AgentConfig` 只有 `name` 和 `model`。`ModelConfig` 是 `base_url` / `api_key` / `model_name` 三元组，正常由 runner 在运行时注入，Task YAML 里可以省略。白盒 Agent 还有 `sampling_params_override`（`temperature` / `top_p` / `top_k` / `max_tokens_per_turn`），省略时用 endpoint 或 Gateway 会话默认值；Gateway 路径下请求级 `temperature` / `top_p` / `top_k` 覆盖需要显式的请求 key 许可。

两种集成风格：

**白盒 Agent**：Uni-Agent 拥有交互 loop、模型调用、Tool 分发、transcript。

- `react`（`uni_agent/agents/react/`）：自己实现的 ReAct loop。按配置建 Toolbox，把消息和工具 schema 发给 OpenAI 兼容 endpoint，执行返回的工具调用，把观测追加进 transcript，直到调了 `submit`/`finish`、token 上限、超时预算或 `max_steps`。默认工具是 `str_replace_editor`、`stateful_shell`、`submit`，默认 `max_steps` 50，`timeout_budget` 3。终止原因会记在轨迹信息里（`max_steps`、`timeout` 等）。
- `mem_agent`（`uni_agent/agents/mem_agent/`）：长上下文记忆 Agent。把长上下文按 token 上限切块，每块开全新的模型上下文，块之间只带一个紧凑记忆，最后用记忆回答问题。配套 recipe 训 Qwen3-4B，在 HotpotQA 32K 训练集上训，在 8K 到 1M 八个上下文长度上评。

**黑盒 Agent**：外部 harness 拥有 loop，跑在沙箱内的进程里。

- `claude_code`：把 `ModelConfig` 转成 harness 需要的环境变量（`ANTHROPIC_BASE_URL` 指向直连 endpoint 或会话级 Gateway endpoint），用 `sandbox.exec()` 启动 `claude -p`。进程退出码 0 时 `AgentResult.finished=true`。约束很严：恰好一条非空 user 消息，至多一条 system 消息；system 文本走 `--system-prompt`，与 `extra_args` 里的 `--system-prompt` / `--system-prompt-file` 冲突会报错。
- `mini_swe_agent`：同理，mini-swe-agent harness 跑在沙箱里。
- `openclaw`：OpenClaw harness 的黑盒接入，带 SWE-bench 推理 recipe 和可选的在线 RL / OPD / Combine 训练路径。

已注册的 Agent 只有这 5 个（`react`、`claude_code`、`mini_swe_agent`、`openclaw`、`mem_agent`）。新 Agent 就是实现 `run` 加注册，不需要改框架。

### 4.3 Tool / Toolbox：白盒专用的宿主侧工具

只服务白盒 Agent。黑盒 harness 自己在沙箱里带工具，不用这一层。

- `Tool` = 一个模型可见的动作：`name`（模型看到的函数名）、`description`、`args_model`（Pydantic 参数模型）、`config_model`（构造期配置）、`run()`（异步实现，返回 `ToolResult`）、`start()` / `close()`（有状态资源的生命周期）。
- `ToolResult` = `text`（模型看到的文本）+ `status`（`ok` / `format_error` / `error` / `timeout`）。`to_observation()` 把结果转成下一轮观测，超长输出截断并标注被删掉的字符数，提示模型用 `head`/`tail`/`grep` 或重定向而不是交互式 pager。
- `Toolbox` 把一批 Tool 绑定到同一个 Sandbox。`Toolbox.from_specs([...], sandbox=...)` 按配置字典建工具，`name` 选注册表项，其余字段由该工具的 `config_model` 校验。`async with toolbox.entered(...)` 启动全部工具，循环里 `toolbox.call(name, args)` 分发，退出时逐个关闭，某个工具关闭失败不会阻止其他工具释放。
- 错误语义分两层：调用格式错误（未知函数、参数解不出 JSON 对象）抛 `ToolCallFormatError`，在调用前拦截，返回 `Invalid action: ...` 观测让策略自纠；运行时错误（路径非法、命令被拒）抛 `ToolError`，转成 `Error: ...` 观测，不炸掉整个 rollout。真正的 bug 不吞，继续向上抛。

内置工具只有 4 个：

| 配置名 | 模型看到的名字 | 行为 |
| --- | --- | --- |
| `stateful_shell` | `shell` | 有状态 shell，底层是 tmux 通道，cwd、export、后台任务跨调用保留 |
| `str_replace_editor` | 同名 | 文件编辑，undo 历史存在宿主侧，文件本身在沙箱里 |
| `submit` | 同名 | 无副作用的控制工具，ReAct loop 看到策略调它就结束 episode |
| `finish` | 同名 | 同 `submit`，ReAct 的 `_FINISH_TOOLS` 集合包含两者 |

关键点：工具跑在宿主侧，命令和文件操作穿过 Sandbox 数据面，工具代码本身不进任务镜像。配置名与模型名可以分离（如 `stateful_shell` 对 `shell`），实现名保持稳定，模型看到的名字保持简短。

### 4.4 Task：episode 的拥有者

Task 是推理和训练的顶层执行单位，把一个样本的 prompt、元数据与 Agent、Sandbox、奖励逻辑组合起来。Task 拥有完整 episode：

```text
开始日志 -> 启动沙箱 -> 运行 Agent -> 评估沙箱状态 -> 返回 TaskResult -> 停止沙箱
```

`TaskConfig`（`uni_agent/tasks/base.py`）只有全任务共有的字段：`name`（注册表 key）、`sandbox`（`SandboxConfig`）、`agent`（具体 `AgentConfig` 子类）、`prompt`（输入时是 agent 无关的数据集消息，模板渲染后是给 Agent 的消息）、`prompt_template`（可选的 recipe 自有文本消息，带任务元数据占位符）、`metadata`（样本专属数据，供执行和评分用）。`extra="forbid"`，未知字段直接拒。`agent` 字段是多态的：传 `{"name", ...}` 映射时按 `name` 经 agent 注册表解析成对应的 `AgentConfig` 子类，`SerializeAsAny` 保证 dump 往返不丢子类字段。

`prompt_template` 的渲染规则是刻意收紧的：只支持 Python 标准库花括号解析替换 `metadata` 的直接字段，占位符必须是简单标识符，属性访问、索引、`!r` 转换、格式规格（如 `:>10`）一律拒绝；缺失字段、畸形模板、非文本替换值、非字符串模板内容都在 Agent 启动前校验失败。多模态模板渲染被明确推迟。没有 `prompt_template` 时数据集消息原样透传，这也是自渲染 Agent（自己在沙箱里套模板）的预期路径。

`Task` 实现 `run()`，不带参数，因为全部样本状态都在 config 上。`TaskResult` 是 `reward` / `accuracy` / `finished` / `extra_info`。`Task` 内部的评分逻辑在各任务的 `reward.py` 里，预处理在 `preprocess.py` 里（把数据集行转成 Task Config 格式的 parquet）。

已注册的 Task（`uni_agent/tasks/registry.py` 的 `TASK_MODULES`）：

| Task | 领域 | 说明 |
| --- | --- | --- |
| `swe_bench` | 软件工程 | SWE-Bench Verified，验证集 |
| `swe_bench_multilingual` | 软件工程 | SWE-Bench 多语言 |
| `swe_bench_pro` | 软件工程 | SWE-Bench Pro，26Q4 路线图里要加强的方向 |
| `swe_rebench` | 软件工程 | SWE-reBench，训练集主力（过滤后 1150 条） |
| `hotpotqa` | 问答/长上下文 | MemAgent 的训练/评测任务 |
| `terminal_bench` | 终端任务 | Terminal-Bench |
| `npu_triton_kernel` | 内核生成 | `kernel_bench` 目录实现，含工作区快照与验证快照 |
| `harbor` | 通用任务格式 | Harbor 任务格式接入 |

Task Config YAML 文件可以包含多个任务名的条目，整文件一起解析建索引；非法 YAML、无 `name` 条目、重名都在加载时失败。解析器只合并与样本任务名匹配的那一条，没用到的条目不会被渲染或传给 Agent。

Task Config 的解析顺序（推理指南）：YAML 默认值（按任务名匹配）→ 样本 Task Config 深度合并覆盖 → 运行时模型 endpoint 最后注入为最终覆盖。

### 4.5 Sandbox：执行边界

Sandbox 是 agent episode 的执行边界，拥有运行时生命周期，提供命令、文件、上传、下载、可选端口暴露的统一数据面。Task 和 Agent 拿到的是完整 `Sandbox` 对象，但启动和停止是 Task 的责任。工具只依赖窄的 `SandboxBackend` 协议。

`SandboxConfig` 字段：`provider`（注册的后端名）、`image`（Docker/Modal 等镜像后端用的容器镜像）、`image_map`（可选，数据集镜像名到集群能拉的镜像名的重写）、`image_mounts`（可选，运行时镜像挂进任务沙箱）、`executable_paths`（命令名到挂载镜像提供可执行文件的映射）、`runtime_timeout`（沙箱最大存活时间）、`sandbox_kwargs`（provider 专属构造参数）。未知字段拒绝，provider 专属选项必须放进 `sandbox_kwargs`。

已注册的 provider（`uni_agent/sandbox/registry.py`）：

| Provider | 说明 |
| --- | --- |
| `local` | 本地执行 |
| `docker` | Docker 容器，支持 `image_mounts` |
| `e2b` | E2B 沙箱（2026-10-14 当天合入的 #261 新增，带 API key、模板、镜像、网络配置说明） |
| `modal` | Modal serverless 云，支持 `image_mounts` |
| `vefaas` | 火山引擎 veFaaS serverless，大规模并行时主力 |
| `openyuanrong` | OpenYuanRong，支持 `image_mounts` 与 Gateway 反向隧道 |

`image_map` 用 glob 规则重写镜像名，`**` 捕获实例专属路径段，第一个匹配的规则生效。例如 `swebench/**:latest` 到 `<registry>/swe-bench-verified/**:v2` 会把 `swebench/sweb.eval.x86_64.astropy_1776_astropy-13033` 重写为对应路径。注意 `image_map` 只重写任务 `image`，挂载镜像引用必须本身就是完整可拉的；`executable_paths` 在 Local 和 veFaaS 上被拒绝。

生命周期：`async with sandbox:` 进入时带重试、超时、全局启动并发控制地启动后端，退出时即使 episode 抛错也调 `stop()`。默认 `SANDBOX_STARTUP_TIMEOUT` 600 秒、`SANDBOX_STARTUP_CONCURRENCY` 64、`SANDBOX_STOP_TIMEOUT` 120 秒。远程 provider 的 `is_alive()` 应该是非抛错的健康探针。

### 4.6 Gateway：把模型 API 翻译成训练轨迹

Gateway 是整个项目最核心的创新点。它不是推理引擎（vLLM/SGLang 才是生成 token 的地方），它拥有会话路由、协议转换、轨迹物化三件事。

**部署形态**：`GatewayActor` 是 Ray actor（`uni_agent/gateway/gateway.py`），内含 FastAPI 路由、provider 能力门控、每会话的 `GatewaySession` 实例。`GatewayManager`（`uni_agent/gateway/manager.py`）在驱动侧拥有 Gateway actor 池：按 CPU 节点 round-robin 放置 actor（避免全挤在驱动节点），记录每个会话归哪个 actor，用最少会话数选 actor 建新会话，生命周期调用转发到正确的 actor。

**协议适配**：`uni_agent/gateway/adapters/` 有 `openai.py` 和 `anthropic.py` 两个适配器。它们把线上的 OpenAI Chat Completions / Anthropic Messages 请求降成内部的 `InternalGenerationRequest`（消息、工具、采样参数），再把响应/SSE 信封编码回对应的协议。另有 Responses API 与 Continuous Token 归一化（PR #224）。默认只允许请求级 `max_tokens` / `stop` 两个采样 key，其余需要配置显式放行。

**会话生命周期**（`docs/source/concepts/gateway-and-trajectories.md`）：

1. Framework 选一个 Gateway actor，建会话。
2. 拿到会话专属的模型 `base_url`。
3. 启动配置好的 Agent Runner（如 `run_task`）。
4. Runner 把会话 endpoint 注入 `agent.model`。
5. Agent 向会话 URL 发 OpenAI / Anthropic 请求：`POST /sessions/{session_id}/v1/chat/completions`、`POST /sessions/{session_id}/v1/messages`。
6. Gateway 把 token 化后的请求转发给 verl rollout 引擎。
7. Runner 返回 `TaskResult`（或不带 episode 标注时返回 `None`）。
8. Framework 终结会话，附加奖励/状态/指标，把轨迹写进 TransferQueue。

会话在 Gateway 内存里一直保留到终结或中止。

**轨迹（`Trajectory`，`uni_agent/gateway/session/types.py`）**：终结时产出的 token 级训练轨迹，字段包括：

- `prompt_ids`：种子 prompt 的 token ID。
- `response_ids`：模型生成的 token 加多轮继续时的间隙 prompt token。
- `response_mask`：与 `response_ids` 对齐，`1` 为模型输出，`0` 为工具结果等继续上下文 token。这是训练时 loss mask 的依据，错了就等于拿工具输出当模型输出训。
- `response_logprobs`：可选，与 `response_ids` 对齐的 log prob，继续上下文 token 用 `0.0`。
- `finished`：三态 episode 完成事实，由 Framework 在 Runner 返回后附加。
- `reward_score` / `reward_metrics`：Framework 从 Runner 或 RewardLoopWorker 附加的标量奖励与验证指标。
- `num_turns`、`chain_id`、`routed_experts`、`multi_modal_data`、`extra_fields`（轨迹物化元数据、权重版本跨度 `min_global_steps` / `max_global_steps` 等）。

**MessageCodec**（`uni_agent/gateway/session/codec.py`）：负责把消息编成 token、做工具调用解析、多模态合并、Continuous Token 上下文合并。这是容易出 bug 的地方，open issue #232 就是它在多轮追加图片时抛 `Continuous Token context merging does not currently support incremental image or video data` 导致 HTTP 500。

**会话治理**：近期 PR 补了一批能力：显式会话准入反压（#240）、会话与轨迹在路由清理时保留（#241）、模型请求与会话准入指标暴露（#242）、会话路由绑定模型版本（#244）、生成完成后再中止会话归属（#256）、容量 token 感知的冷启动移除与副本重绑定（#257）、流式增量跳过（#260）。

### 4.7 Framework：rollout 编排与训练记录交付

`AgentFramework` 是抽象基类（`uni_agent/framework/base.py`），只有两个契约：类方法 `from_config` 与 `generate_sequences(prompts: TensorDict)`（跑 agent 会话，把终结轨迹写进 TransferQueue）。

`GatewayAgentFramework`（`uni_agent/framework/framework.py`）是当前实现：

- `AgentRunner` 协议：一个异步可调用对象，执行一个 agent episode，签名是 `(*, session: SessionHandle, raw_prompt, sample_index, **sample_runner_kwargs) -> TaskResult | None`。函数 runner 与带异步 `__call__` 的类 runner 都接受，不要求继承 ABC。
- Runner 注册：配置里的 `agent_runners` 映射 runner 名到 `runner_fqn` / `runner_kwargs` / `dispatch_mode` / `max_concurrent_sessions` / `trajectory_selection` / `session_timeout_seconds`。`dispatch_mode` 是 `inline_async`（进程内异步跑）或 `ray_task`（每个会话用 Ray task 跑阻塞 runner）。`trajectory_selection` 是 `all`（保留全部轨迹）或 `longest`（只留最长）。
- `AgentFrameworkRolloutAdapter`（`uni_agent/framework/entry.py`）：满足训练器的 `agent_loop_manager_class` 扩展点契约。Recipe 在 YAML 里配一行就能接入，不用每个 recipe 写 Python 胶水：
  ```yaml
  actor_rollout_ref:
    rollout:
      agent:
        agent_loop_manager_class: uni_agent.framework.entry.AgentFrameworkRolloutAdapter
  ```
  `build_gateway_manager` 负责 Gateway 通用的驱动侧连线，`build_agent_framework` 负责建 Framework。RL 正确性边界留在父进程/Gateway 路径：token 真实性、成功提交的会话状态、奖励评分、TransferQueue 写、终结、中止都在父进程；Ray worker 只 import 并跑用户 runner，成功或异常都回传给父进程。
- `generate_sequences` 流程：按 `uid` 分批跑 rollout（`asyncio.gather`，单个失败不炸整批），每个 episode 经过会话创建、Runner 执行（带并发信号量与超时取消）、轨迹选择、轨迹后处理器、奖励评分、轨迹日志、TransferQueue 写。失败时写 `failure` 状态标签，成功写 `finished`。
- 奖励来源二选一：Runner 自带奖励保留；Runner 没给时走 `RewardLoopWorker`（`_trajectory_to_reward_dataproto` 把轨迹包成单样本 DataProto 调 `compute_score`）。注意 Worker 之后还可能做自己的后处理（如 DAPO 的 overlong 惩罚），Framework 的适配不是绕过 Worker。
- 轨迹后处理器：可选的同步/异步可调用对象，在 Gateway 终结之后、奖励评分之前对轨迹做过滤、重排、替换。顺序是 Gateway 终结 → Runner 的 `trajectory_selection` → 后处理器 → 奖励评分 → 轨迹日志与 TransferQueue。配置是 `actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_fqn` 加 `trajectory_postprocessor_kwargs`。后处理器必须保持终结的 `finished` / `reward_score` / `reward_metrics` 与 token 数组对齐，返回空列表表示过滤掉该会话。FQN 在 Framework 启动时 import，失败会让初始化失败。
- `mask_unfinished_episode`：可选布尔配置，未完成 episode 的 loss mask 置零，必须是严格 bool。
- 多模态后处理：`multi_modal_postprocess.py` 计算多模态输入与 position ID。
- 失败隔离：Runner 任务取消时先优雅取消（给 worker 的 `asyncio.run` 抛错的机会），超时后强制取消；取消后用 `shield` 保证 `abort_session` 仍执行，避免会话泄漏。

`task_runner.py` 是 Framework 到 Task 的桥：`run_task` 负责把 Gateway 会话的 endpoint 注入 Task，把 Task 的 `Agent.run` 跑起来，把 `TaskResult` 拿回来。它还处理 openyuanrong 沙箱的 Gateway 反向隧道：把 `base_url` 重写为沙箱内的 `127.0.0.1:<proxy_port>`，把 `upstream`（Gateway host:port）注入沙箱配置。反向隧道只支持 openyuanrong provider，在其他 provider 上配 `proxy_port` 会直接报错，而不是静默忽略。

### 4.8 Agent-Aware Router：为多轮 rollout 做的负载均衡

`uni_agent/agent_aware_router/` 是 verl 推理侧的插件式负载均衡器，注册名 `kvcaware`，在 `LLMServerManager` 侧通过 `get_router_handle` 接入。

动机来自 agentic RL rollout 的工作负载形状，与单轮 RL rollout 完全不同：

- 一个样本对应一个 agent 会话，一个会话可能有几百次工具调用，单次 prefill 经常几万 token，回合数分布极不均匀。
- 同一会话内多轮请求共享不断增长的前缀历史。如果每轮都能命中上一轮留在副本上的 KV，就能跳过大量 prefill 重算；否则每轮重算，prefill 成本线性累积。
- 会话长、KV 工作集大，单个会话的前缀 KV 能占掉副本一半以上的 KV 容量，引擎会驱逐旧前缀块，下一轮 miss，又重算，形成“填充 → 驱逐 → miss → 重算”的恶性循环。
- 平均请求上下文大且分布宽，batch 内与副本间 GPU 气泡多，MFU 被拖低。

设计目标：前缀命中优先（后续回合尽量回原副本命中保留的 KV）、只在真正过载时迁移、减少 GPU 气泡。

结构：

```text
agent_aware_router/
  balancer.py                  # 纯外壳：生命周期 + route() 委托
  strategies/                  # 评分算法，kvc_aware.py 为主策略，routing.py 做加权排名
  store/                       # 单例存储：data_store（指标 + sticky 绑定 + 活跃会话）、kv_cache_store（前缀哈希链 + 三层命中率）、per_replica_store / per_request_store
  collectors/                  # 采集 vLLM 信号：collector.py / provider.py 管生命周期，parse/vllm/metrics.py 做 /metrics 轮询，parse/vllm/kv.py 与 kv_event.py 处理 kv-events zmq 增量块状态，transport/{http,zmq,callback}.py
  config/                      # KVCAwareConfig + StrategyConfig
  types/                       # SlowCut / OverloadMode / Layer / MetricKey 等枚举
  insight/emitter.py           # 推给 rl-insight，再到 Prometheus / Grafana
```

路由决策：每次 `acquire_server(request_id, prompt_ids)` 调 `Balancer.route()`，对每个策略调 `score()` 做加权排名，取 `ranking[0]`。`KVCacheAwareStrategy.score()` 的决策树：

1. STICKY 短路：`request_id` 在且存在 sticky 绑定的副本，未过载则直接命中，返回 sticky 最高分，保住前缀局部性；过载则回退。
2. SLOW-CUT 回退评分：`CAPACITY_TOKEN_AWARE` 模式下先做纯容量硬门控，再按剩余容量排名：
   ```text
   avail[i]     = cap × (1 - kv_cache_usage_perc[i])
   need[i]      = len(prompt_ids) × (1 - gpu_hit[i])
   remaining[i] = avail[i] - need[i]
   eligible[i]  = avail[i] >= cap × (1 - capacity_reserve_threshold)
   pick         = argmax(eligible, remaining)
   ```
   前缀命中率通过 `need` 项参与：命中率越高，需要重算的 token 越少，`remaining` 越大，越容易赢。负载均衡先由 `eligible` 硬门控过滤掉满的副本，再由 `avail` 项参与排名。

### 4.9 RL-Insight：只做转发，不定义协议

`uni_agent/rl_insight/adapter.py` 是 Uni-Agent 里唯一知道怎么把完成的 span 归一化并转发给 verl 的模块。Uni-Agent 不定义 lane ID、私有 dashboard 指标、span 协议，Agent Loop Trajectory dashboard 保持框架无关。

被插桩的阶段：

| 阶段 | 位置 | 发出的数据 |
| --- | --- | --- |
| Worker 初始化 | `uni_agent/framework/entry.py` | 用 trainer 的 project/experiment 名初始化 verl 的 `RolloutTraceConfig` |
| 会话开始 | `uni_agent/framework/framework.py` | 建一个 agent-loop 会话，把不可变身份挂到 gateway 元数据与任务参数上 |
| 任务执行 | `uni_agent/framework/task_runner.py` | 一个 `agent_task` span：任务名、沙箱镜像、prompt 哈希、奖励、精度、完成态、奖励投递状态 |
| 模型生成 | `uni_agent/gateway/session/session.py` | 每次 gateway 调用一个 `gateway_generation` span：链/轨迹、回合、token 数、结束原因、内容、工具、错误 |
| 会话结束 | `uni_agent/framework/framework.py` | 发布轨迹摘要与 `agent_session` span（成功/空/失败三种结局） |

链路是 `uni_agent.rl_insight.adapter -> verl RLInsightLogger -> rl_insight API -> Tempo traces + Prometheus gauges -> Grafana`。硬要求只有三条：每个 agent 会话恰好一个会话对象、身份字段一致、恰好一次 `finish` 调用。任务与生成 span 的数量和顺序跟 agent 业务逻辑走。

### 4.10 Logging

基于标准 `logging`，不需要显式配置调用，框架首次使用时自动接线，独立组件（如 router）在 import 时自举。日志级别由 `UNI_AGENT_LOG_LEVEL` 环境变量决定（`CRITICAL` / `ERROR` / `WARNING` / `INFO` / `DEBUG`，非法值回落 `INFO`），只作用于 `uni_agent.*` 的 logger，不读不写宿主进程的 root 日志配置。

`sample_logging` 上下文管理器把一个 log ID 绑定到块内所有记录，可选写到该 run 的日志文件。`uni_agent/logging/redaction.py` 负责敏感数据脱敏，`session.py` 负责会话级日志。PR 描述里不要带凭证、私有模型响应、专有数据或未脱敏日志。

### 4.11 顶层还有什么

- `examples/`：`quickstart/`（inference / training / sandbox / harbor / oracle / agent_aware_router）、`inference/`（`parallel_infer_api.py`、`parallel_infer_verl.py`、`parallel_run_oracle.py`）、`blackbox_recipes/`（OpenClaw 等黑盒 recipe）、`claude_code_swe_task` / `claude_code_kernel_task` / `mini_swe_agent` / `mem_agent` / `agent_aware_router` / `gateway`（debug launcher）。
- `tests/`：`tests/uni_agent/` 按模块分目录（`agents`、`framework`、`gateway`、`agent_aware_router`、`tasks`、`tools`、`sandbox`、`logging`、`rl_insight`），另有 `tests/special_sanity/check_pr_title.py`（PR 标题格式检查）。CI 分 `ci-cpu.yml`（`pytest tests/uni_agent/ -m="cpu and level0"`）、`ci-gpu.yml`（本次核验时上游 HEAD 就是加 GPU CI 的 #198）、`ci.yml`、`docs.yml`、`pre-commit.yml`、`secrets-scan.yml`、`check-pr-title.yml`。pytest marker 有 `cpu` / `gpu` / `npu` 与 `level0`（CI 必跑）/ `level1`（完整测试，不进 CI）。
- `docs/`：mkdocs，`docs/source/` 下 `quickstart/`、`concepts/`、`benchmark/`。README 里原有的 dashboard 章节在上游 main 已不存在，根目录也没有 `dashboard/` 目录，只有 `assets/dashboard.png` 一张配图残留。
- `pyproject.toml`：包名 `uni-agent`，版本动态取 `uni_agent/version/version`（核验时为 `0.1.0.dev`），`requires-python >=3.10,<3.13`，ruff 行宽 120，mypy 默认 `ignore_errors = true`，只对 verl trainer 的几个模块开严格检查。

## 5. 训练与评测流程

### 5.1 训练

训练 = verl 训练器 + Uni-Agent Framework。Recipe 以 shell 脚本 + YAML 任务配置的形式放在 `examples/` 下。

标准训练路径（以 `examples/quickstart/training/train_qwen3p5_dense.sh` 为例，Qwen3.5-4B dense ReAct GRPO，SWE-reBench，colocate_async，64 H100）：

1. 准备数据：`python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/uni_agent` 写出 `swe_rebench_filtered.parquet`（过滤后 1150 条）；`python -m uni_agent.tasks.swe_bench.preprocess` 写出 `swe_bench_verified.parquet`（验证集）。处理后的行与运行时沙箱 provider 和 agent 协议无关，每行含一条数据集/source 用户消息、任务元数据、规范镜像引用、每样本 Task Config。
2. 任务配置：YAML 里给 ReAct 或 Claude Code 各写一套 Task Config（见第 4.4 节的合并顺序）。ReAct 版配 `max_steps: 200`、工具列表、`sampling_params_override: {temperature: 1.0, top_p: 1.0}`；Claude Code 版配 `max_turns: 200`、`run_timeout: 4800`。
3. Ray 运行时环境：用 `runtime_env.yaml` 分发仓库、暴露 `verl` 源码、装轻量任务与沙箱依赖、把凭证传给每个 Agent runner。必须从仓库根目录启动，让 Ray 把 `verl/` 和 `uni_agent/` 都打包进去。
4. 启动训练脚本：脚本里用环境变量暴露全部关键超参。Qwen3.5-4B colocate_async 版的默认值：`GATEWAY_COUNT=32`、`CONCURRENCY=1024`（在途 rollout 会话上限）、`max_prompt_length=8K`、`max_response_length=128K`、`adv_estimator=grpo`、`clip_ratio_low=0.2` / `high=0.28` / `c=10`、`kl_loss_coef=1e-3`、`use_kl_in_reward=False`、`loss_agg_mode=seq-mean-token-mean`、`lr=1e-6`、`warmup=10`、`train_prompt_bsz=64`、`n_resp_per_prompt=8`、`ppo_mini_bsz=16`、`num_warmup_batches=2`、`save_freq=10`、温度 `1.0`/`1.0`、验证温度 `1.0`/`0.95`、`use_dynamic_bsz=True`、`offload=True`、`gpu_memory_utilization=0.8`、`MTP` 推测解码开（2 个推测 token，权重同步不带 MTP 层，rollout 引擎从真实 checkpoint 加载）、TIS（`rollout_is=token`，阈值 2.0）。
5. 拓扑：V1 `colocate_async` 是 actor 与 rollout 共享全部 GPU（训练步时 rollout 副本休眠），`NNODES` 即总节点数。另有 fully async 配方（actor 与 rollout 分离）。

算法：GRPO / GSPO 风格目标，部分 rollout 支持。训练脚本里 `adv_estimator` 默认 `grpo`，另有 GSPO 配方（R2E-Gym 表格里用 GSPO）。`bypass_mode` / `rollout_is` / `rollout_rs` 等是 verl 侧的 token 级重要性采样与拒绝采样开关。

其他训练 recipe：

- `examples/mem_agent/`：MemAgent 训 Qwen3-4B，8 GPU（4 训 + 4 rollout），HotpotQA 32K 训练集，8K 到 1M 八个长度评测。
- `examples/claude_code_kernel_task/`：Claude Code 做 KernelBench / Triton 内核任务，含沙箱 Dockerfile、RL 适配 patch、快照验证工具、`trajectory_processor.py`。
- `examples/blackbox_recipes/` 与 `examples/claude_code_swe_task/`、`examples/mini_swe_agent/`：黑盒 harness 的推理与训练 recipe。
- DeepEyes 训练 recipe（PR #150，open）：`examples/` 下的 DeepEyes 训练路径。

文档里的一个坑（官方自己标了 warning）：标准 verl 数据集按源 prompt 做 prompt 长度过滤，运行时 Task 模板展开后的 Agent 侧 prompt 可能更长。配 prompt 上限时要按所选 recipe 的模板留余量，不能只看源 prompt。

### 5.2 推理与评测

两种推理模式（`docs/source/quickstart/agent-inference.md`）：

- 外部 API 模式：Agent 接已有模型 endpoint，`examples/inference/parallel_infer_api.py`。
- verl rollout 模式：verl 启动 rollout 引擎，走训练 rollout 路径，`examples/inference/parallel_infer_verl.py`。
- Oracle 验证：`examples/inference/parallel_run_oracle.py`，并行跑任务自带的 oracle 解并验证任务奖励。

评测用 SWE-Bench Verified 做运行示例：先 `python -m uni_agent.tasks.swe_bench.preprocess` 生成 parquet，再用 `task_config_react.yaml` / `task_config_claude_code.yaml` 跑。Claude Code 沙箱必须能连到托管其会话级 Gateway endpoint 的 GPU 机器（网络连通性是黑盒路径的常见失败点）。

### 5.3 项目报告的成绩（项目方报告，未独立复现）

并行推理与验证（README 与 `docs/source/benchmark/inference.md`）：

| Benchmark | Agent | 模型 | 设置 | 成绩 |
| --- | --- | --- | --- | --- |
| SWE-Bench Verified | ReAct | Qwen3-Coder-30B | 100 turns，128K | 49.2 |
| SWE-Bench Verified | ReAct | Qwen3-Coder-480B | 500 turns，256K | 64.2 |
| SWE-Bench Verified | Claude Code | Qwen3.5-9B | 200 turns，128K | 51.0 |
| SWE-Bench Multilingual | ReAct | Qwen3-Coder-30B | 200 turns，128K | 35.0 |
| Terminal-Bench v2.0 | ReAct | Qwen3.6-35B | 256K | 42.5 |
| Terminal-Bench v2.1 | Claude Code | GLM5.2-733B | 256K | 67.4 |

Agent RL 训练（基线到 RL 后，SWE-Bench Verified 验证）：

| 模型 | Agent | 数据集 | 设置 | 基线 | RL 后 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | ReAct | R2E-Gym | Fully Async，100 turns，128K | 22.2 | 36.8 |
| Qwen3-Coder-30B-A3B | ReAct | R2E-Gym | Fully Async，100 turns，128K | 46.2 | 52.0 |
| Qwen3.5-9B | ReAct | SWE-reBench | Fully Async，100 turns，128K | 53.8 | 59.2 |
| Qwen3.5-4B | ReAct | SWE-reBench | Colocate Async，200 turns，128K | 52.9 | 59.8 |
| Qwen3-Coder-30B-A3B | ReAct | SWE-reBench | Colocate Async，200 turns，128K | 47.4 | 54.2 |
| Qwen3-Coder-30B-A3B | Claude Code | SWE-reBench | Colocate Async，200 turns，128K | 40.2 | 46.2 |

MemAgent（HotpotQA，8K 到 1M 宏平均）：Qwen3-4B 训前 53.5，训后 58.0，提升 4.5。

README 旧版里的 OpenHands 对比表（Qwen3-Coder-30B 49.2、480B 64.2 等）在上游 main 已被新版结果表取代，新版不再放 OpenHands 对比列。引用成绩时以新版为准。

## 6. 配置与启动

### 6.1 安装

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
git submodule update --init --recursive
pip install --no-deps -e ./verl
pip install -e .
```

非训练流程用标准 Python 3.10+ 环境即可；RL 训练建议用兼容的 verl Docker 镜像。任务依赖与沙箱依赖都是按需装：跑 SWE-Bench 才装 `swebench==4.1.0`，用 Modal 才装 `modal`，用 E2B 才装 `e2b`，用 veFaaS 才装 `volcengine-python-sdk` 与 `swe-rex`。开发时装 `pre-commit` 并 `pre-commit install`。

Ray 分布式任务用 Runtime Environment 把仓库、verl 源码、轻量依赖、凭证分发到每个 worker 节点：

```yaml
working_dir: ./
excludes: ["/.git/"]
pip:
  packages:
    - "volcengine-python-sdk"
    - "swe-rex"
    - "swebench==4.1.0"
env_vars:
  PYTHONPATH: "verl"
```

提交时：`ray job submit --runtime-env runtime_env.yaml -- python entrypoint.py`。

### 6.2 最小使用路径

官方 Quickstart 的顺序是：安装 → 启动沙箱跑代码（`examples/quickstart/sandbox/demo.py`，本地或云）→ 跑 agent 推理（SWE-Bench）→ 跑 agent RL 训练。另有独立的 Harbor 集成、Oracle 验证、Agent-Aware Router、NPU 训练指南。

沙箱最小示例（`SandboxConfig` + `build_sandbox` + `async with`），见第 4.5 节。工具最小示例（`Toolbox.all` + `tools.call("shell", ...)`），见第 4.3 节。训练最小入口是 `examples/quickstart/training/train_qwen3_moe.sh` / `train_qwen3p5_dense.sh` / `train_npu_qwen3_moe.sh` 这类脚本加对应的 `task_config_*.yaml`。

### 6.3 配置的合并与校验要点

- Task Config 解析顺序：YAML 默认值 → 样本 Task Config 深度合并 → 运行时模型 endpoint 最终注入。
- Pydantic 模型普遍 `extra="forbid"`，拼错字段名会直接失败，不是静默忽略。`prompt_template` 的占位符规则见第 4.4 节。
- Gateway 的请求级采样 key 默认只放行 `max_tokens` / `stop`，其余要在 `agent_framework.allowed_request_sampling_param_keys` 里显式配置。
- `mask_unfinished_episode` 必须是严格 bool，传字符串会失败。
- 轨迹后处理器的 FQN 在 Framework 启动时 import，import 失败或目标不可调用都会让初始化失败，不是运行时才报错。
- 沙箱的 `proxy_port` 只在 openyuanrong 上合法，其他 provider 配了会直接报错。

## 7. 与同生态 / 竞品项目的对比

| 项目 | 定位 | 与 Uni-Agent 的关系 |
| --- | --- | --- |
| verl | RL 训练底座（训练器、rollout 引擎、TransferQueue） | Uni-Agent 依赖它，`verl/` 是 submodule |
| verl-recipe | 基于 verl 的端到端 RL 训练示例/配方集 | 配方，不是框架；无 Gateway/会话/沙箱抽象 |
| ProRL-Agent-Server（NVIDIA NeMo） | 在任意 harness 上做 agentic RL | 同目标不同栈，server 形态；不要互相套用接口假设 |
| OpenHands / mini-swe-agent / Claude Code / OpenClaw / Codex / Hermes | Agent harness | 被 Uni-Agent 以黑盒方式接入、评测、训练 |
| Harbor | 任务格式 | Uni-Agent 把它当一种 Task 接入同一套推理管线 |
| SWE-bench / SWE-reBench / Terminal-Bench / HotpotQA / KernelBench | 任务与数据集 | Uni-Agent 的 Task 层封装它们，提供预处理与奖励 |
| vLLM / SGLang | 推理引擎 | 真正生成 token 的地方；Gateway 不是推理引擎 |
| E2B / Modal / veFaaS / Docker / OpenYuanRong | 沙箱后端 | Uni-Agent 的 Sandbox provider，统一数据面 |

Uni-Agent 的差异化不在某一个单点功能，而在“推理与训练用同一套交互栈”这个组合：同一套 Task/Agent/Sandbox/Gateway 代码既跑评测，也产出训练轨迹，轨迹的 token 与训练时的 loss mask 是同一份来源（Gateway），而不是评测与训练各写一套。

## 8. 贡献规范（CONTRIBUTING.md 摘要）

- 欢迎的贡献类型：bug 修复、新 Agent 与 Task 集成、Sandbox 后端、Tool、训练改进、测试、示例、文档。
- 归属原则：行为放在拥有它的抽象里。奖励策略属于 Task，provider 失败属于 Sandbox，协议编码属于 Gateway。不要在拥有层能暴露窄接口时加跨层捷径。
- 每个改动聚焦一个问题。修复要带回归测试，公开契约要带边界测试，异步或远程资源要覆盖生命周期清理与失败路径。API、配置、工作流变化要同步文档与可运行示例。调度、吞吐、内存、训练性能的主张要带可复现的测量数据。不要提交凭证、私有模型响应、专有数据、未脱敏日志。
- 提 PR 前跑：`pre-commit run --all-files --show-diff-on-failure`，以及相关测试（文档示例是 `python -m pytest tests/uni_agent/framework/test_task_runner.py`，实际按改动选最小相关集）。
- PR 标题格式由 CI 检查（`check-pr-title.yml` + `tests/special_sanity/check_pr_title.py`），必须是 `[area] type: summary`。Area 只允许 `agents, framework, gateway, logging, sandbox, tasks, tools, training, app, docs, examples, ci, build, deps, misc`。Type 只允许 `feat, fix, refactor, perf, test, docs, chore, revert`。多个 area 用逗号加空格分隔，如 `[agents, sandbox] feat: ...`。破坏公共 API、Task 或 Sample Config、数据集、模型协议、轨迹契约、checkpoint、文档工作流的改动要在标题最前面加 `[BREAKING]`。堆叠 PR 系列可在前面加 `[1/N]`。
- PR 正文要填 Summary、Changes、Validation、Compatibility，修 issue 时用 `Fixes #123` 关联，列出确切的验证命令与结果，没跑的检查要说明原因，非平凡改动要说明归属层与设计权衡，breaking 改动要写迁移步骤。Reviewer 关注正确性、归属边界、兼容性、失败处理、测试覆盖、可复现性。

## 9. 局限与开放问题

以下均有来源，不是推测：

- **重型依赖**。官方训练示例是 64 H100、1024 并发会话、32 个 Gateway actor 的规模。单机复现的低成本路径只有 quickstart 里的单节点训练（PR #188，open）与 #114 跟踪的单节点端到端 quickstart，尚未成为默认路径。
- **沙箱与网络是常见失败点**。Claude Code 沙箱必须能连到 Gateway endpoint 所在的 GPU 机器；veFaaS / Modal / E2B 都要各自的凭证与网络配置；`image_map`、镜像拉取、私有 registry 都是实际部署里要处理的事。
- **黑盒 harness 的 prompt 不可见**。自渲染 Agent 可能在沙箱内再套一层模板，Runner 观测不到它最终的内部消息。`TaskResult` 只报告 episode 结果，不携带 prompt 来源。这是文档明确承认的边界。
- **多模态模板渲染被推迟**。`prompt_template` 只支持文本，图片/视频/音频等结构化消息内容不支持。#232（多轮追加图片导致 Gateway HTTP 500）说明多模态轨迹合并本身还在修。
- **会话多轨迹的 loss 权重未定**。#185 指出 `trajectory_selection=all` 时一个会话产出多条轨迹，GRPO advantage 按会话算，但 `seq-mean-token-mean` 的 loss 是按行平均，轨迹数多的会话会隐式获得更大权重。Issue 问的是“这是预期语义，还是应该提供可选的会话均衡 loss”，目前没有结论。
- **可观测性缺口**。#129 指出 Uni-Agent 缺少结构化的 rollout 生命周期指标（环境启动、轨迹生成、奖励计算、端到端延迟，含失败会话的汇总），训练与独立评测没有统一的指标视图。#181 指出 router 的 rl-insight emitter 只有 `VERL_RL_INSIGHT_ENABLE` 一个二值开关，全开时高频信号在生产路径上有成本，需要按级别（basic/debug） gating。PR #248（open）开始补 rollout 阶段耗时上报，但 #129 本身仍 open。
- **Router 的结构债**。#182 指出 collector 是 god class（传输生命周期、解析分发、store 业务逻辑、insight 发射四件事混在一起），`DataStore` 是 25 个以上方法的扁平 facade，store 布局或策略语义一变就 ripple 到 collector，collector 离了 store 测不了。
- **混合注意力模型的 KV 路由未处理**。#228 指出 router 的缓存视图没有 KV cache group 身份，vLLM 事件里的 `group_idx` / `kv_cache_spec_kind` 被丢掉，混合模型里不同 group 的块共享前缀哈希但保留与复用规则不同，混成一个视图会误判缓存可用性。
- **模型与任务覆盖仍在补**。路线图里 DeepSeek 模型支持、多模态模型支持、GUI 工具支持、更多云部署后端（Yuanrong Sandbox Management System 等）是未完成项；26Q4 路线图（#191）还列了 SWE-bench Pro、SWE-rebench V2、树形轨迹存储与训练、多 agent 策略训练、推测解码训练、训练恢复增强、AgentEnv 集成、LLM Router 的请求调度与 KV offload、混合注意力模型支持、细粒度奖励建模与信用分配。E2B 集成在 #261 已合入，OpenClaw 在 #174 已合入，Hermes 在 #220（open），Codex 在 #168（open），说明路线图在推进但未完成。
- **训练配置的复现性问题仍有用户报告**。#171 报告用 Qwen3.5 配置训 Qwen3-8B/14B/32B dense 模型时奖励 flat，要求可复现的 Qwen3 dense 训练配置。#166 / #98 是用户要求补充 Qwen3 9B / 30B A3B 训练的更多指标。文档里的训练曲线与配置存在，但外部复现仍需自己调。
- **项目年轻，接口在变**。创建于 2026-01-22，版本 `0.1.0.dev`，`v0.1.0` 在 2026-09-23 才发布。PR 标题规范、归属原则、BREAKING 标记都说明接口仍在演进。`[BREAKING]` 前缀的 PR 在 open 列表里真实存在（如 #199 清理 Docker rollout 超时）。
- **旧 fork README 已过期**。fork main 上的 README 还是 6 月版，提 dashboard、旧版结果表、旧版架构图。上游 main 的 README 已改版（标题改为 Train Long-Horizon Agents at Scale，dashboard 章节删除，结果表更新，Roadmap 只剩指向 #79/#191 的链接）。读 fork main 的 README 会得到过期信息，以上游 main 为准。

## 10. 后续可探索的贡献方向（只列观察，不承诺、不认领、不下 filing 结论）

以下来自 open issue / open PR / 路线图的逐条核对，不是“值得做”的判断。动手前必须重新核对 issue 状态、是否已被认领、是否已有进行中的 PR。

**Bug 类（范围相对收敛）**

- #232 Gateway 多轮追加图片导致 HTTP 500：`GatewaySession._prepare_generation_inputs` → `MessageCodec.merge_context_tokens` 在检测到增量 image/video 数据时直接抛错。复现路径与期望行为（展开新增视觉 token、保留已有生成 token 与 log prob、新增上下文 token 的 mask 为 0、累积媒体与 token 对齐）在 issue 里写得很完整。属于 Gateway + codec 的修复。
- #247 / #246 / #245（均为 open PR）：Sphinx pytest 结果暴露给官方 scorer、SWE 评测前恢复官方测试、shell 正常退出后下一命令重开。三个都是小修复，适合读代码验证。
- #234 / #233（open PR）：本地 exec 超时或取消时杀进程组、veFaaS 单 exec 快照 shell 与连接池。属于 Sandbox 层的失败处理与性能。

**功能类（需要先读设计讨论）**

- #185 会话均衡 loss 权重：涉及 GRPO advantage 与 loss 聚合的语义，改动会碰训练正确性，必须先在 issue 里确认预期语义再动代码。
- #228 多 group KV cache 事件：涉及 router 的事件解析、哈希映射、缓存存储、移除、前缀匹配全链路，PR #216 的 review 里明确要求拆出来单独跟踪。
- #182 router collector 与 DataStore 解耦：重构类，改动面大，需要先理解现有测试对 DataStore facade 的依赖。
- #181 rl-insight emitter 按级别 gating：在 `Emitter._emit` 单点加 `basic` / `debug` 分级，不动调用点，设计在 issue 里已有草案。
- #129 生命周期指标：跨训练、验证、verl 托管评测、独立推理四条路径统一指标名与导出，涉及 Tracking 集成，范围大。
- #171 Qwen3 dense 训练配置：先复现 flat reward 再谈配置，不是只补文档。
- #114 单节点端到端 quickstart：PR #188（open）已在做，动手前必须先看 #188 是否覆盖，避免重复。
- #60 Harbor 集成：文档与示例侧，Harbor 任务格式接入的完整性。
- #132 Client-driven Online RL、#122 Agent-Aware KV Cache Offloading、#77 Multi-policy Disaggregate Trainer、#51 Gateway 多轨迹前缀 Trie 存储：均为 RFC 类，适合先读设计再判断，不适合直接写代码。
- #34 blackbox agent 集成：黑盒接入的通用机制，OpenClaw（#174 已合入）、Hermes（#220 open）、Codex（#168 open）、DeepSeek harness（#259 open）是该方向的具体实例。新 harness 接入的模式是照着已有 recipe 做，而不是改框架。

**路线图类（26Q4，#191）**

- SWE-bench Pro、SWE-rebench V2 任务接入；树形轨迹存储与训练；多 agent 策略训练；推测解码训练；训练恢复增强；AgentEnv 集成；LLM Router 的请求调度与 KV offload；混合注意力模型支持；细粒度奖励建模与信用分配；GUI agent 支持；AI4Science agent 支持。

**动手前的核对清单**

- 重新确认目标 issue 仍 open、无人认领、无进行中 PR。
- 确认归属层（见第 8 节），不要跨层加捷径。
- 确认 PR 标题格式、pre-commit、相关测试命令。
- 任何训练正确性相关的改动（奖励、loss 聚合、轨迹 mask、advantage）必须有最小复现或回归测试，不能只靠“看起来对”。

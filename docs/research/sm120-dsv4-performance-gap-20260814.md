# SM120 DeepSeek V4 Flash 性能差距与上游路线研究

日期：2026-08-14
研究对象：TokenSpeed `feat/sm120-upstream-stack`、DeepSeek-V4-Flash-0731、RTX PRO 6000 Blackwell（SM120）
资料边界：只采用项目官方仓库中的代码、提交、PR、issue、runbook 和原始 benchmark；社区数据均保留其硬件、拓扑和测量口径，不将不同条件的数字直接横比。

## 结论摘要

1. **后续服务不如 53 快，不能归因于“配置了 1M”这一项。** `max_model_len=1,048,576` 只是允许的上限；短 prompt 的 decode 是否快，主要取决于 speculative decode 是否真正生效、target/draft/context-KV 是否完整进入 CUDA graph、KV cache 实际格式、MoE 与 PCIe collective，以及 TP/DCP 组合。当前 TokenSpeed 分支与 RTX6KPro 的快配置并不是同一套合同。
2. **当前分支已有可信且可启动的 SM120 功能基座，但 performance contract 尚未闭环。** 分支已包含上游主线、[#948](https://github.com/lightseekorg/tokenspeed/pull/948) 的架构归一化，以及当前 [#992](https://github.com/lightseekorg/tokenspeed/pull/992) 的 SM120 sparse-MLA、FlashInfer CUTLASS MXFP4 MoE、持久 workspace、跨架构路由和 RSAG fallback。`89b163ac` 修掉 baseline `sm_120` 的 FP4 conversion PTX 泄漏，`c0f28823` 适配 FlashInfer 0.6.16 autotuner API；新增的显式 packed-MXFP4 expert contract 已使 48/48 shards 完整加载。现场还修复了 SM120 上 FlashInfer fused-MoE 调优准备阶段的坏 tactic，以及 DSV4 indexer/SWA stateful cache-write 双流竞态。TP4、target-only、exact max 1M、decode graph 和 16--4096 全部 prefill graph bucket 已完成冷启动与正确性请求。随后 profiler 定位并修复 SM120 compressor/router 错退 FP32 matmul、decode 被过度串行化和 indexer schedule metadata 重复生成，4K/256 decode 从 68.66 提升至 76.22 tok/s；`120f` 干净构建与显式 MoE tactic 表又将热态重复结果提升至平均 79.97 tok/s，但仍未接近 53 的约 107 tok/s。
3. **1M 支持是四层能力，不是一个布尔开关：** 配置接受 1M、KV 容量容纳实际 1M、attention/cache writer 在实际深度正确、调度与图执行在冷 1M prefill 后仍稳定。DGX Spark 的社区 runbook 证明 1M 可以跑通，但它们也公开了 256K 冷 prefill 主机重启、32K decode 补丁回退、600K cache dispatch 回退等问题；“启动显示 1M”不等于完整闭环。
4. **应先工程化分支，再继续性能实验。** 以更新后的 `main` 为唯一基线，在 `dev` 集成分支维护可复现的 feature stack；每个能力从独立 feature 分支进入，并携带 correctness、graph coverage、实际上下文和性能 gate。不要把旧 PR 整包 cherry-pick，也不要继续在远端节点上堆不可追溯 patch。
5. **工程顺序与吞吐收益顺序必须分开。** 先完成实际 1M compact NVFP4 correctness 合同，再做性能优化；就吞吐潜力而言，把已合入 TokenSpeed 的 Week-0 DSpark 做成 SM120 完整全图实现最有价值。RTX PRO 6000 数据显示 speculative 的收益可很大，但它完全受 acceptance 和 verify step 支配；仅加载 drafter、没有高 acceptance 或存在 eager break，甚至会倒退。严格 graph coverage、长 KV indexer、MoE/verify 和 DCP 都必须实现，但要以独立实验臂进入，不能用一个混合配置掩盖因果。
6. **本轮 TP4/DCP2 已跨过 correctness gate，但当前实现明显受通信/launch latency 限制。** exact-1M 容量 receipt 为 6.85 GB/rank；2-GPU NCCL CUDA graph primitive 64/64 replay 通过；4095-token prompt 固定输出 64 tokens 的 20 次服务请求 hash 全同且与 DCP1 相同。但同服务 4095/256 C1 decode 三次只有 46.9/47.0/47.8 tok/s。按 43 层和 21 个 c4 层展开，现路径约有 214 次 DCP collective/token；静态 sink、重复 candidate gather 与未融合 LSE merge 是首先要消除的结构性成本。

## 当前分支的真实能力

研究基座为 `feat/sm120-upstream-stack`，当前验证分支为 `feat/sm120-native-build`，基于上游 `main@d34dcf1a`：

| 能力 | 当前状态 | 证据与限制 |
|---|---|---|
| SM120/SM121 构建 | baseline 已闭环，performance target 待闭环 | #948 与 #992 提供 arch 解析和 opt-in；`89b163ac` 使 baseline `sm_120` 干净构建通过，`c0f28823` 修复 FlashInfer 0.6.16 autotuner initializer API。最终 dense FP4 镜像仍应切到 `120f` 并做数值/性能资格；`120a` 只作独立 A/B。 |
| DeepSeek V4 sparse MLA decode | 已实现 | `deepseek_v4.py` 在 NVIDIA arch major 12 选择 FlashInfer DSV4 sparse MLA；支持 SWA page 64 和压缩 cache page 0/2/64。 |
| SM120 MXFP4 MoE | 已实现并完成真实 checkpoint 启动 | FlashInfer CUTLASS fused MoE，MXFP8 activation × MXFP4 weights，含 TP/EP、持久 workspace 和 clamped SwiGLU 语义。0731 转换 checkpoint 通过显式 `mxfp4_e8m0` expert-format contract 识别 packed half-width E2M1 weights 与 raw E8M0 K/32 scales；48/48 shards 完整加载。 |
| DeepSeek V4 hybrid cache 规格 | 已实现 | full-history compressed groups、FP8 DS-MLA row、68-byte MXFP4 indexer row、精确 token-capacity 反算，允许模型 1,048,576 上限。 |
| CUDA graph decode | 有基础能力 | 目标模型 decode 可进图；主线已有 DFLASH/DSPARK padding 和 breakable-prefill 基础设施，但当前 dispatch 仍以 decode 为门槛，非 decode 路径会退回普通 forward。 |
| DSpark | 功能性实现，不是高性能完成态 | [#940](https://github.com/lightseekorg/tokenspeed/pull/940) 与 [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048) 已在主线：同 checkpoint drafter、canonical paged cache、graph-safe replay。当前实现明确为 Week-0，只接受 TP，proposal/sampling 仍偏 PyTorch/greedy，缺少外部快栈的生产级 drafter kernels、概率采样、动态深度和完整图资格。 |
| 真 compact NVFP4 DS-MLA KV | 上游分支缺失；本地 integration prototype 已运行 | 当前上游 attention cache 仍是 FP8 DS-MLA；MXFP4 indexer cache 不是 compressed MLA KV。旧本地 integration tree 已实现显式 `dsv4_nvfp4_368` writer/reader/dual-cache ABI，并在 59/8030 的 exact-1M 服务运行，但仍未整理到新 `dev`，也未达到目标 decode 性能。 |
| DCP / sequence parallel | 本地 TP4/DCP2 correctness-first 路线已通过容量、graph primitive 与短请求正确性 gate，性能未闭环 | #364 只有 kernel primitive；当前分支已补 DCP subgroup、物理 token stripe、Q/O/LSE collective、prefill/indexer reconstruction 和固定 workspace。`max_model_len=1,048,576` 的容量 receipt 为每 rank 6.85 GB，2-GPU NCCL CUDA graph primitive 连续 replay 64 次通过；同一 4095-token prompt 固定生成 64 tokens，DCP2 重复 20 次输出 hash 全同，且与 matched DCP1 hash 相同。该证据不等于实际 1M prompt 已完成，也不等于性能达标：4095/256 C1 decode 三次只有 46.9/47.0/47.8 tok/s。 |
| ADP / DP attention | **缺失** | 没有可 cherry-pick 的 TokenSpeed 上游实现。它主要针对高并发 KV 去重，不应作为 C1 低延迟的第一优化。 |
| P/D cache handoff | 未纳入 | [#997](https://github.com/lightseekorg/tokenspeed/pull/997) 仍开放，已做 4×SM120、TP2/EP2 的 P/D smoke；更新的 [#1057](https://github.com/lightseekorg/tokenspeed/pull/1057) 正在统一 cache recipe/plan 和异构 TP 合同。不是当前单服务 TP4 的必要依赖。 |
| SM120 启动期调优与 DSV4 cache-write overlap | 已现场闭环 | FlashInfer 0.6.16 的 fused-MoE `tactic=-1` preparation 在异常边界外，SM120 TMA-WS 会污染 CUDA 状态；启动期继续跳过这两个不安全 profiling op，但离线完整枚举显式 profile ID，并按 GPU、CC、FlashInfer、CUTLASS tactic ABI、Torch/CUDA、MoE shape、TP/EP 与 PDL 精确绑定 21-bucket 表。另以四臂 A/B 确认 Phase-2 indexer/SWA/compressor-state 双流并发会触发 4096-token prefill IMA；只在 SM120 的 prefill/mixed indexer 分支串行，decode 恢复 overlap，保留 Phase-1 projection/compressor-GEMM 与 compressor-only cache-write overlap。 |

当前 stack 不是另一套私有实现：除 #948 外，6 个 SM120 kernel/runtime 提交与 #992 当前 head 的初始实现及后续修正对应，另有集成修复；`feat/sm120-native-build` 再独立承载干净 SM120 构建修复。因此现在不该再次从 #992 cherry-pick；应在当前 stack 上逐项闭环。

## 53 与当前 TokenSpeed 服务的现场对照

### 运行合同

2026-08-14 的在机审计确认，两台主机都是相同的 8×RTX PRO 6000D、600 W 上限、PCIe Gen5 x16，桥上 ACS redirect 已关闭且 P2P qualification 通过；硬件型号、功耗上限和基础拓扑不是 34% decode 差距的解释。

| 实验臂 | 实际运行栈 | 关键合同 |
|---|---|---|
| 53/8000 | vLLM | TP4/DCP2、FP8 DS-MLA、B12X attention/MoE/linear、target-only、exact max 1M |
| 53/8001 | vLLM | TP4/DCP1、padded NVFP4 DS-MLA、B12X、DSpark K5、exact max 1M |
| 59/8030 | TokenSpeed 旧私有镜像 | TP4/DCP1、exact-368 NVFP4、B12X W4A16、target-only、strict graph route、exact max 1M |
| 59/8040 | TokenSpeed 新上游栈 | TP4/DCP2、FP8 DS-MLA、FlashInfer MXFP8×MXFP4、target-only、exact max 1M、decode + breakable prefill graph |

这里最关键的工程事实是：59/8030 运行的是 `tokenspeed:deepseek-v4-1m-tp4-nvfp4-graphfix`，**不是**当前 `feat/sm120-upstream-stack`；59/8040 才是 #948/#992 新路径。53/8000 的镜像则明确使用 CUDA 13.2、`TORCH_CUDA_ARCH_LIST=12.0a`、`FLASHINFER_CUDA_ARCH_LIST=12.0f`、B12X A8、DCP2 与 PCIe-local-inference collective；59 新镜像是 `sm_120f`、CUDA 13.0、FlashInfer CUTLASS、DCP2。因此 53/59 不是单一 kernel A/B，而是完整 target stack A/B。

新分支镜像 `tokenspeed:sm120-native-c0f2882` 在 59 的 0--3 卡首次启动时，四个 rank 均在第 0/48 shard 以 `start (0) + length (512) exceeds dimension size (4)` 退出。现场检查确认 0731 checkpoint 虽在 `config.json` 声明 `quant_method=fp8`，expert `weight` 实际为 packed half-width MXFP4，scale 为 E8M0 K/32。新增 `--deepseek-v4-expert-weight-format mxfp4_e8m0` 后，dense/shared tensors 仍遵循模型 FP8 config，只有 routed experts 按 serialized MXFP4 创建并把 scale 映射到 raw `weight_scale`；48/48 shards 在四 rank 全部加载。这个显式合同没有 shape 猜测、兼容层或静默 fallback。

完整启动又暴露两个独立的 SM120 runtime 问题。第一，FlashInfer 0.6.16 在 fused-MoE autotune 的 preparation 阶段以 `tactic=-1` 运行 TMA-WS，异常发生在 per-tactic `try/except` 外；精确 M=4096/TP4 微复现表明 native CUTLASS kernel 本身可正确执行，只需让 `gemm1/gemm2` 使用 FlashInfer 官方 `skip_ops` 的 heuristic tactic，其他 op 继续调优。第二，4096-token prefill graph 在 stateful DSV4 attention phase 触发 illegal memory access。逐阶段同步、Phase-1-only 串行、Phase-2-only 串行、indexer-only 串行四臂证明：Phase 1 投影/GEMM 与 compressor-only cache-write overlap 可保留，Phase 2 的 indexer、SWA cache insert 与 compressor-state write overlap 是根因。只在 SM120 prefill/mixed 串行 indexer 分支、decode 恢复 overlap 后，正常 autotune、decode graph、9 个 prefill bucket、warmup 与 `OK` 正确性请求均通过。

### 相同请求的初步结果

同一 tokenizer 构造精确 token 长度、集群内直连、C1 固定输出的初步 A/B 如下；这些点用于定位因果，不替代最终五次重复的交叉报告：

| 实际 prompt / output | 53/8000 decode | 59/8030 decode | 59/8040 初始 / 修复后 decode | 53 / 旧59 / 修复后新59 TTFT |
|---:|---:|---:|---:|---:|
| 4K / 256（最新 matched） | 107.23 tok/s | 69.69 tok/s | 68.66 / 76.22 tok/s | 0.897 / 0.290 / 0.785 s |
| 32K / 128（旧栈复测） | 108.85 tok/s | 71.43 tok/s | 待完整矩阵 | 4.725 / 5.664 / 待测 |
| 128K / 128（旧栈复测） | 109.50 tok/s | 69.97 tok/s | 待完整矩阵 | 19.612 / 23.028 / 待测 |

在 2026-08-14 再次以同字节 payload 复测：4K/256 为 107.28 对 69.81 tok/s；使用新重复 token 避免刚产生的 prefix 后，128K/128 为 108.23 对 67.31 tok/s，TTFT 分别为 19.59 s 与 9.70 s。59 这次 128K prefill 反而更快，而 decode 仍慢 37.8%，进一步把问题收敛到每步 target execution，而非 1M 配置或冷 prefill。

decode 差距在 4K 到 128K 基本恒定为 34–38%，而 warm prefill 差距约 17–20%。这直接排除了“1M 上限让短上下文 decode 变慢”，也不符合单纯长 KV 扫描瓶颈；首要变量是每 token 都执行的 MoE/linear、collective、DCP 与 native target。首轮 GPU trace 随即证明，新栈的 compressor/router 因旧架构 taxonomy 把 SM120 排除而退到 FP32 matmul，且 prefill 竞态修复错误地把 decode 也串行化；修复后 4K/256 从 68.66 提升到 75.37 tok/s。再将 indexer schedule metadata 从逐 layer 重算收敛为逐 step/key 一次后达到 76.22 tok/s。与此同时，59 的四个 TP rank 均确认 TRTLLM one-shot all-reduce 已启用，53/59 PCIe/NUMA 拓扑相同，故节点间主差距不能归因于 P2P 初始化失败。

### `120f` 与显式 MXFP4 MoE tactic A/B

`CUDA_ARCH_LIST=12.0f` 的干净镜像完成 48/48 shard 加载、9 个 prefill graph bucket、exact max 1M capacity 和短请求正确性。其 4K/256 热态三次为 76.585/77.241/77.152 tok/s，平均 76.99；相对 baseline build 的 76.22 只约 +1.0%。因此 `120f` 是正确的 dense-FP4 构建合同，但它本身不是 53/59 差距的主要答案。

FlashInfer 0.6.16 为该 SM120 CUTLASS MoE 暴露 GEMM1 20 个、GEMM2 40 个绝对 profile ID，但 shape/occupancy 查询仍把许多无法 dispatch 的组合列为可用；真实执行只有各 6 个成功。新增离线 sweeper 不进入会触发 `tactic=-1` 的 autotune context，而是对 1--8192 的完整 hybrid bucket 梯子逐一执行全部 stage 候选和 36 个可执行 pair，先过 finite/reference gate，再用 CUDA event 做 5 次预热、50 次测量。每个 bucket 同时覆盖 experts 分散和高度集中两种路由，最终选择相对两种 profile 各自最优值之最大 regret 最小的 pair。PDL 是策略身份的一部分：PDL=false 表在真实服务 warmup 被 fail-closed 拒绝后，重新生成 PDL=true 表；没有通过禁用 PDL 迁就旧表。

PDL=true 微基准中，单 token 的稳健策略把 spread/concentrated 从 164.445/155.560 µs 降到 72.049/72.724 µs，即 2.28×/2.14×；64-token 为 1.91×/2.31×，512-token 为 1.75×/1.33×，4096-token 为 1.12×/0.97×。这说明原生 heuristic 在 decode 小 M 明显失配，但收益会随 batch/routing 收敛，不能把单 kernel 的 2×外推到整模型。

接入真实 TP4、target-only、FP8 KV、exact max 1M、全图服务后，4095/256 的三次 decode 为 79.4/80.2/80.3 tok/s，平均 79.97；相对同镜像未固定 tactic 的 76.99 为 +3.9%。服务仍比 53/8000 的 107.23 低约 25.4%。结论是：显式 tactic 表值得保留并已工程闭环，但剩余主差距位于 MoE 之外或 graph 内多项叠加，下一步应回到 step-level target profile、DCP2/compact KV 与完整 graph，而不是继续把全部差距归因于单个 MoE kernel。

53/8001 的历史 speculative acceptance 只有 `94 / 21,130 = 0.445%`，平均 decode 约 77.6 tok/s，慢于无 speculative 的 53/8000 约 107.3 tok/s。因此**当前 53 的优势不是 DSpark**；它来自 FP8+DCP2+B12X/vLLM 这一整套 target path。DSpark 仍有很高的后续收益潜力，但必须先修 acceptance 与 full graph，不能把开关状态当成能力完成。

同集群此前保留的 r33 原始 JSON 又给出一个更强的硬件上限证据：55 上 TP4、131K 上限、DSpark/B12X 的单服务持续 decode 为 C1 222.7、C4 478.3、C8 637.9 aggregate tok/s；另一轮双服务并行时该臂为 221.4/495.3/510.2。它不是 exact-1M 合同，不能拿来替代最终服务，但证明 6000D 上 200+ C1、500+ aggregate 已经在本集群真实出现过。当前 exact-1M TokenSpeed 的 70 tok/s 不是硬件上限。

### TP4/DCP2 本轮实测基线与根因

下面只记录 2026-08-14 当前 DCP feature stack 的现场结果。容量、collective primitive、端到端正确性和端到端吞吐是四种不同口径，不能互相替代：

| 验证层 | 精确口径 | 结果 | 能证明什么 / 不能证明什么 |
|---|---|---|---|
| exact-1M 容量 receipt | DeepSeek-V4-Flash-0731，TP4/DCP2，`max_model_len=1,048,576`；按当前 DSV4 hybrid-cache recipe 对完整模型层数做物理分片和启动期容量核算 | **6.85 GB/rank**；保留 receipt 原始 GB 单位，不换算 GiB | 证明 DCP2 的每-rank cache allocation 能按 exact-1M 配置建立；这不是一条实际 1M-token prompt 的 prefill/decode 成功记录。 |
| 真实 NCCL CUDA graph primitive | 2 张 GPU、DCP world size 2；attention tensor 为 BF16、LSE/candidate score 为 FP32、candidate row 为 INT32；`rows=2`、local heads 2、head dim 4、top-k 3；3 次 warmup 后 capture，输入在两种 pattern 间交替；graph 内同时执行 Q all-gather、fused LSE/output merge + reduce-scatter、packed indexer candidate all-gather/top-k | **64/64 replays 通过**，每次均与显式 reference 一致 | 证明 `c99217c` 修复后的固定 workspace和新的 A/B/C 组合可被 capture/replay，并覆盖 multi-row RS 布局；它是小 shape 的 distributed primitive gate，不是 43 层生产服务 full-graph 性能证明。 |
| 端到端确定性 | TP4/DCP2 服务；同一 tokenizer 产生的实际 4095-token prompt、相同生成参数、固定输出 64 tokens；连续重复 20 次，并与 matched DCP1 输出比较 | **20/20 完成且输出 hash 全同；该 hash 与 DCP1 相同** | 证明该短请求上 DCP2 的 cache sharding、indexer reconstruction 和 attention merge 语义与 DCP1 一致；不外推到实际 256K--1M 深度或非确定性采样。 |
| 端到端 C1 decode | 同一 TP4/DCP2 服务、同一实际 4095-token prompt、固定输出 256 tokens、单并发；沿用 matched harness 的 decode-only 口径，不含 TTFT；连续 3 次完整请求 | **46.9 / 47.0 / 47.8 tok/s**，平均 47.23、median 47.0 tok/s | 这是当前 DCP2 性能基线，不是目标值。此前同栈 DCP1 热态约 79.97 tok/s、53 的 vLLM TP4/DCP2 约 107 tok/s 只能作为方向性参照；由于 feature stack/backend 不完全相同，不能把差值全部归因于 DCP degree。 |

本轮还定位了一个独立的 **multi-row correctness root cause**。旧 `merge_dcp_attention_states()` 先取 `reduce_scatter_input[:, :rows]`，再执行 `permute(1, 0, 2, 3).reshape(rows, group_heads, head_dim)`，把结果当作 `torch.mul(..., out=corrected)` 的 destination。当 `rows > 1` 时，该 permute 后的 stride 无法按目标 shape 表示为 view，`reshape` 会物化临时 tensor；乘法写进临时 tensor，而后续 NCCL reduce-scatter 读取的原 `reduce_scatter_input` 没有被写入。`rows=1` 恰好可形成 view，所以普通 C1 decode 会掩盖问题。当前修复按 destination rank 显式 view/permute `local_out` 与 weight，并直接 `out=reduce_scatter_input[:, :rows]`；上述 2-GPU、multi-row、64-replay test 是该修复的 graph correctness gate。这个 bug 解释旧 multi-row 错误，**不解释修复后 C1 仍只有约 47 tok/s**。

修复后的 C1 性能问题首先是 collective/launch latency，而不是 PCIe payload bandwidth。官方 0731 配置有 43 个 attention 层、64 heads、head dim 512；TP4 每 rank 为 16 local heads，DCP2 kernel 覆盖 32 group heads。43 层中有 21 个 c4 indexer 层。当前每个生成 token 的 DCP hot path 为：

- 每个 attention 层执行 Q all-gather、sink all-gather、LSE all-gather 和 output reduce-scatter，共 `43 × 4 = 172` 次 collective；
- 每个 c4 层分别 all-gather top-k scores 和 rows，共 `21 × 2 = 42` 次；
- 合计约 **214 次 DCP collective/token**。该数目不包含 TP linear/MoE 已有的 all-reduce，也不包含进程级 barrier。

在 C1、BF16 output/FP32 LSE 下，每 rank 每层的逻辑 peer payload 只有约 16 KiB Q、64 B sink、128 B LSE 和约 16 KiB output contribution；21 个 c4 层的 scores/rows 约再增加 84 KiB/token。总量约 1.4--1.5 MB/token，47 tok/s 只对应约 70 MB/s，远低于 PCIe Gen5 x16 的可用带宽。相反，47.23 tok/s 是约 21.17 ms/decode step，79.97 tok/s 是约 12.50 ms/step；把两者约 8.67 ms 的差额仅作为诊断估算摊到 214 次 collective，约为 40.5 µs/次，符合小报文 collective、stream ordering 和 launch latency 的量级。该换算不是逐 kernel profiler 归因，但足以否定“先继续调大通信 chunk 就会解决”的方向。

此外，当前 sink 是每层静态参数，却在每个 decode step 做一次 all-gather；c4 candidate 的同一 `[rows, topk]` 记录被拆成 scores/rows 两次 all-gather；LSE merge 又由 `nan_to_num/amax/sub/exp/sum/log/div/mul` 等约 10 个 PyTorch CUDA op 组成，然后才进入 reduce-scatter。这三项共同造成大量小同步点。后续按下列 A--E 顺序推进，每一步都保留独立 correctness 与 matched performance gate：

| 阶段 | 最小改造 | hot-path 影响 | 必须通过的 gate |
|---|---|---|---|
| A：静态 sink | 利用 checkpoint loader 已经看到完整 `attn_sink` 的事实，为每层保留完整 global sink；owner rank 直接取所属 DCP subgroup view，非 owner 使用预分配全 `-inf` tensor。不要在第一次 graph capture 中懒初始化，也不要每步 copy/fill。 | 去掉 43 次 sink all-gather，约 `214 → 171` collectives/token。 | DCP1/DCP2 sink head slice、owner 语义、全层输出 hash；capture 前地址固定。 |
| B：合并 indexer candidate | 把一个 candidate 的 score bits 与 global row 组成一个 8-byte record，一次 all-gather 后再由 fused/materialize kernel拆出；不改变 global top-k 语义。 | 每个 c4 层 2 次 all-gather 变 1 次，再少 21 次，约 `171 → 150` collectives/token。 | FP32 score 的 bit-exact round trip、无效 row、tie/top-k 次序、prefill/decode 和 graph replay。 |
| C：融合 LSE 与 RS-layout writer | 在 `tokenspeed-kernel` 的 attention merge-state family 新增一个 kernel：从 gathered LSE 做 nan-safe max/exp/sum/log，计算本 rank partial weight，并直接写 destination-major RS input，同时写 merged LSE。runtime 仍保留 NCCL LSE all-gather 与 RS。 | collective 数仍约 150，但每层约 10 个小 CUDA op 收敛为 1 个，并从结构上固定 multi-row RS layout。 | DCP2/4、rows 1/多行、head dim 512、NaN/+inf/-inf/all-`-inf`、预分配 output 地址与 64 次 graph replay。 |
| D：DCP-group IPC collective | 为 DCP group 建独立 CUDA-IPC/TRTLLM one-shot workspace，替换精确小 shape 的 Q all-gather/RS；不能复用当前 TP4 全局 workspace，也不能在配置启用后静默回退 NCCL。 | 先减少每次 collective 的固定延迟；调用次数不因替换 backend 自动下降。 | 本机 P2P/IPC qualification、进程重启与资源释放、graph replay generation、NCCL matched 数值和逐 shape CUDA-event A/B。 |
| E：peer-read fused state merge | FlashInfer attention 把 partial output/LSE 写入固定 IPC shared buffer；单个 kernel peer-read 两个 DCP rank 的 state，完成 online-softmax merge并只写本 rank heads，取代 LSE all-gather + correction + RS。 | 在 A/B 后再移除 `43 × 2 = 86` 次 LSE/RS collective，剩约 43 次 Q gather + 21 次 candidate gather，即约 **64 collectives/token**，另有每层一次 merge/barrier kernel。 | Lamport/barrier 代际、graph replay、peer lifetime、empty partition/all-`-inf`、长上下文一致性，以及故障时 fail-closed。 |

A+B 合计只移除 64/214、约 30% 的 DCP collective。若粗略按上述 latency 线性估算，只可能回收约 2--2.7 ms/step，对应约 53 tok/s 的量级；这是规划估算而非实测，不能写成收益承诺。要从约 47 追回 DCP1 的约 80，至少需要 C 的 launch fusion，并很可能需要 D/E 的 IPC 与 peer-read merge。每阶段必须分别记录 Q gather、attention kernel、LSE merge、RS 和 indexer candidate gather 的 CUDA-event/NVTX 时间，不能只看一个端到端 tok/s 猜根因。

### 对新分支的判定方法

- #948 的架构归一化和 #992 的 SM120 sparse MLA 是必要基座，价值明确。
- #992 的 CUTLASS MXFP4 MoE 是可用候选；修复每-token SM120 dispatch/overlap、切换 `120f` 并固定离线穷举 tactic 后，现场 matched 路线已从 68.66 升到平均 79.97 tok/s，越过旧 B12X W4A16 的 69.69，但仍未接近 53 B12X A8/DCP2 的 107.23；下一步继续分层 profile linear/collective/attention，并推进 DCP2/compact KV。
- #992 的公开验证只有 exact-shape MoE profiling 和双卡 TEP2 smoke，没有 TP4、exact-1M 或吞吐表。本地首次干净 `sm_120` build 又发现 SM100/feature-specific FP4 PTX 泄漏；所以它“有意义但实现不完整”，不能直接称作成熟部署方案。
- GPU trace 已证实 DeepSeek V4 compressor/router 的旧 gate 只允许 Hopper 或 SM10x Blackwell，导致 SM120 compressor 反复把 BF16 activation/weight 转成 FP32 matmul；将其改为 NVIDIA SM90+ 的既有 BF16→FP32 GEMM 路径，并只对 SM120 prefill/mixed 禁用 indexer cache-write overlap，matched decode 提升 9.8%。
- 新分支已经落在约 80 tok/s，结论是实现可用但性能能力仍不完整。`120f` 与 MoE tactic 已完成，下一步直接做 linear/collective/attention 的 graph 内 profile、compact NVFP4/DCP2 与 full DSpark graph；不是继续换机器，也不是继续修 loader。

## TokenSpeed 上游：哪些值得合，哪些不值得

### 已经覆盖，不要重复搬运

- [#992](https://github.com/lightseekorg/tokenspeed/pull/992)：当前 SM120 基座，已经在分支中。PR 的有效内容包括 sparse MLA、SM120 MXFP4 MoE、持久 workspace、路由和真实双卡 SM120 smoke；PR 没有公开可用于横比的完整吞吐表。
- [#948](https://github.com/lightseekorg/tokenspeed/pull/948)：SM120 架构归一化，已经在分支中。
- [#1048](https://github.com/lightseekorg/tokenspeed/pull/1048)：H20 TP8 的 DSV4 FP8 DSpark + CUDA graph 验证已经在主线。公开结果是 GSM8K 1,319/1,319、mean accuracy 0.9659、verify width 6、接受长度约 2.4–4.4；它证明语义与 replay 路线，不证明 SM120 性能。
- [#1076](https://github.com/lightseekorg/tokenspeed/pull/1076)：Blackwell Flat table runtime geometry/gather 已在主线。PR 声称匹配 workload 下 TPS/GPU 2.9–3.9×，但详细 provenance 非公开，不能拿来预测我们的绝对吞吐。
- [#1026](https://github.com/lightseekorg/tokenspeed/pull/1026)：sparse compress ratio ≥128 的 16-warp 路线已在主线，但当前只 gate SM100；B200 TP4 早期 arm 报告 +39–42% TPS/GPU。SM120 仍走 4 warps，应先做 microbenchmark 再决定是否扩展 gate，不能把 B200 收益直接套到 6000D。
- [#597](https://github.com/lightseekorg/tokenspeed/pull/597)、[#611](https://github.com/lightseekorg/tokenspeed/pull/611)、[#613](https://github.com/lightseekorg/tokenspeed/pull/613)、[#614](https://github.com/lightseekorg/tokenspeed/pull/614)：breakable prefill graph、full-window metadata sizing 和 SWA sanitize 已在主线；#614 的 B200 p4 公开结果约 -0.46 ms/step，同样不是 SM120 速度承诺。
- [#820](https://github.com/lightseekorg/tokenspeed/pull/820)：启动期 dummy-prefill autotune、serving 只查 cache/heuristic 已在基线；SM120 调优时应保留 autotune，不要为了减少启动时间把它关闭。
- #1065、#1069、#1073：largest-first graph capture、aux-stream warmup、workspace 复用均已在基线，无需再移植。

### 可以重实现设计，不应直接 cherry-pick

| 上游项 | 一手结果 | 决策 |
|---|---|---|
| [#714](https://github.com/lightseekorg/tokenspeed/pull/714) DSA prefill graph | 关闭未合并；旧 backend 上把 61 个 eager attention break/62 个 graph segment 纳入 capture，TP4 GSM8K 前 50 条 0.98，无吞吐数据。 | 在新 DSV4 backend 重实现“严格 prefill graph + 可观测 break 原因”，不要搬旧 backend diff。 |
| [#555](https://github.com/lightseekorg/tokenspeed/pull/555) mixed prefill/decode split | 关闭未合并；把同 tick mixed batch 拆成 decode graph first + prefill extend。作者后续 DSV4-Pro agentic C1/C2/C4/C8/C16 总吞吐分别报告 +55.2/+9.5/+16.1/+30.7/+21.4%，但承认 C1 baseline 未 prewarm，Qwen C8/C16 反而 -1.27/-3.29%。 | 在 #714 之后做 DSV4-only 重实现和 matched A/B；旧 scheduler diff 不直接 cherry-pick。 |
| [#563](https://github.com/lightseekorg/tokenspeed/pull/563) indexer-Q/host-sync cleanup | 关闭未合并；SM100 TensorRT kernel 不适用于 SM120，但 CPU length mirror、persistent metadata、top-k 直写 destination 可减少 host sync。 | 只移植 vendor-neutral runtime 思路；不要搬 SM100 CUDA adapter。 |
| [#620](https://github.com/lightseekorg/tokenspeed/pull/620) L2 KV | 已合并；DSV4 TP4/DP1、容量上限 81,920 tokens 的 prefix-heavy workload，output 53.23→69.57 tok/s，TTFT 平均 -28.91%。 | 适合重复 prefix 和容量卸载，不是 raw decode 优化；放在 exact 1M 核心路径之后。 |
| [#997](https://github.com/lightseekorg/tokenspeed/pull/997) / [#1057](https://github.com/lightseekorg/tokenspeed/pull/1057) P/D | #997 兼容 #992，公开的是 4 GPU smoke；#1057 是 K3/B300 的统一 cache-contract 重构，无 SM120/DSV4 性能数字。 | 只有确定要做 P/D 时跟踪 #1057 的最终合同，不在当前 IFB 分支直接 pick #997。 |

### 明确不要进入 dev

- [#645](https://github.com/lightseekorg/tokenspeed/pull/645) 与 [#648](https://github.com/lightseekorg/tokenspeed/pull/648) 是被 #992 取代的旧 SM120 栈。#648 的旧 TP2/EP2 结果（随机 8K/1K 到 C32 均 252/252，C32 648.5 output tok/s）只能作历史参照，不能与当前代码混合。
- [#551](https://github.com/lightseekorg/tokenspeed/pull/551) 的 Blackwell capability taxonomy 因 inactivity 关闭；只有发现 registry 仍把 SM100 TMEM kernel 错派给无 TMEM 的 SM120/121 时，才以小提交重实现 taxonomy，不整支 pick。
- [#617](https://github.com/lightseekorg/tokenspeed/pull/617) 的 custom tree mask 明确拒绝 tree-mask + DCP，并让该 decode 绕开 CUDA graph，不符合当前 strict graph/DCP 目标。
- TokenSpeed 官方 open PR/issues/branches 中，没有成熟的 DSV4 DCP、ADP 或真 compact NVFP4 DS-MLA KV 分支可直接 cherry-pick。相关能力需要以当前 API 边界重实现。

## 外部一手结果：能证明什么，不能证明什么

### DGX Spark 的 1M 路线

[tonyd2wild 的 2×DGX Spark 仓库](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) 使用每节点 1 个 GB10、TP2：

- [1M checkpoint](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/benchmarks/20260629-dspark-nvfp4-1m-context-checkpoint.md)：`max_model_len=1,048,576`、`max_num_seqs=1`、`max_num_batched_tokens=8192`；KV 14.48 GiB、2,044,166 tokens、1M 并发 1.95×。短 prompt 的 p256/g64 为 54.46 tok/s，p256/g256 为 65.38 tok/s。这里的 Stage-C 是 **584-byte padded envelope**，不是真 360-byte compact NVFP4；早期真布局在约 411 prompt tokens 处失败。
- [seqs=6 concurrency](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/benchmarks/20260629-dspark-1m-seqs6-concurrency.md)：同样声明 1M，6 个约 200-token completion 的聚合吞吐 181.7 tok/s、每流约 30 tok/s；单流约 67 tok/s。它测的是 1M 服务上限下的短请求，并非 6 条实际 1M 请求。
- [200K concurrency checkpoint](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/benchmarks/20260629-dspark-keys-concurrency-checkpoint.md)：静态 C1/C4/C8/C16 为 57.6/140.8/252.6/315.1 tok/s；staggered C4/C8/C16 为 109.2/147.3/205.0，说明调度形态会改变结论。
- [1.5M C12 checkpoint](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/benchmarks/20260702-keys-c12-1p5m-nvfp4-checkpoint.md)：MTP5、KV 3,225,280 tokens，C1 52.79，C2 79.76，C4 134.70，C6 127.78，C12 230.10 aggregate tok/s。仍主要证明容量和并发，不是 1.5M 深度 decode。
- [1M agent stability](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/benchmarks/20260630-asusi-spark4-nvfp4-1m-agent-stability.md)：code decode mean 54.22，C2/C4/C6 aggregate 60.95/83.21/104.11；acceptance 随并发降到 C6 的 0.307，展示了 speculative 收益并不恒定。

[MiaAI-Lab 的 2×DGX Spark 仓库](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) 同样是 TP2/ConnectX-7，默认 `max_model_len=1,048,576`、seqs 6、MTP5、B12X MoE 和 NVFP4 DS-MLA：

- README 公开的 KV 为 18.08 GiB、2,493,464 tokens、1M 并发 2.38×；completion 2048 时 C1 82.4，C2 aggregate 98，C3 134.6，C4 120.4 tok/s，C4 TTFT 升至 5.36 s。
- 实际长 prefill 数据中，约 900K prompt 的 TTFT 为 1,028.85 s、prefill 约 874.8 tok/s，并完成 sentinel correctness；这是“实际接近 1M”的重要证据，但它没有给出该深度下固定输出 decode 的严格 A/B。
- [issue #22](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/22) 找到一次严重 dispatch 错误：所谓 `nvfp4_ds_mla` 被送入 BF16 cache path，约 600K 时约 1 tok/s，而 FP8 path 约 17 tok/s。修复是把同一 584-byte layout 走 FP8 consumer；这进一步说明该路径不是完整 compact NVFP4 writer/reader。跟进 A/B 还显示不同镜像的 cache footprint 并不一致，因此 dtype 名字本身不能作为物理布局证据。
- [issue #32](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/32) 在当前默认配置下完成 131,176 和 147,426 prompt tokens（decode 61.26/64.00 tok/s），随后单条 262,086-token 冷请求让 GB10 head 硬重启。没有 OOM-killer/Xid，不能确认根因，但足以否定“配置显示 1M 就已经稳定”的判断。
- [issue #39](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/39) 的受控 A/B 显示一个默认 thinking-budget 补丁把 32K decode 从 14.1 降到 3.0 tok/s；关闭该预算路径恢复 13.8 tok/s。根因是每 decode step、每 speculative row 对全 history 做 Python O(context) 扫描。它是软件路径回退，不是 1M attention 的必然成本。
- 同仓库 regular FULL graph 与 breakable graph 的 matched A/B 中，regular C1 为 95.9 tok/s，相对 breakable 的 74.55 为 +28.6%；C2 aggregate 151.8 对 134.2（+13.1%），约 14K prefill 基本不变。结论不是“breakable 一定更快”，而是要让实际目标 shape 进入稳定 full graph，并把 fallback 可观测。

[r0b0tlab 的双 GB10 native benchmark](https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark) 又提供了两个可移植诊断：其原始 CUDA 13/SM121a、TP2/EP、FULL_AND_PIECEWISE、FP8 KV、MTP2 基线为 C1 38.4 tok/s，65K/C16 为 144.6 aggregate；驱动 `580.142` 升到 `580.159.03` 后 decode 约回退 3.5×，而 raw GEMM 不变，说明 driver/graph launch path 可以制造“所有 kernel 看似正常但 E2E 巨慢”的现象。它还把 DeepGEMM MoE workspace 的 aligned-M 从固定 `local_experts` 收紧为 `min(M*topk, local_experts)`，C1 的 padded M 从 16,384 降到 768（21×）。这两个思路应变成我们的 driver receipt 和 MoE workspace/profile 检查项，不能直接搬 vLLM patch。

DGX Spark 使用 GB10 统一内存和 ConnectX-7；单机官方带宽只有 273 GB/s。6000D 是 84GB GDDR7、约 1.398 TB/s/卡、PCIe、无 NVLink；TP4 聚合显存带宽约 5.6 TB/s，是双 Spark 约 0.55 TB/s 的十倍量级，但 PCIe collective、分片方式和实际 kernel 利用率会吞掉其中一部分。它们因此既不能给出 SM120 绝对速度目标，也明确说明：当前 70 tok/s 远非硬件上限；53 的 107 tok/s 已超过多数公开双 Spark C1，而本集群 r33 的 222 tok/s 才更接近短上下文优化完成态。

### SM12x 构建目标本身也是性能合同

CUDA 13 官方把同一 SM12x 设备拆成三种 feature target：baseline `sm_120`、family-specific `sm_120f`、architecture-specific `sm_120a`。`120f` 可在 SM120/SM121 family 内运行；`120a` 只运行在精确的 12.0 设备。这里必须纠正一个容易误读的边界：PTX ISA 的 feature table 把 FP4/FP6 conversion 和 **dense** block-scaled MXFP4/NVFP4 MMA 放在 `sm_120f`；只有 sparse `mma.sp` 的 `mxf4` / `mxf4nvf4` 变体要求 `sm_120a` / `sm_121a`。[FlashInfer SM121 audit #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170) 已在 2026-06-12 明确修订了早先“所有 MXFP4/NVFP4 MMA 都是 a-only”的错误结论，并确认其 `120f` release wheel 已能发出 native dense FP4 MMA。因此：

- baseline `120` 只能作为“没有 feature-PTX 泄漏”的保守 build gate，不能作为最终 FP4 性能镜像；
- `120f` 应是 SM120/SM121 共用的默认 dense FP4 performance contract；`120a` 只在实际使用 sparse FP4 MMA、其他 a-only 指令，或 SASS/benchmark 证明确有 architecture-specific schedule 收益时作为独立实验臂，不能预设它对 dense FP4 必然更快；
- CUDA 官方提供 `__CUDA_ARCH_FAMILY_SPECIFIC__` 和 `__CUDA_ARCH_SPECIFIC__` 区分 `f` / `a` feature set。涉及 FP4 inline PTX 的源码不应只用 `__CUDA_ARCH__ >= 1200` 判断，否则 baseline `120`、family `120f` 和 arch `120a` 会被错误折叠；
- FlashInfer/CUTLASS 的历史 issue 记录过错误 target、未生成 cubin、TMA tactic 初始化失败和数值错误，但不能把某个旧版本的失败外推成“SM120 不支持 120a”或“120f 没有 dense FP4”。每个 backend 仍必须过数值 reference，并记录 AOT/JIT target、实际 cubin/SASS 和 autotuner 成功/失败 tactic。

### 可迁移到整个 SM120/SM121 家族的优化清单

这次不能只盯 RTX 6000D 私有配方。vLLM、FlashInfer 在 SM120 与 GB10/SM121 上已经把一批共性问题拆得很清楚；这些设计应按 TokenSpeed 的 kernel/runtime 边界重实现，而不是整支搬运：

| 一手来源 | 已验证事实 | 对 TokenSpeed 的可迁移动作 |
|---|---|---|
| [vLLM #40082](https://github.com/vllm-project/vllm/pull/40082) | 已合并 SM120/121 b12x dense FP4 GEMM 和 fused MoE；MoE 把 dispatch、W1、SwiGLU、W2 合进单次调用，dense GEMM 针对 decode 小 M 自适应 tile。双 GB10 的 Qwen3-30B-A3B-NVFP4 A/B 仅为 C1 +1.8%、C8 +6.0%，说明 backend 名称本身不是数量级收益。 | 检查当前 FlashInfer CUTLASS MoE 是否同样做到 fused activation quant、持久 weight layout 和小-M tile；以真实 DSV4 shapes 数值校验与 A/B，不把 Qwen 收益外推。 |
| [vLLM #43477](https://github.com/vllm-project/vllm/pull/43477) | 已合并 DSV4/GLM-5.1 SM120 sparse MLA、decode autotune warmup、DeepGEMM MXFP4 和 grouped-GEMM heuristic。2×RTX PRO 6000、TP2、8K/1K 的原版 DSV4（非 0731）C1 为 target-only 88.1、MTP1 132.4、MTP2 158.5 tok/s；MTP1 acceptance 96.75%/1.97。 | 将 sparse MLA warmup/autotune、grouped-GEMM shape heuristic、spec acceptance/length 纳入基线 receipt。该表不是 exact-1M，MTP 数字也不能替代 0731 DSpark A/B，但证明同类 SM120 上完整 target/spec path 的收益来自组合闭环。 |
| [vLLM #41834](https://github.com/vllm-project/vllm/pull/41834) | 面向 SM120/121 和 `DeepSeek-V4-Flash-0731` 的开放集成分支；删除 C128A metadata 每 step 的 device→host sync，修复 sparse-MLA + speculative prefix block 过早发布，提供 release-dependency fallback 和跨 SM120/121 correctness gate。其 V2/V1 受控 A/B 报告 prefill +1.1–4.3%、KV 容量 +24.9%，也记录了早期重启后 long-recall 随机坏态，最终定位为 prefix/spec cache race。 | 优先审计每-step `.item()`/长度同步、speculative 写入完成前的 prefix 可见性、cache manager 多组一致性；这些是 vendor-neutral runtime 设计。不要搬 187 文件的大分支，也不要保留 release fallback。 |
| [FlashInfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170) | SM121 全栈审计显示 SM120/121 的 dense FP4、XQA/MLA 等能力大量共用，实际缺口常是 `minor == 0`、`family(100)`、AOT module list 或 wheel/JIT target 错配，而非硬件没有指令。 | 全仓检查 `major == 12` 与 `family 120` 应覆盖却被写成 `minor == 0`/SM100-only 的 registry、autotuner、comm 与测试 gate；任何放宽都必须同时补 cubin/JIT 和数值测试，不能只改 Python selector。 |
| [vLLM SM12x arch audit #45260](https://github.com/vllm-project/vllm/issues/45260) | 源码审计同样确认 dense MXFP4/NVFP4 是 `120f` family feature，并把 future sparse FP4、`.s2f6x2`、部分 FP8 `multimem` 列为 a-only 风险。 | build receipt 增加每个扩展的 virtual/code target 与 `cuobjdump` 检查；新增 PTX 前先按 dense/sparse 和 feature macro 分类，不再用“Blackwell”一个布尔值代理。 |

由这些资料归纳出的 SM12x 通用 profiler checklist 是：实际 kernel/backend 选择、AOT/JIT target、autotune tactic、每-step host sync、decode 小-M expert padding、persistent workspace/weight repack、graph 内部断点、prefix/spec cache 发布顺序、collective 拓扑，以及驱动版本。任何一项都可能让 micro-kernel 正常而 E2E 慢或错；它们也比继续换机器更能解释 53 与 59 的稳定差距。

这些新的一手数字也收紧了对当前差距的判断：vLLM #43477 在 **TP2**、8K/1K、target-only 已有 88.1 tok/s，而 53 的 TP4/DCP2 target-only 是约 107 tok/s；59 旧栈 TP4 的约 70 tok/s 甚至低于这个公开 TP2 基线。因而新 TokenSpeed 分支的第一道性能 gate 应是 target-only 至少明显脱离 70，而不是先用 speculative 掩盖 target path。若新栈仍停在约 70，优先证伪的就是 sparse-MLA/MoE 实际 dispatch、`120f` native kernel、decode autotune warmup、每-step host sync 与 graph 内部 break；不是 1M 配置上限，也不是 GPU 算力不足。只有 target-only 接近 53 后，DSpark 才能用 acceptance/step breakdown 解释从约 100 向 150–250 tok/s 的下一段收益。

### 同硬件 TP4 exact-1M 的直接基线

[ambientlight/rtx-pro-6000-bench](https://github.com/ambientlight/rtx-pro-6000-bench) 是目前最接近我们“4×SM120、TP4、实际 1M”的公开数据：4×RTX PRO 6000 Blackwell Max-Q 96GB、每卡 300W、PCIe 无 NVLink、Threadripper PRO 7985WX、512GB RAM。其 [部署文档](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md) 和 [服务配置](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml) 明确为：

- SGLang 自有 SM120 fork，TP4/PP1、`context-length=1,048,576`、max running 16、chunked prefill 8192、page 256、FP8 E4M3 KV、memory fraction 0.90；graph batch 1/2/4/8/16，禁用 custom all-reduce。
- 原生 MXFP4 W4A4 experts、FlashInfer fused MoE、custom HMMA sparse decode/prefill、capture-safe FP8 paged indexer；`SGLANG_SM120_INDEXER_SPLIT=1` 用 256 个 split CTA 扫长 KV。公开配置没有 speculative/DSpark，也没有 DCP，因此它是 target-only FP8-KV 基线。
- 单流实际 prompt、固定 output 1024 的长深度结果：32K 为 TTFT 0.31s / decode 52.9 tok/s；128K 0.79s / 38.7；256K 1.58s / 28.6；512K 3.26s / 18.7；1,047,552 tokens 为 6.46s / **11.1 tok/s**。这说明即使是高度特化 SM120 kernel，实际 1M decode 也会随 KV 扫描显著下降；但只声明 1M、实际短 prompt 不应自动变慢。
- 2K/C64 sustained 756 aggregate tok/s，64K/C16 为 40 tok/s；完整矩阵 8,576/8,576 请求、约 27 小时无失败。它们是高并发 aggregate，不是 C1 数字。

不可比项同样重要：该结果使用原始 `deepseek-ai/DeepSeek-V4-Flash`，不是 0731 revision；engine 和 HMMA/indexer kernel 也是 ambientlight 的 SGLang fork；Max-Q 功耗、主板拓扑与 6000D server 可能不同。因此它适合作为**同芯片 exact-1M 的量级和 kernel 设计参考**，不能直接作为 TokenSpeed 回归阈值。它最值得移植的设计是长 context 的 split-KV indexer，而不是整套 SGLang fork。

### RTX PRO 6000 的 0731 / DSpark 参照

[local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) 是最接近当前硬件的公开一手资料。其服务器通常为 RTX PRO 6000 Blackwell/GB202/SM120、PCIe 5.0 x16、无 NVLink，但不同机箱是 CPU root-port 或 PCIe-switch 拓扑；这会直接改变 TP4/TP8 collective。

| 配置 | 精确条件 | 公开结果 | 不可比项 |
|---|---|---|---|
| [r33](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md) | 4×GPU 主机中的 TP2/TP4，B12X W4A8、FP8 DS-MLA、DCP1、fixed DSpark K5、max 131,072 | TP2 C1/C4/C8 180.6/397.1/580.7；TP4 247.0/541.9/804.5；8K prefill TP2 12,849 tok/s | 不是 1M；full target/draft/context-KV graph；与当前 TokenSpeed target-only/graph coverage 不同。 |
| [Infernal Invocation r4](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-infernal-invocation-r4.md) | TP2/TP4、B12X W4A8、FP8 compressed MLA、DCP1、full graph、TP4 seqs8/max131K | TP2 target-only C1 151.88/C4 397.97；TP4 K5 C8 905.68；coding median 404.40；8K/64K prefill 16,444/16,011 | 多数高吞吐是短上下文或 aggregate；不是 TP4 exact-1M。 |
| [v20 r10](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r10.md) | TP4、max262,144、seqs64、不同 MoE/standard MTP/DSpark 组合 | 代表性 K5 行 C1/C16/C32/C64 298.5/1702.6/2499.6/3454.8；coding 416.7 | 高并发 aggregate，不是单流速度；版本、kernel 和 topology 与当前 stack 不同。 |
| [FlashInfer PCIe IPC qualification #62](https://github.com/local-inference-lab/rtx6kpro/issues/62) | 同机 4×RTX PRO，TP2，B12X vs FI IPC，target-only 与 K5 分开 | 最终 target C1 B12X 129.9 vs FI 126.8；C32 1135.7 vs 1139.5；K5 基本持平，FI prefill 低 6–7%。TP8 topology-scoped B12X 另有 C1/C4/C8 +12/+18/+20.2% | TP2 不存在“换 collective 就必快”；TP8 收益不能外推到 TP4。 |
| [true NVFP4 PR #22](https://github.com/local-inference-lab/rtx6kpro/pull/22) | 开放、未合并；2×RTX PRO、TP2 DSpark；DSV4-native 360 B/token page | 相对 FP8 每 GiB 1.47× KV tokens，decode 约 195→243–262 tok/s，1,030,039 实际 prompt needle recall；262K standing config KV 1,539,217 | 1M 命令为 eager，因为 graph 预留挤压 KV；代码在外部 vLLM/FlashInfer patch series，未进入 rtx6kpro 主线。 |

RTX6KPro 自己的 profiling 给出最重要的性能因果：[DSpark consolidation study](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/dspark-upstream-consolidation.md) 中，单步约 13.6–14.9 ms，target verify 约 11.3 ms（约 83%），draft 约 2.01 ms。清理 drafter kernel 可让 draft 快 10%，但端到端只约 +1.4%；真正的大项是 verify/MoE、acceptance 与 full graph。因此“把 DSpark 开关打开”不等于获得 r33 数字。

最新 [r9 source merge contract issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) 也显示所谓成熟配方依赖一整套可追溯 integration tree：官方 checkpoint、TP2/DCP1、FP8 compressed MLA、fixed probabilistic K5、target/draft/context-KV FULL graph、B12X、CUDA 13.3/PyTorch 2.13，以及多个 correctness PR。其 194.02 tok/s CC1 只是 sanity point，作者明确没有把它当完整性能 sweep。

## vLLM / SGLang 上游给出的可移植设计

这些代码不能跨项目 cherry-pick，但可作为 TokenSpeed 重实现规范：

| 一手来源 | 结论 | 对 TokenSpeed 的动作 |
|---|---|---|
| [vLLM #46995](https://github.com/vllm-project/vllm/pull/46995) DSpark | 8×B300 TP8，target verify 11–13 ms、draft 0.6 ms、sampler 0.6 ms，约 14 ms/step，accepted length 约 5，coding/no-thinking >350 tok/s。 | 移植 full target/draft/sampling graph 和概率 sampler 的设计；绝对数字不适用于 PCIe SM120。 |
| [vLLM #46789](https://github.com/vllm-project/vllm/pull/46789) DSV4 SP | 4×GB200、TP4+EP+SP、FP8 weights/KV、原生 1M；DSpark acceptance 64–67%、accepted length 4.2–4.4。 | 作为 SP/DCP 后续设计参考；先解决 PCIe collective 和 current cache API，不直接搬。 |
| [vLLM #47979](https://github.com/vllm-project/vllm/pull/47979) SM120 PCIe stack | Hy3 FP8 295B TP4（非 DSV4）full stack 相对 NCCL output +8–12%；含 SP/async TP、FULL graph、PCIe barrier/collective。 | 只重实现安全 barrier、拓扑发现和 async-TP 思路，并在 DSV4 同合同 A/B；不能引用 8–12% 作承诺。 |
| [vLLM #48047](https://github.com/vllm-project/vllm/pull/48047) unpadded q heads | DSV4 128 heads 在 TP1/2/4/8/16 直接使用 128/64/32/16/8 heads。 | 检查 TokenSpeed→FlashInfer 调用是否已有原生 TP4 32-head；若已有则无需做。 |
| [vLLM #48993](https://github.com/vllm-project/vllm/pull/48993) compact MXFP4 indexer | GB200 DSV4-Pro，block 数 +6.5%（FI）/+12.05%（FlashMLA），stride -6.10/-10.75%。 | 当前 68-byte MXFP4 indexer cache 已覆盖核心布局，不重复实现。 |
| [vLLM #49486](https://github.com/vllm-project/vllm/pull/49486) all-candidates topk skip | DSV4 TP4、FP8、短 input/output2048；decode 基本中性，标题给出 TTFT 约 3.4%。 | 小优化，放在 graph/DSpark/NVFP4 之后。 |
| [vLLM #50004](https://github.com/vllm-project/vllm/pull/50004) adaptive topk width | DSV4-Pro TP8/1M/FP8，8192 prompt TTFT 373.45→369.42，约 1%。 | 低优先级；先确认 #1076 runtime geometry 未覆盖。 |
| [vLLM #50298](https://github.com/vllm-project/vllm/pull/50298) workspace reuse | B300 combined topk+SWA micro-op 0.093→0.049 ms。 | 当前 #992/#1073 已有持久 workspace，先 profile 后补缺口。 |
| [vLLM #51865](https://github.com/vllm-project/vllm/pull/51865) graph dispatch guard | 只有所有请求确实处于 decode 才进入 uniform graph。 | 在 TokenSpeed 增加 mixed-state graph correctness/telemetry 测试，避免静默错误或回退。 |

SGLang 的 [DCP roadmap #21788](https://github.com/sgl-project/sglang/issues/21788) 与 [refactor #29736](https://github.com/sgl-project/sglang/issues/29736) 说明 DCP 通过 sequence 维分割 KV，从而消除 TP 内相应复制并提高长上下文容量；DeepSeek MLA decode 已有落地，DSV4 prefill CP/Helix 仍在路线图。DP attention 文档声称在 8×H200 高 batch 最多约 1.9× decode throughput，但 [DSV4 DP attention bug #31699](https://github.com/sgl-project/sglang/issues/31699) 仍展示特定配置的错误输出。结论是：DCP/ADP 值得建设，但不是当前 SM120 C1 性能的无风险补丁。

[HiCache 官方设计](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx) 也把边界写得很明确：MLA 在 multi-TP 下每个 rank 的 GPU L1 都持有完整、相同的 KV；HiCache 的 MLA 优化只是让一个 rank 发起 L2/L3 write-back，避免外存重复写入，而且 `hicache-size` 仍是**每 rank**的 host pool。它能做 prefix reuse、分层容量和外存去重，但不会把 TP 内 GPU 的物理 MLA KV 自动均摊。要消除这部分 L1 复制仍需要 DCP/CP，不能把 HiCache、LMCache 一类 external-cache reuse 与 DCP 的 device-resident KV sharding 混为一谈。

## 性能差距的优先级判断

### 1. Target-only 的 expert format、W4A8 MoE 与每步执行

当前 matched A/B 的三边都是 target-only，且 decode 差距从 4K 到 128K 稳定，因此这才是解释“后面都没有 53 快”的首要变量。53 使用 B12X A8，59 旧栈使用 B12X W4A16，新分支使用 FlashInfer MXFP8×MXFP4。新分支已经越过 packed checkpoint loader、autotune、graph gate、`120f` 与离线 tactic 表，4K decode 达到平均 79.97 tok/s，但距 53 仍约 25.4%。下一步应逐层测 shared expert/linear、router、all-reduce、DCP 与 graph 内部 launch；不能用 DSpark 掩盖 target path。

### 2. Speculative/DSpark 是否真的工作

公开 RTX PRO 数据中，完整 K5 能把单流从 target-only 继续推到约 180–300 tok/s，但收益取决于严格 acceptance、accepted length 和 verify step。若 acceptance 接近零、draft/context KV 未进图，DSpark 只增加工作量。

**必须记录：** proposed/accepted tokens、strict acceptance、accepted length、target verify/draft/sample/prep 各阶段 ms、每个 graph 的 hit/fallback 原因。没有这些指标，不能只凭启动参数判断 speculative 已生效。

### 3. CUDA graph 覆盖，而非 eager 与否的单一开关

RTX6KPro 快栈明确 capture target decode、DSpark proposal、sampler/context-KV；Tony/Mia 也显示 graph capacity 和 mixed state 会造成静默 fallback。当前 TokenSpeed 需要的是“哪些 op/shape 不在图中”的可观测性，以及严格失败 gate，而不是保留不可见 eager fallback。

### 4. Cache 的物理格式、consumer dispatch 和长 KV indexer

`fp8_ds_mla`、padded `nvfp4_ds_mla` 和 true compact 360-byte NVFP4 是三种不同合同。错误 dispatch 曾带来约 16× 深上下文回退；真 NVFP4 的外部 TP2 数据又显示 25–34% decode 提升和 1.47× 容量。ambientlight 的 TP4 数据还表明，FP8 KV 下把长 context indexer 拆到最多 256 个 CTA，在 1M 可带来约 1.5× output 提升。当前 TokenSpeed 缺的是完整 writer、page geometry、consumer、scale、graph-safe cache write 和 SM120 split-KV indexer，不是再添加一个 dtype 名字。

### 5. MoE verify 与 PCIe collective

#992 已给出可运行的 SM120 FlashInfer CUTLASS MXFP4 MoE 基座，但外部 profile 表明 verify/MoE 占 DSpark step 的约 83%。现场已经证明 FlashInfer 0.6.16 的 TMA-WS tuning preparation 不可用于 SM120，并用显式、环境绑定的 tactic 表把小-M fused MoE 提升到约 2×；真实 target-only 服务只提升 3.9%，因此 tactic 失配不是剩余全部差距。TP4 collective 需要按具体 PCIe 拓扑 A/B；RTX6KPro 的 TP2/TP4 数据显示收益可能只有 0–3%，TP8 才可能达到两位数。先做 graph 内 step profile，再决定 B12X 思路、RSAG、DCP2 或其他 kernel 路线哪条值得移植。

### 6. DCP/SP/ADP

DCP 按 DCP degree 分摊 sequence/KV，直接改善 1M 容量和长上下文 attention 带宽，但会增加通信和调度复杂度，并不保证单流翻倍。当前优先实现 DCP2 的 correctness/capacity，再测 TP4/DCP2；不要把 ADP 放在第一阶段，也不要在没有长深度 fixed-output benchmark 时宣称性能收益。

## 建议的工程路线

### Phase A：建立唯一可复现基线

1. `main` 只跟随 `upstream/main`；`dev` 是可运行集成分支；每项工作从 `dev` 建 `feat/*`，通过 gate 后回到 `dev`。
2. 固定一个官方 checkpoint revision、镜像/编译器/CUDA/FlashInfer pins、GPU 拓扑、TP/EP/DCP、cache 物理布局、graph mode 和 benchmark corpus。部署产物输出 machine-readable receipt。
3. 先做 `sm_120` build-only、`sm_120f` correctness/profile、`sm_120a` correctness/profile 三臂；最终服务镜像必须记录各 native/JIT kernel 的实际 target 和 FlashInfer autotune tactic，禁止把 baseline 构建冒充 FP4 优化完成。
4. 显式 checkpoint expert-format、SM120 autotune 和 Phase-2 cache-write overlap 已修复，当前 #948+#992 stack 的 **target-only、TP4/DCP1、exact max 1M** 已通过冷启动、短请求和 graph gate。继续完成冷 1K/8K/32K/128K/256K/512K/900K correctness；实际 900K–1M 必须产生首 token并完成固定长度输出，而不是只看 KV capacity 日志。
5. 同一份请求在 53 基线与新 TokenSpeed 上跑；不同 cache/spec/并发的服务只作为不同实验臂，不能互称回归。

### Phase B：完整 1M NVFP4

从 `dev` 建 `feat/sm120-dsv4-compact-nvfp4-kv`，按 DSV4-native 448 NoPE + 128 RoPE 设计 writer/page/scale/reader，复用 `tokenspeed-kernel` 作为唯一 kernel 边界。以 RTX6KPro PR #22 的 360 B/token 布局作参考，但重新实现并验证：

- bit-exact pack/unpack 与 FP8/BF16 reference；
- page boundary、prefix reuse、hybrid cache group、graph replay；
- 1K→1M needle/quality、cold/warm、并发容量；
- 任何 unsupported shape 直接失败，不静默走 BF16/FP8 fallback。

这一阶段的完成定义是“实际 1M correctness + 固定 decode + 可重复重启”，不是先追求最高吞吐。

### Phase C：把 DSpark 从 Week-0 变成 SM120 性能能力

从 `dev` 建 `feat/sm120-dsv4-dspark-fullgraph`：

1. 概率 K5 sampler、per-request draft/context-KV 状态和 ragged batch 语义；
2. target verify、proposal、sampling、context-KV 全图 capture；
3. 固定 graph capacity，不允许并发超过 capture 上限后静默 eager；
4. acceptance/accepted length/step breakdown 作为服务指标和 benchmark 必填项；
5. FP8 与 compact NVFP4 两个 cache consumer 都要通过相同 gate。

这是预期吞吐收益最大的 feature，但只有在 acceptance 和 full graph 同时达标时才启用默认。

### Phase D：SM120 PCIe 与并行策略

1. 对实际 6000D 主机采集 `nvidia-smi topo -m` 和 P2P read/write/atomics，按 root-port/switch 岛构建拓扑合同。
2. 同合同 A/B NCCL、当前 RSAG/FlashInfer 和 topology-scoped protocol；先 TP4，再考虑 TP8。禁止把 TP8 +20% 外推到 TP4。
3. 当前 DCP2 prototype 已证明 exact-1M 容量 receipt、2-GPU graph primitive 和 4095-token 短请求结果一致，但 4095/256 C1 只有 46.9/47.0/47.8 tok/s。下一步按本报告 A--E 路线先消除静态/重复 collective 与 LSE 小 kernel，再测 C1/C4 和实际 256K/512K/900K decode；DCP4 和 SP 后置。
4. ADP/DP attention 仅在高并发、KV capacity 确实是瓶颈时进入路线；它不解决 DSpark verify/MoE 的 C1 主耗时。

## 推荐优先级：cherry-pick 与重实现

| 顺序 | 项目 | 方式 | 预期价值 | 风险/备注 |
|---:|---|---|---|---|
| 0 | 闭环 #948+#992 的 SM120 build/dispatch | 当前独立 feature 分支修复 | 得到可构建、数值正确且能确认快 kernel 的 SM120 基座 | 必须分别测 120/120f/120a；#992 当前只有 smoke。 |
| 1 | 真 compact NVFP4 DS-MLA KV | TokenSpeed 原生重实现 | 1M 容量闭环；外部 TP2 显示 1.47× capacity、25–34% decode 潜力 | 外部 PR 未合并；必须先 correctness。 |
| 2 | DSpark K5 + full target/draft/context graph | 基于 #940/#1048 重构 | 最大的 C1/aggregate 潜力 | acceptance 低时会负收益。 |
| 3 | strict graph coverage + telemetry | 重实现 #714/#51865 的设计 | 消除隐藏回退，稳定性能 | 不以“eager 能跑”作为完成态。 |
| 4 | SM120 split-KV indexer + #1026 wide-compress A/B | 在当前 kernel API 重实现/扩展 gate | actual 512K–1M decode；ambientlight 1M indexer 约 1.5× | 先做 microbench 和长深度 E2E，不外推 SGLang 数字。 |
| 5 | MoE/verify 与 topology-scoped collective | profile 后选择性重实现 | TP4 可能个位数，TP8 可能两位数 | 极度依赖拓扑，不能整包搬 B12X。 |
| 6 | DCP2 / SP | 基于 #364 primitive 新建 runtime feature | 长上下文容量和带宽 | 仍需 subgroup、KV shard 和 O/LSE merge；通信可能损害低并发。 |
| 7 | 小型 DSV4 优化 | 选择性移植 vLLM #49486/#50004 等设计 | 单项约 0–4% | 避免在主缺口未解决前堆 patch。 |
| 8 | #997 P/D、#620 L2 KV | 有明确业务 workload 后再集成 | prefix/P-D 吞吐 | 不解决当前 raw decode gap。 |

## 必须采用的交叉 benchmark 合同

### 实验臂

至少包含：

1. target-only + FP8 DS-MLA + TP4/DCP1；
2. target-only + compact NVFP4 + TP4/DCP1；
3. DSpark K5 full graph + FP8 + TP4/DCP1；
4. DSpark K5 full graph + compact NVFP4 + TP4/DCP1；
5. 上述胜者的 TP4/DCP2；
6. 53 当前快服务的完全冻结基线。

### 横纵维度

- 实际 prompt：1K、8K、32K、128K、256K、512K、900K/接近 1M；不能以 `max_model_len` 代替实际深度。
- concurrency：C1、C2、C4、C8，超过实际 KV capacity 的格子标 `N/A`，不缩短 prompt 偷换口径。
- 固定 output：256 和 2,048 tokens；decode 从 first streamed token 开始计算，另报 end-to-end。
- cold unique prefix 与 warm prefix 分开；每次 cold payload 使用随机 nonce，避免被 prefix cache 命中。
- 至少 1 次 warmup + 5 次 measurement，报告 median、p5/p95 和失败率；1–4% 差异必须复测。

### 每格必报指标

- TTFT、prefill tok/s、ITL/TPOT、单 active-user decode tok/s、aggregate output tok/s；
- proposed/accepted tokens、strict acceptance、accepted length；
- target verify/draft/sample/prep ms、graph hit/fallback 次数与原因；
- KV layout 的真实 bytes/token/layer、KV pool tokens、峰值显存、prefix hit；
- P2P/collective backend、拓扑、TP/EP/DCP、CUDA graph capture sizes；
- engine crash、OOM、Xid、timeout、错误输出和 needle/quality 结果。

只有满足这套合同，才能回答“53 为什么更快”和“SM120 最优组合是什么”。依据当前一手证据，最值得先验证的目标组合是：**TP4/DCP1 + #992 SM120 sparse MLA/MXFP4 MoE + 真 compact NVFP4 KV + fixed probabilistic DSpark K5 + target/draft/context-KV strict FULL graph**。DCP2 已经是可运行的 1M 容量/correctness 实验臂，但当前 4095/256 的约 47 tok/s 说明它还不是性能完成态；在 A--E 通信路线和实际长深度矩阵闭环前，不应让 DCP2 独自承担所有吞吐预期。

## 资料索引

- TokenSpeed：[#947 SM120 tracking](https://github.com/lightseekorg/tokenspeed/issues/947)、[#992 SM120 DSV4](https://github.com/lightseekorg/tokenspeed/pull/992)、[#940 DSV4/DSpark](https://github.com/lightseekorg/tokenspeed/pull/940)、[#1048 DSpark CUDA graph](https://github.com/lightseekorg/tokenspeed/pull/1048)、[#1025 prefix replay](https://github.com/lightseekorg/tokenspeed/pull/1025)、[#364 DCP kernel primitive](https://github.com/lightseekorg/tokenspeed/pull/364)、[#714 prefill graph](https://github.com/lightseekorg/tokenspeed/pull/714)、[#555 mixed prefill/decode](https://github.com/lightseekorg/tokenspeed/pull/555)、[#563 host-sync ideas](https://github.com/lightseekorg/tokenspeed/pull/563)、[#1026 wide compress](https://github.com/lightseekorg/tokenspeed/pull/1026)、[#620 L2 KV](https://github.com/lightseekorg/tokenspeed/pull/620)、[#997 P/D handoff](https://github.com/lightseekorg/tokenspeed/pull/997)、[#1057 unified P/D cache contract](https://github.com/lightseekorg/tokenspeed/pull/1057)。
- RTX PRO 6000：[ambientlight exact-1M benchmark](https://github.com/ambientlight/rtx-pro-6000-bench)、[TP4 config](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/bench/deepseek-v4-flash_W300_TP4_sglang/sglang-single.yaml)、[SM120 deploy doc](https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md)；[rtx6kpro](https://github.com/local-inference-lab/rtx6kpro)、[r33](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md)、[Infernal Invocation r4](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-infernal-invocation-r4.md)、[true NVFP4 PR #22](https://github.com/local-inference-lab/rtx6kpro/pull/22)、[PCIe IPC qualification #62](https://github.com/local-inference-lab/rtx6kpro/issues/62)、[source contract #67](https://github.com/local-inference-lab/rtx6kpro/issues/67)。
- DGX Spark：[tonyd2wild 1M NVFP4](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)、[MiaAI-Lab DSpark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)、[Mia #22](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/22)、[Mia #32](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/32)、[Mia #39](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/39)、[r0b0tlab native SM121 benchmark](https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark)。
- CUDA/SM12x：[CUDA feature-set targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)、[`f` / `a` feature macros](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)、[PTX ISA FP4 target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/)、[FlashInfer SM121 audit #3170（含 dense/sparse 修订）](https://github.com/flashinfer-ai/flashinfer/issues/3170)、[FlashInfer SM120 NVFP4 MoE issue #2723](https://github.com/flashinfer-ai/flashinfer/issues/2723)、[vLLM SM12x arch audit #45260](https://github.com/vllm-project/vllm/issues/45260)、[vLLM SM120 CUTLASS NVFP4 #33417](https://github.com/vllm-project/vllm/pull/33417)。
- 跨项目设计：[vLLM SM12x b12x #40082](https://github.com/vllm-project/vllm/pull/40082)、[DSV4/GLM SM120 #43477](https://github.com/vllm-project/vllm/pull/43477)、[DSV4 SM12x integration #41834](https://github.com/vllm-project/vllm/pull/41834)、[#46995](https://github.com/vllm-project/vllm/pull/46995)、[#46789](https://github.com/vllm-project/vllm/pull/46789)、[#47979](https://github.com/vllm-project/vllm/pull/47979)、[#48047](https://github.com/vllm-project/vllm/pull/48047)、[#48993](https://github.com/vllm-project/vllm/pull/48993)、[#49486](https://github.com/vllm-project/vllm/pull/49486)、[#50004](https://github.com/vllm-project/vllm/pull/50004)、[#50298](https://github.com/vllm-project/vllm/pull/50298)、[#51865](https://github.com/vllm-project/vllm/pull/51865)；SGLang [DCP roadmap #21788](https://github.com/sgl-project/sglang/issues/21788)、[#29736](https://github.com/sgl-project/sglang/issues/29736)、[HiCache design](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx)、[DSV4 DP attention #31699](https://github.com/sgl-project/sglang/issues/31699)。

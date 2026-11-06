[English](README.md) | [简体中文](README.zh-CN.md)

# lfm3-1.7b-mi50

用 Go 实现的 [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) 单流推理，运行在 AMD Radeon Instinct MI50（gfx906）上。

产物是一个静态 ELF 文件，里面包含量化权重、分词器和全部 GPU 内核，并提供 OpenAI 兼容的 HTTP 接口。运行时不需要 Python、ROCm 或 libc，只需要内核自带的 amdgpu/KFD 驱动。

它用来替代同一张卡上同一模型的 vLLM 部署，两边使用相同的 DSpark kernel 和策略：

| 上下文 | TTFT，vLLM → 本项目 | greedy decode tok/s，vLLM → 本项目 |
|---|---|---|
| 1k | 0.201 → 0.124 s | 974 → 1207 |
| 32k | 5.829 → 4.617 s | 364 → 613 |
| 128k | 44.738 → 28.739 s | 301 → 307 |

TTFT、prefill、热启动和采样路径在每一档都更好。prefill attention 的 PV 用 int8 计算，softmax 概率量化为 u8，速度是 fp16 版本的两倍；代价是长上下文的精度略有下降，`--exact-attention` 可以换回 fp16 kernel（见[正确性](#正确性)）。上表的 decode 是 3 条流的中位数，**受混沌影响很大，不能单独看**：72 条流的总体基本打平（均值 559 → 545），把落入重复循环的流分开后，正常文本每一档快 5% 到 16%。详见[评测](#评测)，优化全过程见[优化报告](docs/OPTIMIZATION_REPORT.md)。

只有 `temperature: 0` 的请求走 DSpark。模型默认的 temperature 是 0.1，走采样路径，约 260 tok/s。这和 vLLM 行为一致。

## 快速开始

```bash
# 1. 下载权重（校验和见"模型权重"）
rev=ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7
mkdir -p model
for f in config.json tokenizer.json model.safetensors; do
  curl -L -o "model/$f" "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/$rev/$f"
done

# 2. 构建 dist/lfm3-1.7b-mi50（约 2 GB）。drafter 目录见"模型权重"
./build.sh model drafter

# 3. 运行。访问 /dev/kfd 需要 root 或 render 组
sudo ./dist/lfm3-1.7b-mi50 --port 18000

# 4. 调用
curl localhost:18000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Reply briefly."}],"temperature":0}'
```

可执行文件是自包含的，可以直接复制到另一台装有同样显卡的机器上运行。

## 环境要求

**运行**

| | 测试环境 | 说明 |
|---|---|---|
| GPU | AMD Radeon Instinct MI50 32 GB（gfx906） | 唯一支持的设备，启动时检查 KFD 拓扑 |
| 内核 | Linux 7.2.11（Debian），内核自带 amdgpu | KFD ioctl ABI 1.x |
| 权限 | root | 或者加入 `/dev/kfd` 所属的组（未测试） |
| 空闲显存 | 32 GB，卡上没有其他任务 | 默认 131072 token 上限时至少需要约 7 GB，不足时拒绝启动 |
| CPU | Intel i5-1340P | 处理请求时有一个核心在自旋 |
| 内存 | 16 GB | 启动时峰值约 7.3 GB（含文件映射），运行时约 1.8 GB |

**构建**

| | 版本 | 说明 |
|---|---|---|
| Go | 1.27（测试用 1.27.3） | 无第三方模块，`CGO_ENABLED=0` |
| 模型文件 | 见下文 | 打包时内存峰值约 8 GB（含文件映射），16 线程约 10 秒 |

**修改 GPU 内核**（只有这时才需要）

| | 版本 | 说明 |
|---|---|---|
| Docker 镜像 | `rocm/dev-ubuntu-24.04:7.2.4` | 运行 `kernels/build.sh`，不需要 GPU，可用 `HIP_IMAGE` 覆盖 |

编译好的内核已经提交在 `kernels/*.co`，所以普通构建既不需要 ROCm，也不需要 Docker。用上面的镜像重新编译，得到的指令完全相同。

## 模型权重

权重不在本仓库里。请从 Hugging Face 下载代码验证时使用的确切版本：

| 文件 | 大小（字节） | SHA-256 |
|---|---|---|
| `model.safetensors` | 3457635448 | `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `tokenizer.json` | 17908635 | `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701` |
| `config.json` | 1454 | `8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d` |

版本号：`ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`。可以用"快速开始"里的 `curl` 循环下载，也可以用 Hugging Face CLI：

```bash
hf download LiquidAI/LFM3-1.7B model.safetensors config.json tokenizer.json \
  --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 --local-dir model
```

`cmd/pack` 在构建时量化一次权重。输出可以逐字节复现，且在所有比对过的矩阵上都与原 Python 量化实现一致。

DSpark drafter 的权重也不在本仓库里，目前没有公开发布，代码验证时用的是训练中途的检查点 `lfm3-1.7b-dspark-l4b44-step10260`：

| 文件 | 大小（字节） | SHA-256 |
|---|---|---|
| `model.safetensors` | 519113708 | `7a1e6d24a916b0814fd106b6194cdcd309eec7919f013d89798896d84efbdf0f` |
| `config.json` | 1855 | `930d3e014c73ec67f2fa66e782997dc779e2a0bd5195e5c508f76a69b0e8fc37` |

head 裁剪用的 16384 个高频 token 已提交在 `cmd/pack/crop_ids.bin`。换权重只需要 `./build.sh model <新目录>`（先删除 `build/payload.bin`），但策略要重新回放。

权重采用 LFM Open License v1.0，对超过一定收入门槛的商业使用有限制。构建出的可执行文件包含由权重派生的数据，分发前请先阅读该许可证。

## 技术路线与理由

### 直接对接 KFD，不链接 ROCm

`internal/kfd` 约 1100 行 Go 代码，负责打开 `/dev/kfd`、分配和映射显存、创建 AQL 队列、加载 code object、提交 packet。

其他方案不符合目标。HIP 和 ROCr 是庞大的 C++ 动态库，处理 code object 时还要通过 comgr 引入 LLVM，基于它们做单一静态二进制并不现实；用 cgo 也会失去 Go 的静态构建优势。而本项目只用到其中很小一部分：一个队列、一个设备、预编译的内核。

代价是驱动的适用面很窄：只支持这一款 GPU，不提供 scratch 内存（这里的内核都用不到），也没有 hostcall。为了确认行为正确，我记录了 ROCr 实际发出的 ioctl 和 AQL packet，并逐字段比对。

### 使用 Go，不用 cgo

`CGO_ENABLED=0` 默认就能产出静态可执行文件，标准库覆盖了 HTTP 和 JSON，所以 `go.mod` 没有任何依赖。GC 也不会造成影响：token 循环的内存分配很少，所有张量都在 Go 堆之外。

### GPU 内核保留 HIP 并预编译

Go 无法编译到 GPU，现有内核又经过大量调优，所以继续用 HIP C++，以 gfx906 code object 的形式通过 `go:embed` 嵌入：

- `dspark.hip`：DSpark megakernel。一次 launch 在 GPU 上完成 draft、verify 和 commit，最多 32 轮。greedy 请求走这里。
- `decode.hip`：decode megakernel。一次持久 launch 跑完 24 层和 LM head，一次一个 token。采样、logprobs 和 `min_tokens` 的请求走这里。
- `prefill_attn.hip`：基于分页 int8 KV cache 的 flash attention，QK 和 PV 都用 int8（softmax 概率按每 64 个 key 量化为 u8）。
- `prefill_attn_exact.hip`：上一版 fp16 概率的 flash attention，供 `--exact-attention` 使用。
- `gemm.hip`：prefill 所有线性层的 W8A8 矩阵乘法。
- `ops.hip`：新写的。原生实现 vLLM 里走 torch 或 Triton JIT 的算子：embedding、RMS norm、因果卷积、Q/K norm 与 RoPE、int8 KV 写入、gate、fp16 LM head。

运行时不做任何编译。模型启动约 7.5 秒就绪，vLLM 部署约需 22 秒。

编译好的内核（约 420 KB）提交在仓库里，这样只用 Go 工具链就能构建项目。

Go 驱动没有 scratch 内存，所以 `dspark.hip` 必须用 `-mllvm -disable-machine-licm` 编译，并保持 `private_segment_fixed_size` 为 0。

### DSpark 写进 megakernel

decode 的钱主要花在读权重上，一次验证 M 个 token 也只读一遍：8k 上 M=1 每轮 3.73 ms，M≈4 只要 4.06 ms。所以让 3 层的 drafter 先猜，再一次验证，`tok/s = τ / R`。draft 和 verify 每秒交替 200 多次，回 CPU 太贵，全部写在一个 kernel 里，CPU 只把提交的 token 发出去。被拒的 token 不需要回滚：卷积历史放在按位置索引的环里，K/V 写在临时页，提交时才拷进 KV cache。

和 vLLM 版本相比多了一点：verify 可以越过 32 token 的 block 边界，边界处的卷积状态由 kernel 从历史环里写一份。回放增益 0.6% 到 1.3%。

策略（8k 以下最多 7 个 draft、32768 以上最多 3 个、置信度阈值 −0.4）由固定序列回放选出，不是看基准数字选的，原因见[评测](#评测)。

### 把权重附加在 ELF 之后

模型数据约 1.9 GB，而 Go 链接器把数据段限制在约 2 GB，留给程序的余量太小，所以不用 `go:embed`。`cmd/pack` 把数据附加到构建好的可执行文件末尾，并写入一个定位用的 footer，程序启动时以只读方式映射自身文件。

权重分块做熵编码（标准库的 Huffman），启动时并行解码到内存，不写磁盘。需要注意的是 `strip` 这类工具会丢掉这些数据，所以 `build.sh` 在附加数据之前就先去掉了 Go 二进制的符号。

### 只存一种权重布局，另一种现算

decode 和 prefill 用的是相同的 int8 数值，只是布局不同。这里只存 decode 布局，prefill 布局在启动时生成，这样可执行文件能少约 1.5 GB。

同样不存的还有 LM head（就是量化后的 embedding，260 MB）、drafter 的 head 裁剪和 k/v 投影（量化是逐行独立的，直接抄已有的行）。熵编码对 int8 权重只省 7%，因为它们接近随机噪声；对 embedding 省 34%，因为 BF16 转 fp16 后低 3 位全是 0。结果是多装了一个 drafter，文件反而从 2.3 GB 降到 1.97 GB。

### 为单流设计的前缀缓存

full attention 的 KV 块按 LRU 淘汰。sliding attention 块和卷积检查点放在环形缓冲里，因为命中时只需要最近一个窗口的数据。命中点和 vLLM 部署一样落在 32 token 的 block 边界上，比对的 18 个请求命中数全部相同。`cache_salt` 用于隔离请求。

### 手写分词器和 chat template

模型的预分词正则需要 lookahead，而 Go 的 `regexp` 不支持，所以这部分切分是手写的扫描器。chat template 是对模型 Jinja 文件的直接翻译，没有用模板引擎。两者都用从参考部署抓取的输出做了校验（`testdata/`）。

### 没有实现的内容

范围只包括这张卡上的单流推理。结构化输出（需要语法引擎）、多模态输入和批处理都没有实现。

## 必须保证正确的几点

这些都是实际调试才发现的，会约束后续改动，代码注释里也有对应说明。

- **dispatch 使用 agent scope 的 fence，和 ROCr 一样。** 推理路径上没有地方需要 system scope 的 acquire 或 release，下一条是它的代价。唯一的 system scope barrier 在 `cmd/diag` 里，用于从显存读回结果。
- **主机读取的数据都不需要回写 L2。** token 和 logits 直接写入不经缓存的主机内存，并用标记确认到达。每个 token 回写一次 L2 最多要多花 0.9 ms，去掉它是 decode 比 vLLM 快的主要原因：两边内核耗时相同，而驱动每个 token 只增加约 0.04 ms。
- **运行中会变化的数据必须由 GPU 写入。** 主机通过 BAR 写入会绕过 L2，L2 可能继续提供旧值，所以 `Engine.upload` 和 `Engine.loadToVRAM` 都经由 GPU 拷贝。
- **doorbell 和 HDP 寄存器要用普通写入。** Go 的 `atomic.Store` 在 amd64 上是带锁的 `XCHG`。doorbell 的值是最后一个 packet 的 id，不是写指针。
- **decode 内核曾有一处竞争，已修复。** 它在等待 grid barrier 之前就预取 `h_old`。这对其他层都安全，但第 0 层的 `h0` 正是上一阶段由 block 0 写入的，其他 block 可能读到上一个 token 的 embedding，导致整个 block 算错。在这个驱动下，它在超过 16 个 block 时出现。问题依赖时序：未修复的内核在 ROCr 下能通过同样的逐位比较测试，原因没有查明。修复只需要改 `decode.hip` 的一行。在 ROCr 下它使内核时间增加约 0.6%，33 个测评请求的前 256 个 greedy token 全部不变，所以 vLLM 与 HF 一致度较低并不是这个竞争造成的。
- **acquire VM 之前要等一下。** 打开 render 节点会在内核里启动异步的页表更新。如果 `ACQUIRE_VM` 在约 1 毫秒内紧跟其后，这些更新会覆盖 KFD 为 trap handler 写入的页目录项。缺了这个映射，进程仍能正常运行，直到某个正在运行的 wave 被抢占，而任何进程的队列发生变化都会引发抢占：wave 取不到 trap handler，队列随之卡死。ROCr 两次调用之间的间隔足够长，所以不会遇到。这里的表现是在 vLLM 旁边启动时偶尔卡住，因为启动期间主机内存的 compaction 让 KFD 停止并重启了 vLLM 的队列。通过 debugfs 遍历页表看到该项缺失，加入 1 毫秒延迟后恢复正常。由于没有 ioctl 能等待这些更新，`kfd.Open` 会等 200 毫秒。
- **prefill attention kernel 同样不能溢出。** 每个 block 512 线程，VGPR 上限 128，而驱动没有 scratch 内存。编写过程中有三种编译器行为造成溢出，代码里分别做了规避：barrier 之前算好的值被挪到 barrier 之后的使用处，导致所有已加载的操作数一直占着寄存器（logits 和累加器用空 `asm` 钉住）；两个分支里相同的 barrier 让编译器把两组 wave 的循环合并（两组的 barrier asm 不同）；同一对加载值的两半在两个条件下分别写出，被合并成一次写入并写错了一半（改为每个线程只加载一个 dword 到一个变量）。`kernels/build.sh` 遇到溢出或 scratch 会直接失败。
- **不要占满显存。** 显存不足时，内核会把其他进程的缓冲区驱逐到主机内存，可能触发 OOM-killer。所以程序启动时先检查空闲显存，保留 512 MiB 余量，不足就拒绝启动，不会挤占别的进程。

## 接口

`/v1/completions`、`/v1/chat/completions`（支持流式和非流式）、`/v1/models`、`/tokenize`、`/detokenize`、`/health`、`/reset_prefix_cache`。

支持：thinking（`reasoning` 字段）、工具调用（Qwen3 XML 格式，按工具 schema 转换参数类型）、`cache_salt`、`logprobs`、`stop`、`stop_token_ids`、`ignore_eos`、`min_tokens`、`logit_bias`、`allowed_token_ids`、`seed`、`return_token_ids`、`stream_options`。默认采样参数来自模型的 `generation_config.json`：temperature 0.1、top_k 50、repetition_penalty 1.1。

以下参数会返回 HTTP 400：`response_format`、`structured_outputs`、`guided_*`、`tool_choice` 设为 `required` 或指定函数、`n > 1`。

和 vLLM 接口的差异：

- 流式工具调用的参数按完整参数逐个发送，不是逐字符发送，拼接后结果相同。
- 流式响应最后多一个携带 `finish_reason` 的空 chunk。
- `seed` 可以复现结果，但随机数序列和 torch 不同。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host`、`--port` | `0.0.0.0`、`18000` | 监听地址。服务没有鉴权，在不可信网络上请限制 `--host` |
| `--max-model-len` | `131072` | prompt 加输出的上限（token） |
| `--gpu-memory-utilization` | `0.90` | 可用的显存比例，KV 池占用余下部分 |
| `--kv-tokens` | 按显存计算 | full attention KV 容量（token） |
| `--payload` | 可执行文件本身 | 从单独的 payload 文件读取模型（开发用） |
| `--no-draft` | 关 | greedy 请求不走 DSpark，用于对比 |
| `--exact-attention` | 关 | prefill attention 用 fp16 softmax 概率：attention 速度减半（128k TTFT 44.4 s，默认 28.7 s），长上下文精度回到上一版 |

## 项目结构

```
build.sh           构建 dist/lfm3-1.7b-mi50
cmd/lfm3-1.7b-mi50      服务主程序
cmd/pack           量化权重和 drafter、编译分词器、附加 payload（构建时用）
cmd/bench          测评客户端，可用于任何 OpenAI 兼容服务
cmd/diag           GPU 检查：各路径的确定性、内核时间与端到端时间
internal/kfd       纯 Go 的 KFD 驱动：ioctl、显存、AQL 队列、code object 加载
internal/model     权重布局、prefill 流水线、decode、DSpark、前缀缓存、生成循环
internal/sample    采样器，操作顺序与 vLLM 相同
internal/tokenizer BPE 分词器
internal/chat      手工翻译的 chat template
internal/server    HTTP 接口、reasoning 与工具调用解析
kernels/           GPU 内核：.hip 源码和编译好的 gfx906 code object
testdata/          从参考部署抓取的分词、模板和接口样例
benchmarks/        测评原始输出
docs/              优化报告
```

## 评测

**环境。** MI50 32 GB（gfx906，60 CU），Intel i5-1340P，16 GB 内存，Linux 7.2.11。基线是 vLLM（ROCm 7.2.4），使用相同的模型、相同的 DSpark kernel、权重 step 10260 和相同的策略。每次测试都独占 GPU。所有数字都是 `cmd/bench` 测得的绝对值，原始输出在 `benchmarks/`。

### 性能

单流、流式响应、输出 1024 token、冷缓存、temperature 0。请求来自 Open-SWE-Traces 的真实 agent 轨迹，以 token id 发送。每个数值取 3 个会话的中位数。

| 上下文 | 输入 token | TTFT vLLM | TTFT 本项目 | prefill tok/s vLLM | prefill tok/s 本项目 | decode tok/s vLLM | decode tok/s 本项目 |
|---|---|---|---|---|---|---|---|
| 1k | 1024 | 0.201 s | 0.124 s | 5090 | 8230 | 974.2 | 1206.8 |
| 8k | 8192 | 1.157 s | 0.983 s | 7082 | 8330 | 532.3 | 954.4 |
| 32k | 32768 | 5.829 s | 4.617 s | 5622 | 7096 | 363.5 | 612.7 |
| 64k | 65536 | 15.574 s | 11.187 s | 4208 | 5858 | 412.8 | 449.6 |
| 128k | 126976 | 44.738 s | 28.739 s | 2838 | 4418 | 301.0 | 307.4 |

TTFT 和 prefill 来自 `benchmarks/lfm3-1.7b-mi50_cold.txt`，是换成 int8 PV 的 prefill attention 之后测的。decode 两列沿用之前那次（`benchmarks/lfm3-1.7b-mi50_cold_exact-attention.txt`，和 `--exact-attention` 的结果相同）：decode kernel 没有变，重新跑一次的 greedy 中位数会随下文说的重复循环"抽签"而变。attention kernel 本身在 128k prompt 上快 2.06 倍（全部层合计 32.6 s → 15.9 s），见 `benchmarks/prefill_attention.txt`。

多轮对话回放，前缀缓存已预热（3 个会话，每个 3 轮），取 9 个请求的中位数，换 prefill attention 之前测：

| 档位 | TTFT vLLM | TTFT 本项目 | decode tok/s vLLM | decode tok/s 本项目 |
|---|---|---|---|---|
| 32k | 1.021 s | 0.913 s | 436.4 | 447.1 |
| 128k | 1.359 s | 1.172 s | 331.0 | 335.7 |

18 个请求的命中 token 数都和 vLLM 相同。

采样路径（temperature 0.6、top_p 0.95、top_k 20，不走 DSpark），换 prefill attention 之前测：

| 档位 | decode tok/s vLLM | decode tok/s 本项目 |
|---|---|---|
| 8k | 191.1 | 260.6 |
| 128k | 150.9 | 189.1 |

### 为什么 decode 的中位数不能单独看

`tok/s = τ / R`。R 可以比，单条流的 τ 不能比：M 不同，浮点取整就不同，近似打平的位置换一个 token，后面的文本和 τ 就全变了。这些 prompt 是截断的轨迹，又带 `ignore_eos`，模型跑过自然结束点后常常掉进循环，这在 vLLM 和无 draft 路径上都有。好猜的循环能到 1300 tok/s，难猜的只有 250，哪条流掉进哪种循环就是抽签。一条回放里基本中性的策略改动，能把上表 1k 从 1207 变成 706，128k 从 307 变成 493。

所以另外测了 72 条流（同样的 prompt 去掉开头 16·v 个 token），把前 256 个 token 中重复 8-gram 比例超过 0.3 的流分开：

| 档位 | 流数 | 均值 vLLM → 本项目 | 中位数 vLLM → 本项目 | 非循环流均值 vLLM → 本项目 | 循环流均值 vLLM → 本项目 |
|---|---|---|---|---|---|
| 1k | 18 | 767 → 614 | 762 → 411 | 438 → 481 | 1031 → 747 |
| 8k | 18 | 595 → 622 | 489 → 471 | 469 → 499 | 794 → 775 |
| 32k | 18 | 498 → 557 | 449 → 504 | 417 → 455 | 709 → 760 |
| 64k | 9 | 407 → 443 | 414 → 439 | 406 → 471 | 408 → 408 |
| 128k | 9 | 347 → 334 | 301 → 308 | 317 → 334 | 452 → 无 |
| 全部 | 72 | 559 → 545 | 456 → 436 | | |

**总体打平，1k 的均值和 8k 的中位数低于 vLLM，这些差异都来自循环流的抽签；非循环流每一档都更快（+5% 到 +16%）。** 策略用固定序列回放选，不用这些数字选。

### 正确性

MATH-500 和 GPQA-Diamond 各随机 20 题，thinking 开启，max_tokens 32768。题目相同，都是真实的长输出（平均 4000 到 12000 token）：

| 路径 | MATH-500 正确率 | MATH decode tok/s | GPQA 正确率 | GPQA decode tok/s | 循环 | 截断 |
|---|---|---|---|---|---|---|
| 本项目，greedy，DSpark | 95% | 639.5 | 50% | 507.5 | 0 | 0 |
| vLLM，greedy，DSpark | 90% | 616.0 | 50% | 504.6 | 0 | 2 |
| 本项目，greedy，不用 draft | 90% | 293.2 | 50% | 290.5 | 0 | 1 |
| 本项目，采样（temperature 0.6） | 95% | 262.8 | 65% | 262.6 | 0 | 0 |

| 检查项 | 结果 |
|---|---|
| 与 HF fp32 greedy 的前缀一致（repetition_penalty 1.0，5 个提示各 48 token） | 48、48、17、31、48 / 48 |
| 分词器和 chat template，47 组抓取样例 | 逐 token 一致 |
| 权重，6 个矩阵及 embedding、norm 与 Python 量化实现比对 | 逐字节一致 |
| 确定性，prefill 和 decode 内核 | 多次运行 logits 逐位相同 |
| 确定性，DSpark，15 个评测请求 | 不同运行的前 256 个 token 全部相同 |
| 抢占下的确定性，2000 步 decode，其间 130 个其他进程创建队列 | 逐位相同，无 GPU 故障（测于接入 DSpark 之前） |

**prefill attention 的 softmax 概率是 u8。** P' = p × V scale 按每行、每 64 个 key 的 tile 相对 tile 内最大值量化，PV 用 `v_dot4` 计算。用 HF fp32 模型模拟 kernel 的量化方式，最后 192 个位置的下一 token 分布对 fp32 的 KL：16k 不变（0.00025），60k 从 0.0171 升到 0.0251（两个会话的均值；top-1 一致率 96.1% → 93.0%）。作为参照，较难的那个会话里仅 int8 KV cache 本身就有 0.031。旧 vLLM 部署按整行量化 P，导致长输出崩坏（KL 4.8）；按 tile 量化不会。`--exact-attention` 以一半的 attention 速度换回 fp16 P。细节见 `benchmarks/prefill_attention.txt`。

**GEMV 的 x 是 int8。** 这是最早一处用精度换速度的决定：每轮快 8% 到 17%，代价是 KL 约为 fp16 的 6 倍（0.0028 vs 0.0004），top-1 在 119 个位置上多翻转 1 到 4 处，都发生在第一名和第二名接近的位置。fp16 回退路径需要 scratch 内存，Go 驱动没有。

解读时需要注意：

- 20 题的抽样误差约 ±10 个百分点，随机数序列也不同，所以这只能说明没有精度崩坏，不能说明更准。
- greedy 输出和 vLLM 并非逐 token 相同，和本项目的 `--no-draft` 路径也不相同。分叉的位置不固定，通常在前 100 个 token 内。
- 逐轮比较 draft：同一 prompt、同一前缀，72 个对齐轮次中 60 个的 M 和 draft 与 vLLM 完全相同。

### 没有测的部分

- 8k 以上上下文与 HF 的一致度。CPU 基准在这个长度上太慢，长上下文只抽查了输出。
- 接入 DSpark 之后，抢占下的确定性。
- 长时间运行的稳定性。最长的一次是连续跑完一轮基准和三轮正确性评测，约 1 小时。
- 这张 MI50 之外的任何 GPU。

### 已知问题

早期版本偶尔会出现来自命令处理器（client CPF）、位于用户地址的 GPU 页故障，至少有一次导致运行卡住。原因不明，可能就是上文的页表竞争：只创建队列的进程在修复前约 1500 次启动中出现过 2 次，修复后约 600 次中没有出现，但次数太少，无法下结论。如果请求超时，请检查 `dmesg`。

### 复现

```bash
# 测评。<dir> 下是 <case>.jsonl 文件，每行一个请求：
# {"session_id": "...", "output_length": 1024, "payload": {<OpenAI completions 请求体>}}
go run ./cmd/bench -url http://127.0.0.1:18000 -cases <dir>

# GPU 检查。所需空闲显存：decode-determinism 约 3.3 GB，prefill-determinism 约 3 GB，
# decode-timing 约 4 GB
go build -o build/diag ./cmd/diag
sudo build/diag -payload build/payload.bin -mode decode-determinism
sudo build/diag -payload build/payload.bin -mode prefill-determinism
sudo build/diag -payload build/payload.bin -mode decode-timing

# 抢占下的同一检查：运行期间其他进程不断创建队列
sudo build/diag -payload build/payload.bin -mode decode-determinism -n 2000 &
while [ -n "$(jobs -r)" ]; do sudo build/diag -mode queue-churn; done

# 单元测试。分词器相关测试需要 payload，没有时会跳过
LFM3_PAYLOAD=$PWD/build/payload.bin go test ./...
```

本仓库不包含轨迹数据集、MATH、GPQA 数据和 drafter 权重。

## 许可证

本项目的代码（包括 GPU 内核）按 [Unlicense](LICENSE) 释放到公有领域。

这不包括模型权重，权重仍适用 LFM Open License v1.0。`build.sh` 构建出的可执行文件包含由权重派生的数据，因此再分发该文件时同样受此许可证约束。

[English](README.md) | [简体中文](README.zh-CN.md)

# lfm3-1.7b-mi50

Single-stream inference for [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on an AMD Radeon Instinct MI50 (gfx906), written in Go.

The result is one static ELF file holding the quantized weights, the tokenizer and every GPU kernel. It serves an OpenAI-compatible HTTP API and needs no Python, ROCm or libc at run time, only the kernel's amdgpu/KFD driver.

It replaces a vLLM deployment of the same model on the same card, running the same DSpark kernel and policy:

| Context | TTFT, vLLM → this | Greedy decode tok/s, vLLM → this |
|---|---|---|
| 1k | 0.201 → 0.124 s | 974 → 1174 |
| 32k | 5.829 → 4.607 s | 364 → 688 |
| 128k | 44.738 → 28.623 s | 301 → 424 |

TTFT, prefill, warm starts and the sampling path are better at every tier. Prefill attention runs PV in int8 with u8 softmax probabilities, twice as fast as the fp16 version; this costs some long-context precision, and `--exact-attention` restores the fp16 kernel (see [Correctness](#correctness)). Decode's verify attention reads the K/V cache once per layer for all verified tokens, 1.2 to 1.6 times as fast as before at 32k and 128k, with unchanged precision. The decode column above is a median of 3 streams and is **very noisy, so do not read it alone**: over 72 streams of the previous version the two tied overall (mean 559 → 545), and with repetition-loop streams separated, normal text was 5% to 16% faster at every tier; against that version, 18 streams per tier now give 32k 564 → 671 and 128k 362 → 410 (means). See [Evaluation](#evaluation), and the [optimization report](docs/OPTIMIZATION_REPORT.md) (Chinese) for the whole history.

Only `temperature: 0` requests use DSpark. The model's default temperature is 0.1, which takes the sampling path at about 260 tok/s. vLLM behaves the same way.

## Quick start

```bash
# 1. Download the weights (see "Model weights" for checksums)
rev=ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7
mkdir -p model
for f in config.json tokenizer.json model.safetensors; do
  curl -L -o "model/$f" "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/$rev/$f"
done

# 2. Build dist/lfm3-1.7b-mi50 (about 2 GB). See "Model weights" for the drafter directory.
./build.sh model drafter

# 3. Run. /dev/kfd needs root or the render group.
sudo ./dist/lfm3-1.7b-mi50 --port 18000

# 4. Use it
curl localhost:18000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Reply briefly."}],"temperature":0}'
```

The executable is self-contained, so you can copy it to another machine with the same card and run it there.

## Requirements

**To run**

| | Tested with | Notes |
|---|---|---|
| GPU | AMD Radeon Instinct MI50 32 GB (gfx906) | The only supported device. KFD topology is checked at startup |
| Kernel | Linux 7.2.11 (Debian), in-tree amdgpu | KFD ioctl ABI 1.x |
| Privileges | root | Or membership of the group that owns `/dev/kfd` (not tested) |
| Free VRAM | 32 GB card with nothing else on it | At least about 7 GB for the default 131072-token limit; the program refuses to start below that |
| CPU | Intel i5-1340P | One core spins while a request runs |
| RAM | 16 GB | Startup peaks at about 7.3 GB including the mapped file, then about 1.8 GB |

**To build**

| | Version | Notes |
|---|---|---|
| Go | 1.27 (tested 1.27.3) | No third-party modules. `CGO_ENABLED=0` |
| Model files | see below | Packing peaks at about 8 GB of RAM, including the mapped model file, and takes about 10 s on 16 threads |

**To change a GPU kernel** (only then)

| | Version | Notes |
|---|---|---|
| Docker image | `rocm/dev-ubuntu-24.04:7.2.4` | Run `kernels/build.sh`. No GPU needed. Override with `HIP_IMAGE` |

The compiled kernels are committed under `kernels/*.co`, so a normal build needs neither ROCm nor Docker. Rebuilding them with the image above gave instruction-identical code.

## Model weights

Weights are not in this repository. Download them from Hugging Face at the exact revision the code was verified against:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `model.safetensors` | 3457635448 | `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `tokenizer.json` | 17908635 | `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701` |
| `config.json` | 1454 | `8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d` |

Revision: `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`. The `curl` loop in Quick start fetches these files, or use the Hugging Face CLI:

```bash
hf download LiquidAI/LFM3-1.7B model.safetensors config.json tokenizer.json \
  --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 --local-dir model
```

`cmd/pack` quantizes the weights once at build time. Its output is byte-for-byte reproducible and matches the original Python quantizer on every matrix compared.

The DSpark drafter weights are not in this repository either and are not publicly released yet. The code was verified against a mid-training checkpoint, `lfm3-1.7b-dspark-l4b44-step10260`:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `model.safetensors` | 519113708 | `7a1e6d24a916b0814fd106b6194cdcd309eec7919f013d89798896d84efbdf0f` |
| `config.json` | 1855 | `930d3e014c73ec67f2fa66e782997dc779e2a0bd5195e5c508f76a69b0e8fc37` |

The 16384 frequent tokens of the head crop are committed as `cmd/pack/crop_ids.bin`. To change weights, delete `build/payload.bin` and run `./build.sh model <new directory>`, then replay the policy.

The weights are under the LFM Open License v1.0, which limits commercial use above a revenue threshold. A built executable contains weights derived from them, so read that license before you redistribute one.

## Design decisions

### Talk to KFD directly instead of linking ROCm

`internal/kfd` is about 1,100 lines of Go that open `/dev/kfd`, allocate and map GPU memory, create an AQL queue, load code objects and submit packets.

The alternatives were worse for this goal. HIP and ROCr are large C++ shared libraries, and their code-object handling brings in LLVM through comgr, so a single static binary built on them was not realistic. Using cgo would also give up Go's static builds. This program only needs a small part of those libraries: one queue, one device and precompiled kernels.

The cost is a narrow driver. It supports only this GPU, provides no scratch memory (no kernel here needs it) and no hostcall. To check it against the real thing, the ioctls and AQL packets that ROCr issues were logged and compared field by field.

### Go without cgo

`CGO_ENABLED=0` gives a static executable by default, and the standard library covers HTTP and JSON, so `go.mod` has no dependencies. Go's garbage collector does not get in the way: the token loop allocates little and all tensors live outside the Go heap.

### GPU kernels stay in HIP, precompiled

Go cannot target the GPU, and the existing kernels are heavily tuned, so they stay as HIP C++ and are embedded as gfx906 code objects with `go:embed`:

- `dspark.hip`: the DSpark megakernel. One launch does draft, verify and commit on the GPU, up to 32 rounds. Greedy requests use it. Its verify attention reads each 32-key page of K/V once for all verified tokens (up to 8, int8 QK, fp16 PV).
- `decode.hip`: the decode megakernel. One persistent launch runs all 24 layers and the LM head, one token at a time. Sampling, logprobs and `min_tokens` requests use it.
- `prefill_attn.hip`: flash attention over the paged int8 KV cache, with int8 QK and int8 PV (u8 softmax probabilities per 64-key tile).
- `prefill_attn_exact.hip`: the previous flash attention with fp16 probabilities, for `--exact-attention`.
- `gemm.hip`: W8A8 matrix multiply for every prefill linear layer.
- `ops.hip`: new. Native versions of what vLLM ran through torch or JIT-compiled Triton: embedding, RMS norm, causal conv, Q/K norm and RoPE, int8 KV store, gate, and the fp16 LM head.

Nothing is compiled at run time. The model is ready about 7.5 s after launch, against about 22 s for the vLLM deployment.

The kernels are committed in compiled form (about 530 KB) so that a Go toolchain alone can build the project.

The Go driver has no scratch memory, so `dspark.hip` must be built with `-mllvm -disable-machine-licm` and keep `private_segment_fixed_size` at 0.

### DSpark inside the megakernel

Decode spends most of its time reading weights, and verifying M tokens reads them only once: at 8k, M=1 takes 3.73 ms per round and M≈4 only 4.06 ms. So a 3-layer drafter guesses and the target verifies in one pass, `tok/s = τ / R`. Draft and verify alternate more than 200 times per second, too often to return to the CPU, so everything is in one kernel and the CPU only sends out committed tokens. A rejected token needs no rollback: conv history is a position-indexed ring and K/V goes to a temporary page until commit.

One thing goes beyond the vLLM version: verify may cross the 32-token block boundary, and the kernel writes the conv state at the boundary from its history ring. Replay shows a gain of 0.6% to 1.3%.

The policy (at most 7 drafts below 57344 tokens, 3 from there on, confidence threshold −0.4) comes from fixed-sequence replay rather than benchmark numbers, for the reason given in [Evaluation](#evaluation). The switch point moved from 32768 when verify attention became one pass over all verified tokens: over 18 streams, 7 drafts win at 32k and 48k and 3 at 64k and 128k (`benchmarks/verify_attention.txt`).

### Append the weights after the ELF image

The model data is about 1.9 GB and the Go linker caps data sections at about 2 GB, which leaves too little room for `go:embed`. `cmd/pack` appends the data to the built executable, with a footer locating it, and the program maps its own file read-only at startup.

The weights are entropy coded in chunks with the standard library's Huffman coder and decoded in parallel at startup, into memory rather than to disk. The catch is that `strip` and similar tools would drop the data, so `build.sh` strips the Go binary before attaching it.

### Store one weight layout, derive the other

Decode and prefill use the same int8 values in different layouts. Only the decode layout is stored, and the prefill layout is built at startup, which keeps about 1.5 GB out of the executable.

Also not stored: the LM head (the quantized embedding, 260 MB) and the drafter's head crop and k/v projection (quantization is per row, so those rows are copied). Entropy coding saves only 7% on int8 weights, which are close to random noise, and 34% on the embedding, where BF16 to fp16 leaves the low 3 bits zero. With a drafter added, the file still shrank from 2.3 GB to 1.97 GB.

### Prefix cache suited to a single stream

Full-attention KV blocks are evicted LRU. Sliding-attention blocks and conv checkpoints live in ring buffers, because a hit only needs the most recent window of them. Hits land on the same 32-token block boundaries as the vLLM deployment, and the hit counts matched on all 18 requests compared. `cache_salt` isolates requests.

### Hand-written tokenizer and chat template

Go's `regexp` has no lookahead, which the model's pre-tokenizer pattern needs, so that split is a hand-written scanner. The chat template is a direct translation of the model's Jinja file rather than a template engine. Both are checked against output captured from the reference deployment (`testdata/`).

### What was left out

Only single-stream inference on this card is in scope. Structured output (it needs a grammar engine), multimodal input and batching are not implemented.

## What had to be right

These are the things that took debugging. They constrain future changes, and the comments in the code point back to them.

- **Dispatches use agent-scope fences, as ROCr does.** Nothing in the serving path needs a system-scope acquire or release, and the next point shows what one costs. The only system-scope barrier is in `cmd/diag`, which reads results back from VRAM.
- **Nothing the host reads needs an L2 writeback.** Tokens and logits go straight to uncached host memory, with a marker to confirm arrival. Writing back the L2 on every token cost up to 0.9 ms, and removing it is the main reason decode beats vLLM: kernel time is the same on both, and the driver adds about 0.04 ms per token.
- **Data rewritten at run time goes through the GPU.** Host writes through the BAR bypass the L2, which can keep serving old values, so `Engine.upload` and `Engine.loadToVRAM` copy through it instead.
- **Doorbells and HDP registers take plain stores.** Go's `atomic.Store` is a locked `XCHG` on amd64. The doorbell value is the last packet's id, not the write index.
- **The decode kernel had a race, now fixed.** It prefetched `h_old` before the grid barrier. That is safe for every layer except layer 0, whose `h0` was written by block 0 in the phase just finished, so other blocks could read the previous token's embedding and compute a whole block wrong. In this driver it appeared above 16 blocks. It depends on timing: the unfixed kernel under ROCr passed the same bit-exact test, and why was not established. The fix is one line in `decode.hip`. Under ROCr it adds about 0.6% kernel time and left the first 256 greedy tokens unchanged on all 33 benchmark requests, so the race was not behind vLLM's lower agreement with HF below.
- **Loads issued before a grid barrier may only touch data of earlier phases.** The DSpark verify attention issues its first page's loads before it waits for the barrier after the query phase, so they overlap the wait. The verified tokens' fresh K/V is written in that query phase by other blocks, so a split whose first page is the fresh one (block 0 with no context pages, as on every sliding-window layer with 256 keys) has to issue it after the barrier. A development version did not, and its results for M = 2 depended on timing (worst error 1.7e-3 against fp64 instead of 3.8e-4).
- **Wait before acquiring the VM.** Opening the render node starts asynchronous page-table updates in the kernel. If `ACQUIRE_VM` follows within about a millisecond, they overwrite the page directory entries KFD writes for its trap handler. A process runs normally without that mapping until a running wave is preempted, which happens whenever any process's queues change: the wave cannot fetch the trap handler, and the queue hangs. ROCr is slow enough between the two calls never to hit this. Here it appeared as an occasional hang at startup beside vLLM, because the host memory compaction during startup made KFD stop and restart vLLM's queues. A page-table walk through debugfs showed the entry missing, and a 1 ms delay restored it. `kfd.Open` waits 200 ms, since there is no ioctl to wait for those updates.
- **The prefill attention kernel must not spill either.** It runs at 512 threads per block, which caps it at 128 VGPRs, and the driver has no scratch memory. Three compiler behaviors broke that while it was written, and the code works around each: values computed before a barrier were sunk past it to their use, keeping every loaded operand live (the logits and accumulators are pinned with empty `asm` statements); identical barriers in two branches let the compiler merge the two wave groups' loops (each group's barrier asm differs); and two conditional stores of the two halves of one loaded pair were merged into one store of the wrong half (each thread now loads one dword into one variable). `kernels/build.sh` fails if any kernel spills or needs scratch.
- **Do not fill VRAM.** When VRAM runs out the kernel evicts other processes' buffers into host RAM, which can invoke the OOM killer. The program therefore checks free VRAM up front, keeps 512 MiB of headroom and refuses to start rather than evict.

## API

`/v1/completions`, `/v1/chat/completions` (streaming and not), `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/reset_prefix_cache`.

Supported: thinking (the `reasoning` field), tool calls (Qwen3 XML format, arguments typed from the tool schema), `cache_salt`, `logprobs`, `stop`, `stop_token_ids`, `ignore_eos`, `min_tokens`, `logit_bias`, `allowed_token_ids`, `seed`, `return_token_ids`, `stream_options`. Sampling defaults come from the model's `generation_config.json`: temperature 0.1, top_k 50, repetition_penalty 1.1.

These return HTTP 400: `response_format`, `structured_outputs`, `guided_*`, `tool_choice` set to `required` or to a named function, and `n > 1`.

Differences from vLLM's API:

- Streaming tool-call arguments arrive one complete parameter at a time rather than character by character. Concatenated, they are the same.
- A streamed response ends with an extra empty chunk carrying `finish_reason`.
- `seed` is reproducible, but the random sequence differs from torch's.

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `18000` | Listen address. The server has no authentication, so restrict `--host` on an untrusted network |
| `--max-model-len` | `131072` | Longest prompt plus output, in tokens |
| `--gpu-memory-utilization` | `0.90` | Share of VRAM to use; the KV pool takes what remains |
| `--kv-tokens` | from VRAM | Full-attention KV capacity, in tokens |
| `--payload` | the executable | Read the model from a separate payload file (development) |
| `--no-draft` | off | Keep greedy requests off DSpark, for comparison |
| `--exact-attention` | off | Prefill attention with fp16 softmax probabilities: half the attention speed (128k TTFT 44.4 s instead of 28.7 s), the long-context precision of the previous release |

## Project layout

```
build.sh           build dist/lfm3-1.7b-mi50
cmd/lfm3-1.7b-mi50      the server
cmd/pack           quantize weights and the drafter, compile the tokenizer, attach the payload (build time)
cmd/bench          benchmark client for any OpenAI-compatible server
cmd/diag           GPU checks: determinism per code path, kernel vs wall time
internal/kfd       pure-Go KFD driver: ioctls, memory, AQL queue, code object loader
internal/model     weight layouts, prefill pipeline, decode, DSpark, prefix cache, generation loop
internal/sample    sampler, in vLLM's order of operations
internal/tokenizer BPE tokenizer
internal/chat      the chat template, translated by hand
internal/server    HTTP API, reasoning and tool-call parsing
kernels/           GPU kernels: .hip sources and compiled gfx906 code objects
testdata/          tokenizer, template and API samples captured from the reference deployment
benchmarks/        raw benchmark output
docs/              optimization report
```

## Evaluation

**Setup.** MI50 32 GB (gfx906, 60 CUs), Intel i5-1340P, 16 GB RAM, Linux 7.2.11. The baseline is vLLM (ROCm 7.2.4) with the same model, the same DSpark kernel, drafter step 10260 and the same policy. Each ran alone on the GPU. All numbers are absolute measurements from `cmd/bench`. Raw output is in `benchmarks/`.

### Performance

Single stream, streaming responses, 1024 output tokens, cold cache, temperature 0. Requests are real agent traces from Open-SWE-Traces, sent as token ids. Each value is the median of 3 sessions.

| Context | Input tokens | TTFT vLLM | TTFT this | Prefill tok/s vLLM | Prefill tok/s this | Decode tok/s vLLM | Decode tok/s this |
|---|---|---|---|---|---|---|---|
| 1k | 1024 | 0.201 s | 0.124 s | 5090 | 8258 | 974.2 | 1174.0 |
| 8k | 8192 | 1.157 s | 0.976 s | 7082 | 8396 | 532.3 | 436.1 |
| 32k | 32768 | 5.829 s | 4.607 s | 5622 | 7112 | 363.5 | 687.9 |
| 64k | 65536 | 15.574 s | 11.160 s | 4208 | 5873 | 412.8 | 435.0 |
| 128k | 126976 | 44.738 s | 28.623 s | 2838 | 4436 | 301.0 | 423.7 |

All columns of this program are from `benchmarks/lfm3-1.7b-mi50_cold.txt`, after the int8 PV prefill attention and the one-pass verify attention. The same run of the previous version (`benchmarks/lfm3-1.7b-mi50_cold_before-verify-attention.txt`) gave 393.0, 484.8, 425.8, 446.6 and 305.4: the 8k and 64k medians went down with τ, not with R (the next section), and a rerun gave the same tokens. The prefill attention kernel alone is 2.06 times as fast on a 128k prompt (32.6 s to 15.9 s over all layers); see `benchmarks/prefill_attention.txt`.

**Verify attention.** Each verify round runs attention for M = 1 + drafts tokens over the whole cache. It used to run as passes of up to 4 tokens, each reading all of K/V; now one pass per layer takes all M tokens (`attention_v2` for M ≤ 3, `attention_v3` for M = 4 to 8): a wave stages one 32-key page of its KV head in LDS, and 64 lanes split the 4 query heads × 4 keys × 4 dimension groups so that 8 tokens' queries fit in registers. Per layer at 131000 tokens (µs, query and combine included):

| M | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| before | 247.0 | 288.2 | 440.6 | 451.9 | 889.6 |
| now | 210.0 | 231.7 | 311.7 | 335.1 | 557.3 |

M = 1 is near the memory floor (176 µs). Each further token costs about 55 µs per layer at 128k, the instruction rate at the clock the 178 W cap leaves (about 1350 MHz with the K/V loads running). Over 18 streams per tier, R at the same M and τ fell by 5% at 32k, 6% at 64k and 11% at 128k and is unchanged at 1k and 8k; the decode means went 564 → 671 at 32k and 362 → 410 at 128k. Details in `benchmarks/verify_attention.txt` and `benchmarks/lfm3-1.7b-mi50_multistream_verify-attention.txt`.

Multi-turn replay with a warm prefix cache (3 sessions of 3 turns), median of the 9 requests, measured before the int8 PV prefill attention:

| Tier | TTFT vLLM | TTFT this | Decode tok/s vLLM | Decode tok/s this |
|---|---|---|---|---|
| 32k | 1.021 s | 0.913 s | 436.4 | 447.1 |
| 128k | 1.359 s | 1.172 s | 331.0 | 335.7 |

Cached token counts matched vLLM on all 18 requests.

The sampling path (temperature 0.6, top_p 0.95, top_k 20, no DSpark), measured before the int8 PV prefill attention:

| Tier | Decode tok/s vLLM | Decode tok/s this |
|---|---|---|
| 8k | 191.1 | 260.6 |
| 128k | 150.9 | 189.1 |

### Why the decode median cannot be read alone

`tok/s = τ / R`. R can be compared; the τ of a single stream cannot. A different M rounds differently, a near-tie goes to another token, and all later text and τ change. These prompts are truncated traces with `ignore_eos`, so the model often runs past its natural end into a repetition loop, on vLLM and on the no-draft path as well. An easy loop reaches 1300 tok/s, a hard one 250, and which stream lands in which is a lottery. A policy change that is neutral in replay moved the table above from 1207 to 706 at 1k and from 307 to 493 at 128k.

So 72 further streams were measured (the same prompts minus their first 16·v tokens), separating those whose first 256 tokens have over 0.3 repeated 8-grams:

| Tier | Streams | Mean vLLM → this | Median vLLM → this | Non-loop mean vLLM → this | Loop mean vLLM → this |
|---|---|---|---|---|---|
| 1k | 18 | 767 → 614 | 762 → 411 | 438 → 481 | 1031 → 747 |
| 8k | 18 | 595 → 622 | 489 → 471 | 469 → 499 | 794 → 775 |
| 32k | 18 | 498 → 557 | 449 → 504 | 417 → 455 | 709 → 760 |
| 64k | 9 | 407 → 443 | 414 → 439 | 406 → 471 | 408 → 408 |
| 128k | 9 | 347 → 334 | 301 → 308 | 317 → 334 | 452 → none |
| All | 72 | 559 → 545 | 456 → 436 | | |

**Overall they tie. The 1k mean and the 8k median are below vLLM, and those gaps come from the loop lottery; non-loop streams are faster at every tier (+5% to +16%).** The policy is chosen by replay, not from these numbers.

### Correctness

20 random MATH-500 and 20 random GPQA-Diamond problems, thinking on, max_tokens 32768. Same problems throughout, all real long outputs (4,000 to 12,000 tokens on average):

| Path | MATH-500 accuracy | MATH decode tok/s | GPQA accuracy | GPQA decode tok/s | Loops | Truncated |
|---|---|---|---|---|---|---|
| This, greedy, DSpark | 95% | 639.5 | 50% | 507.5 | 0 | 0 |
| vLLM, greedy, DSpark | 90% | 616.0 | 50% | 504.6 | 0 | 2 |
| This, greedy, no draft | 90% | 293.2 | 50% | 290.5 | 0 | 1 |
| This, sampling (temperature 0.6) | 95% | 262.8 | 65% | 262.6 | 0 | 0 |

| Check | Result |
|---|---|
| Prefix agreement with HF fp32 greedy (repetition_penalty 1.0, 5 prompts of 48 tokens) | 48, 48, 17, 31, 48 of 48 |
| Tokenizer and chat template, 47 captured samples | Token-for-token identical |
| Weights, 6 matrices plus embeddings and norms compared with the Python quantizer | Byte-identical |
| Determinism, prefill and decode kernels | Logits bit-identical across runs |
| Determinism, DSpark, 15 benchmark requests | First 256 tokens identical across runs (rechecked after the verify attention change) |
| Verify attention against an fp64 reference of the same int8 Q/K/V, contexts 1 to 131000, M = 1 to 9, window 0 and 256 | Worst relative error 3.80e-4, as before (3.79e-4) |
| Determinism under preemption, 2000 decode steps while 130 other processes created queues | Bit-identical, no GPU faults (measured before DSpark) |

**Prefill attention's softmax probabilities are u8.** P' = p x V scale is quantized per row and 64-key tile against the tile's largest value, and PV runs on `v_dot4`. In an HF fp32 simulation of the kernels' quantization, the KL of the last 192 next-token distributions against fp32 is unchanged at 16k (0.00025) and rises at 60k from 0.0171 to 0.0251 (mean of two sessions; top-1 agreement 96.1% to 93.0%). For reference, the int8 KV cache alone gives 0.031 on the harder session. Per-row quantization of P, which the old vLLM deployment used, collapses long outputs (KL 4.8); the per-tile scale does not. `--exact-attention` brings back fp16 P at half the attention speed. Details in `benchmarks/prefill_attention.txt`.

**GEMV's x is int8.** This was the first place speed was bought with precision: 8% to 17% per round, at a KL about 6 times fp16's (0.0028 vs 0.0004) and 1 to 4 more top-1 flips in 119 positions, all where the top two are close. The fp16 fallback needs scratch memory, which the Go driver does not have.

Read these with some care:

- With 20 problems the sampling error is about ±10 points and the random sequences differ, so this shows the absence of a precision collapse, not better accuracy.
- Greedy output is not token-identical to vLLM, nor to this program's `--no-draft` path. The split position varies, usually within the first 100 tokens.
- Per-round drafts: for the same prompt and prefix, 60 of 72 aligned rounds had the same M and drafts as vLLM.

### Not measured

- Agreement with HF above 8k context. The CPU reference is too slow there, so long contexts were only checked by sampling their output.
- Determinism under preemption since DSpark was added.
- Long-running stability. The longest run was one benchmark pass and three accuracy passes in a row, about 1 hour.
- Any GPU other than this one MI50.

### Known issue

Early builds intermittently logged a GPU page fault from the command processor (client CPF) at a user address, at least once leaving a run hung. The cause is unknown. It may be the page-table race above: processes that only create a queue logged it twice in about 1500 starts before that fix and not at all in about 600 after, but those counts are too small to tell. Check `dmesg` if a request times out.

### Reproducing

```bash
# Benchmark. <dir> holds <case>.jsonl files with one request per line:
# {"session_id": "...", "output_length": 1024, "payload": {<OpenAI completions body>}}
go run ./cmd/bench -url http://127.0.0.1:18000 -cases <dir>

# GPU checks. Free VRAM needed: decode-determinism about 3.3 GB, prefill-determinism about
# 3 GB, decode-timing about 4 GB.
go build -o build/diag ./cmd/diag
sudo build/diag -payload build/payload.bin -mode decode-determinism
sudo build/diag -payload build/payload.bin -mode prefill-determinism
sudo build/diag -payload build/payload.bin -mode decode-timing

# The same check under preemption: other processes keep creating queues while it runs.
sudo build/diag -payload build/payload.bin -mode decode-determinism -n 2000 &
while [ -n "$(jobs -r)" ]; do sudo build/diag -mode queue-churn; done

# Time per DSpark phase and round, in the server log at the end of each request (development aid:
# launches run at most 4 rounds while it is on). The value is the block whose marks are recorded.
sudo env LFM3_DS_TRACE=7 ./dist/lfm3-1.7b-mi50 --port 18000

# Unit tests. The tokenizer tests need a payload and are skipped without one.
LFM3_PAYLOAD=$PWD/build/payload.bin go test ./...
```

The trace dataset, the MATH and GPQA data and the drafter weights are not included in this repository.

## License

The code, including the GPU kernels, is released into the public domain under the [Unlicense](LICENSE).

This does not cover the model weights, which stay under the LFM Open License v1.0. An executable built by `build.sh` contains data derived from them, so redistributing it is subject to that license as well.

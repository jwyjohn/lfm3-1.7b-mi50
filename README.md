[English](README.md) | [简体中文](README.zh-CN.md)

# lfm3-1.7b-mi50

Single-stream inference for [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) on an AMD Radeon Instinct MI50 (gfx906), written in Go.

The result is one static ELF file holding the quantized weights, the tokenizer and every GPU kernel. It serves an OpenAI-compatible HTTP API and needs no Python, ROCm or libc at run time, only the kernel's amdgpu/KFD driver.

It replaces a vLLM deployment of the same model on the same card, and measured faster at every context length tested:

| Context | TTFT, vLLM → this | Decode tok/s, vLLM → this |
|---|---|---|
| 1k | 0.211 → 0.126 s | 286.4 → 302.2 |
| 32k | 5.984 → 5.798 s | 257.0 → 270.1 |
| 128k | 46.506 → 45.982 s | 200.5 → 208.1 |

See [Evaluation](#evaluation) for the method, the full tables and what was not measured.

## Quick start

```bash
# 1. Download the weights (see "Model weights" for checksums)
rev=ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7
mkdir -p model
for f in config.json tokenizer.json model.safetensors; do
  curl -L -o "model/$f" "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/$rev/$f"
done

# 2. Build dist/lfm3-1.7b-mi50 (about 2.3 GB)
./build.sh model

# 3. Run. /dev/kfd needs root or the render group.
sudo ./dist/lfm3-1.7b-mi50 --port 18000

# 4. Use it
curl localhost:18000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Reply briefly."}]}'
```

The executable is self-contained, so you can copy it to another machine with the same card and run it there.

## Requirements

**To run**

| | Tested with | Notes |
|---|---|---|
| GPU | AMD Radeon Instinct MI50 32 GB (gfx906) | The only supported device. KFD topology is checked at startup |
| Kernel | Linux 7.2.11 (Debian), in-tree amdgpu | KFD ioctl ABI 1.x |
| Privileges | root | Or membership of the group that owns `/dev/kfd` (not tested) |
| Free VRAM | 32 GB card with nothing else on it | At least about 5.5 GB for the default 131072-token limit; the program refuses to start below that |
| CPU | Intel i5-1340P | One core spins while a request runs |
| RAM | 16 GB | The weights are read through the page cache |

**To build**

| | Version | Notes |
|---|---|---|
| Go | 1.27 (tested 1.27.3) | No third-party modules. `CGO_ENABLED=0` |
| Model files | see below | Packing peaks at about 8 GB of RAM, including the mapped model file, and takes about 13 s on 16 threads |

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

- `decode.hip`: the decode megakernel. One persistent launch runs all 24 layers, the LM head and greedy sampling.
- `prefill_attn.hip`: flash attention over the paged int8 KV cache.
- `gemm.hip`: W8A8 matrix multiply for every prefill linear layer.
- `ops.hip`: new. Native versions of what vLLM ran through torch or JIT-compiled Triton: embedding, RMS norm, causal conv, Q/K norm and RoPE, int8 KV store, gate, and the fp16 LM head.

Nothing is compiled at run time. The model is ready about 5 s after launch, against about 38 s for the vLLM deployment.

The kernels are committed in compiled form (about 170 KB) so that a Go toolchain alone can build the project.

### Append the weights after the ELF image

The model data is about 2.3 GB, and the Go linker caps data sections at about 2 GB, which rules out `go:embed`. `cmd/pack` appends the data to the built executable, with a footer locating it, and the program maps its own file read-only at startup.

Nothing is unpacked to disk. The catch is that `strip` and similar tools would drop the data, so `build.sh` strips the Go binary before attaching it.

### Store one weight layout, derive the other

Decode and prefill use the same int8 values in different layouts. Only the decode layout is stored, and the prefill layout is built at startup, which keeps about 1.5 GB out of the executable.

### Prefix cache suited to a single stream

Full-attention KV blocks are evicted LRU. Sliding-attention blocks and conv checkpoints live in ring buffers, because a hit only needs the most recent window of them. Hits land on the same 32-token block boundaries as the vLLM deployment, and the hit counts matched on all 18 requests compared. `cache_salt` isolates requests.

### Hand-written tokenizer and chat template

Go's `regexp` has no lookahead, which the model's pre-tokenizer pattern needs, so that split is a hand-written scanner. The chat template is a direct translation of the model's Jinja file rather than a template engine. Both are checked against output captured from the reference deployment (`testdata/`).

### What was left out

Only single-stream inference on this card is in scope. Structured output (it needs a grammar engine), speculation, multimodal input and batching are not implemented.

## What had to be right

These are the things that took debugging. They constrain future changes, and the comments in the code point back to them.

- **Dispatches use agent-scope fences, as ROCr does.** Nothing in the serving path needs a system-scope acquire or release, and the next point shows what one costs. The only system-scope barrier is in `cmd/diag`, which reads results back from VRAM.
- **Nothing the host reads needs an L2 writeback.** Tokens and logits go straight to uncached host memory, with a marker to confirm arrival. Writing back the L2 on every token cost up to 0.9 ms, and removing it is the main reason decode beats vLLM: kernel time is the same on both, and the driver adds about 0.04 ms per token.
- **Data rewritten at run time goes through the GPU.** Host writes through the BAR bypass the L2, which can keep serving old values, so `Engine.upload` and `Engine.loadToVRAM` copy through it instead.
- **Doorbells and HDP registers take plain stores.** Go's `atomic.Store` is a locked `XCHG` on amd64. The doorbell value is the last packet's id, not the write index.
- **The decode kernel had a race, now fixed.** It prefetched `h_old` before the grid barrier. That is safe for every layer except layer 0, whose `h0` was written by block 0 in the phase just finished, so other blocks could read the previous token's embedding and compute a whole block wrong. In this driver it appeared above 16 blocks. It depends on timing: the unfixed kernel under ROCr passed the same bit-exact test, and why was not established. The fix is one line in `decode.hip`. Under ROCr it adds about 0.6% kernel time and left the first 256 greedy tokens unchanged on all 33 benchmark requests, so the race was not behind vLLM's lower agreement with HF below.
- **Wait before acquiring the VM.** Opening the render node starts asynchronous page-table updates in the kernel. If `ACQUIRE_VM` follows within about a millisecond, they overwrite the page directory entries KFD writes for its trap handler. A process runs normally without that mapping until a running wave is preempted, which happens whenever any process's queues change: the wave cannot fetch the trap handler, and the queue hangs. ROCr is slow enough between the two calls never to hit this. Here it appeared as an occasional hang at startup beside vLLM, because the host memory compaction during startup made KFD stop and restart vLLM's queues. A page-table walk through debugfs showed the entry missing, and a 1 ms delay restored it. `kfd.Open` waits 200 ms, since there is no ioctl to wait for those updates.
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

## Project layout

```
build.sh           build dist/lfm3-1.7b-mi50
cmd/lfm3-1.7b-mi50      the server
cmd/pack           quantize weights, compile the tokenizer, attach the payload (build time)
cmd/bench          benchmark client for any OpenAI-compatible server
cmd/diag           GPU checks: determinism per code path, kernel vs wall time
internal/kfd       pure-Go KFD driver: ioctls, memory, AQL queue, code object loader
internal/model     weight layouts, prefill pipeline, decode, prefix cache, generation loop
internal/sample    sampler, in vLLM's order of operations
internal/tokenizer BPE tokenizer
internal/chat      the chat template, translated by hand
internal/server    HTTP API, reasoning and tool-call parsing
kernels/           GPU kernels: .hip sources and compiled gfx906 code objects
testdata/          tokenizer, template and API samples captured from the reference deployment
benchmarks/        raw benchmark output
```

## Evaluation

**Setup.** MI50 32 GB (gfx906, 60 CUs), Intel i5-1340P, 16 GB RAM, Linux 7.2.11. The baseline is vLLM (ROCm 7.2.4) with the same model, the same custom kernels (with the decode fix below), an int8 KV cache and speculation off. Each ran alone on the GPU. All numbers are absolute measurements from `cmd/bench`, with no comparisons under load.

### Performance

Single stream, streaming responses, 1024 output tokens, cold cache. Requests are real agent traces from Open-SWE-Traces, sent as token ids. Each value is the median of 3 sessions. Raw output is in `benchmarks/`.

| Context | Input tokens | TTFT vLLM | TTFT this | Prefill tok/s vLLM | Prefill tok/s this | Decode tok/s vLLM | Decode tok/s this |
|---|---|---|---|---|---|---|---|
| 1k | 1024 | 0.211 s | 0.126 s | 4859 | 8138 | 286.4 | 302.2 |
| 8k | 8192 | 1.158 s | 1.049 s | 7074 | 7812 | 278.8 | 293.2 |
| 32k | 32768 | 5.984 s | 5.798 s | 5476 | 5651 | 257.0 | 270.1 |
| 64k | 65536 | 16.125 s | 15.806 s | 4064 | 4146 | 233.7 | 244.9 |
| 128k | 126976 | 46.506 s | 45.982 s | 2730 | 2761 | 200.5 | 208.1 |

Multi-turn replay with a warm prefix cache (3 sessions of 3 turns), median of the 9 requests:

| Tier | TTFT vLLM | TTFT this | Decode tok/s vLLM | Decode tok/s this |
|---|---|---|---|---|
| 32k | 1.048 s | 0.931 s | 259.0 | 273.2 |
| 128k | 1.386 s | 1.180 s | 201.1 | 210.0 |

Cached token counts matched vLLM on all 18 requests, and TTFT was lower on all of them.

Sampling on the host (temperature 0.6, top_p 0.95, top_k 20) decoded at about 261 tok/s, against about 214 for vLLM, measured during the accuracy runs below.

### Correctness

| Check | Result |
|---|---|
| MATH-500, 20 random problems | 90% (vLLM 90%), no loops, 1 truncated (vLLM 0) |
| GPQA-Diamond, 20 random problems | 65% (vLLM 55%), no loops, 2 truncated (vLLM 1) |
| KL against HF fp32, first generated token, 9 prompts | 0.00001 to 0.098 (vLLM 0.00004 to 0.263). Top-1 matches HF on 9 of 9 (vLLM 8 of 9) |
| Agreement with HF greedy, 200 tokens on 5 real 1k and 8k traces | 193 to 198 of 200 (vLLM 185 to 193) |
| Tokenizer and chat template, 47 captured samples | Token-for-token identical |
| Weights, 6 matrices plus embeddings and norms compared with the Python quantizer | Byte-identical |
| Determinism, prefill and decode | Logits bit-identical across runs |
| Determinism under preemption, 2000 decode steps while 130 other processes created queues | Bit-identical, no GPU faults. A ROCr process passed the same kind of test |

Read these with some care:

- The accuracy, KL and HF agreement rows were measured just before the final cleanup, and vLLM's before its decode fix. Both newer builds reproduce the first 256 tokens of their earlier selves on all 33 benchmark requests.
- With 20 problems the sampling error is about ±10 points and the random sequences differ, so the accuracy runs show the absence of a precision collapse, not better accuracy.
- Greedy output is not token-identical to vLLM. On short prompts two of three matched for all 64 tokens and one split at position 29. On long traces the split comes earlier, at positions that vary.

### Not measured

- Agreement with HF above 8k context. The CPU reference is too slow there, so long contexts were only checked by sampling their output.
- Long-running stability. The longest run was the accuracy suite, about 20 minutes.
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

# Unit tests. The tokenizer tests need a payload and are skipped without one.
LFM3_PAYLOAD=build/payload.bin go test ./...
```

The trace dataset and the MATH and GPQA data are not included in this repository.

## License

The code, including the GPU kernels, is released into the public domain under the [Unlicense](LICENSE).

This does not cover the model weights, which stay under the LFM Open License v1.0. An executable built by `build.sh` contains data derived from them, so redistributing it is subject to that license as well.

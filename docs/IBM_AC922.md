# Experimental: IBM Power System AC922 (POWER9 + 4x V100, NVLink 2.0)

This branch (`ac922`) runs Strata on an IBM AC922: two POWER9 sockets (20 cores each, SMT4, ppc64le) and four
NVIDIA Tesla V100-SXM2 16 GB (sm_70), each GPU joined to its socket by NVLink 2.0 (~72 GB/s one way) and to its
partner GPU by a second NVLink. The CPU and the GPUs share one coherent, unified memory (ATS over NVLink): a GPU
reads the sockets' RAM directly. Tested with Qwen3.8-Flash-Next **UD-Q4_K_XL** (Unsloth 4-bit, 71.7 GiB of experts)
and **IQ2_XS**, on 2 and 4 GPUs.

Everything here was measured on one AC922 (RHEL 8, driver 550.54.15 - ppc64le's last - CUDA 12.4, gcc-toolset-12).
It is not an upstream-supported platform.

- [Build](#build)
- [Run](#run)
- [Speed](#speed)
- [What this branch adds](#what-this-branch-adds)
- [Quality](#quality)
- [Tried and dropped](#tried-and-dropped)
- [Hardware rules learned on the AC922](#hardware-rules-learned-on-the-ac922)

## Build

```sh
source /opt/rh/gcc-toolset-12/enable          # RHEL 8's gcc 8 is too old
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DSTRATA_GGML_DIR=third_party/llama.cpp
cmake --build build --target strata
```

- `STRATA_EXPERIMENTAL_SM60` lets the build and setup accept the V100s (setup otherwise wants sm_75+).
- ggml-cpu is built with `-mcpu=power9` (its POWER9 VSX code) and as gnu++17 (that code uses the `vector` keyword).
- `setup.sh` works on ppc64le (it compiles the engine: the ready-made engine is x86-64; driver floor 525 for a
  local build). It still installs UD-Q4_K_XL on one GPU; the multi-GPU configs below are written by hand.

## Run

The fastest configuration keeps the whole model page-locked in RAM, half on each socket, and splits the layers
across the GPUs (`--resident-budget-gib` with `--layer-split auto`). A server config (`serve/server.py --config`):

```json
{
 "exe": "/path/to/Strata-AC922/build/strata",
 "args": ["--pack", "/path/to/packs/unsloth-ud-q4_k_xl",
          "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
          "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto",
          "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/path/to/mtp/rt",
          "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768", "--ple-io", "ram",
          "--resident-budget-gib", "72"],
 "gpu": [0, 1, 2, 3],
 "layer_split": "auto"
}
```

and the environment of the server:

```sh
# best mode: default outputs change slightly (FP16 instead of BF16 weights), quality the same (see Quality)
export STRATA_FP16=load STRATA_FP16_GATES=1 STRATA_ATTN_WMMA=1
# opt-in, faster prompts (another summation order; quality checked)
export STRATA_FUSED_EXPERTS=1 STRATA_SELECT_VOLTA=1
```

On **2 GPUs** (`"gpu": [0, 1]`, one socket) add the CPU share of the decode misses:

```sh
export STRATA_VSX_EXPERTS=1 STRATA_POOL_SMT=2
# and in args: "--pcie-frac", "0.6", "--pool-workers", "39"
```

Startup takes 3-5 minutes (the 72 GiB arena is copied in, the 27 GiB PLE table locked).

## Speed

Method: [llama-benchy](https://github.com/eugr/llama-benchy), `--pp 2048 8192 --tg 128 --depth 0 16384 65536 --runs 3
--no-cache`; prefill and decode taken from the server's log (`strata serve: prompt ... read in ... generated in`),
3 runs averaged after a warm-up request. Decode is with MTP speculative decoding (`--spec 4`, 73-82% accepted).

### 4 GPUs, UD-Q4_K_XL: where it started and where it is

| Test | First port (`--mmap-experts`) | Per-socket arena, BF16 default (2026-10-02) | **`ac922` now (best mode)** |
|---|---:|---:|---:|
| Prefill pp2048 | 426 | 659 | **812** |
| Prefill pp8192 | 755 | 1,106 | **1,583** |
| Prefill pp2048 @ 16K | 864 | 1,098 | **1,981** |
| Prefill pp8192 @ 16K | 952 | 1,145 | **2,034** |
| Prefill pp2048 @ 64K | 1,106 | 1,265 | **3,601** |
| Prefill pp8192 @ 64K | 1,121 | 1,262 | **3,320** |
| Decode (mean of the 6 rows) | 50 | 69.5 | **76.7** |

The `ac922` column is after the merge of upstream 0.1.38, best mode without the opt-in prompt kernels. The first
column is the first working port with the CPU-affinity fix; before that fix a 2K prompt read at 38 tok/s.

### Long prompts with the opt-in prompt kernels (4 GPUs, one 64,907-token prompt, server log)

| Mode | Read |
|---|---:|
| best mode | 4,558 tok/s |
| + `STRATA_FUSED_EXPERTS=1` | 5,206 tok/s (+14%) |
| + `STRATA_SELECT_VOLTA=1` | **5,673 tok/s (+24%)** |

A 7,846-token prompt: 1,675 -> 1,951 tok/s with the fused experts (+16%; selection does not apply below 2K cells).

### 2 GPUs (one socket, GPUs 0-1), UD-Q4_K_XL

| Test | Earlier on this branch (`94d7556`, BF16) | Best mode |
|---|---:|---:|
| Prefill pp2048 | 630 | 787 |
| Prefill pp8192 | 1,111 | 1,569 |
| Prefill pp2048 @ 16K | 1,216 | 1,724 |
| Prefill pp8192 @ 16K | 1,247 | 1,811 |
| Prefill pp2048 @ 64K | 1,614 | 2,322 |
| Prefill pp8192 @ 64K | 1,561 | 2,261 |

Decode on 2 GPUs: 51-58 tok/s before the peer fetch, 60-68 after it (benchy). With the CPU share (3 greedy
answers of 384 tokens; ms per verify window, lower is better; tok/s at 2.4 tokens per window):

| Config | ms/window | tok/s |
|---|---:|---:|
| default | 35.7 | ~67 |
| default after the activation-quantization skip | 34.5 | ~70 |
| `STRATA_VSX_EXPERTS=1 STRATA_POOL_SMT=2 --pcie-frac 0.6 --pool-workers 39` | **32.5 (-9%)** | **~74** |

4 GPUs keep ~0.3 missed experts per layer, too few for the CPU share to matter (29.7 vs 29.2 ms/window at best).

## What this branch adds

Opt-in flags change outputs slightly (another summation order or precision) and are off by default; everything
else keeps the default outputs.

### Port and memory placement

| Change | Effect |
|---|---|
| ppc64le build and setup (no AVX2 checks, local engine build, gnu++17 ggml-cpu, POWER spin hint) | Runs at all |
| Helper threads leave the host thread's core (they inherited its pin) | 1 GPU: 10K prompt 87 -> **894 tok/s**; 4 GPUs: 2K prompt 38 -> 426 tok/s |
| **Per-socket page-locked arena** (`--resident-budget-gib` with a layer split): one `cudaHostAlloc` per NUMA node, allocated and filled by that node's CPUs; each stage probes its own socket | 4 GPUs decode **37-55 -> 67-74 tok/s** |
| `PinnedArena` from `cudaHostAlloc` on its GPU's node | IQ2_XS experts load at 27.7 GiB/s (registered: 2.4) |
| NVLink rule: a link probed at >= 60 GB/s takes every missed expert (`pcie_frac` 1.0) | 1 GPU decode 33.7 -> 38.7 tok/s |
| The missed-expert fetch on its own stream beside the cached experts (`STRATA_FETCH_OVERLAP`) | 2 GPUs +5% decode |
| **The idle same-socket peer GPU fetches half the misses** over its own NVLink (`STRATA_FETCH_PARTNER`) | 2 GPUs decode **51-58 -> 60-68 tok/s**; outputs byte-identical |
| Prompt chunks pipelined through every layer-split stage (a middle stage waited for the chain below) | 4 GPUs 64K prefill **+56-64%**, bit-identical |
| Stage hand-off threads off CPU 0 | 4 GPUs 64K prefill +21-26% |
| The drafter's prompt pass batched on a layer split and with its K/V ring | +2-6% prefill |

### Volta tensor cores (V100 has FP16 tensor cores only: no BF16, TF32 or INT8 MMA, no `cp.async`)

| Change | Effect |
|---|---|
| **`STRATA_FP16=load`** (+ `STRATA_FP16_GATES=1`): the 483 BF16-form tensors read from the GGUF at load as FP16 (Q8_0/BF16 -> FP16), so every GEMM runs on the tensor cores | prefill **+14-17%**, decode unchanged |
| **`STRATA_ATTN_WMMA=1`**: int8-KV prompt attention on Volta `wmma` | attention per 8K chunk 405 -> 166 ms; 65K prompt **+17%** |
| Decode hc read compiled per token count and weight form (`fused_gr`), loads a row ahead, bank-conflict padding | decode window 32.1 -> 29.9 ms (**+7% decode**), bit-identical |
| `gdn_ab_multi` per token count | 29.85 -> 29.66 ms/window, bit-identical |
| **`STRATA_FUSED_EXPERTS=1`**: fused W4A16 prompt experts - Q4_K gate/up and Q5_1/Q8_0 down dequantized tile by tile into shared memory, `wmma`, SwiGLU in the epilogue, 32 experts per launch pair, no FP16 copy of the expert | per expert 5.5x / 2.7x / 1.7x vs dequant + cuBLAS at 40 / 160 / 320 tokens; prompts **+14-18%** |
| **`STRATA_SELECT_VOLTA=1`**: QSA block selection scores on `wmma` (16 queries share each key read; FP32 split into FP16 hi + lo, 3 products) | scores 8x at 57K context; 65K prompt **+9%** |

**Volta's tensor cores truncate when they accumulate.** A long chain of `mma` into one accumulator (K = 2,560: 160
steps) drifts toward zero; the first fused-expert kernel raised KL by 0.003 on both test texts. Every new kernel
here adds each 16- or 64-wide step's products into the total with ordinary FP32 adds; then it is more accurate than
cuBLAS (down projection RMS error vs FP64 2.4e-7 against 4.2e-6 at 160 tokens per expert).

### POWER9 CPU: VSX, SMT and NUMA

| Change | Effect |
|---|---|
| VSX BF16 router dot | the router lookahead on POWER |
| **`STRATA_VSX_EXPERTS=1`**: multi-token VSX kernels for the CPU's share of decode misses (Q4_K x Q8_K, Q5_1 x Q8_1; each weight block decoded once for all its tokens, FP16 scales by `xvcvhpsp`) | one core vs ggml's POWER9 `vec_dot`: Q4_K **1.7-2.3x**, Q5_1 **4.3-7.7x** (1-3 tokens) |
| **`STRATA_POOL_SMT=2`**: two hardware threads per core in the CPU expert pool | a core's kernel throughput 1.95x (SMT4: 2.1x, but the pool is slower with 80 threads: 239 vs 139 us per layer) |
| **`STRATA_POOL_PER_NODE=1`**: one CPU expert pool per NUMA node; a layer-split stage uses its GPU's socket's pool | CPU cost per expert 250-440 -> ~105 us on 4 GPUs |
| A layer with no CPU entries skips quantizing the activations (ggml's Q8_K quantizer is scalar on POWER) | 2 GPUs decode -3%, default outputs unchanged |

POWER9 SMT4 cores split into two halves; two threads per core use both. One core streams ~13 GB/s; the expert
kernels run at 2.5-3.1 GB/s per thread.

## Quality

Teacher-forced: one request of a 2,048-token (or 32,768-token) prompt, then the next 600 tokens logged position by
position (`STRATA_LOGPOS`, top 64). KL against a reference that reads everything through the FP32 decode path.

| Run | 2K KL | 2K PPL (ref 4.491) | 32K KL | 32K PPL (ref 4.570) |
|---|---:|---:|---:|---:|
| BF16 default | 0.0144 | 4.509 | | |
| `STRATA_FP16=load`, gates F32 | 0.0118 | 4.437 | | |
| best mode (load, FP16 gates, attention WMMA) | 0.0157 | 4.499 | 0.0203 | 4.543 |
| + fused experts | 0.0139 | 4.488 | 0.0227 | 4.521 |
| + fused experts + Volta selection | 0.0139 | 4.488 | **0.0183** | 4.550 |
| 2 GPUs, CPU share (VSX, `--pcie-frac 0.6`) | 0.0165 | 4.502 | | |

All within one text's noise band: changing only the decode weights' format (BF16 -> FP16 from Q8_0) moves KL by 0.013
and 5% of top-1 picks. The 2K text is below the 2,051-cell selection width, so selection does not change it.

## Tried and dropped

| Idea | Result |
|---|---|
| MMQ for Q4_K/Q5_K/Q5_1 prompt experts (`-DSTRATA_MMQ_KQUANTS=ON`) | +17-24% at 2K, ~0 at long context (no INT8 tensor cores); shifts outputs; superseded by the fused experts |
| GDN recurrence: state in registers / shuffles / chunked WY form on tensor cores | correct, but 1.7x / 3.4x / 2.6x slower than the current kernel |
| Expert plan on the GPU (branch `ac922-gpu-plan-wip`) | 32.2 -> 32.7 ms/window: the host plan was already hidden |
| Next-layer expert prefetch (branch `ac922-prefetch-wip`) | 58-61% of misses predicted, but 34.8 -> 39.4 ms/window |
| Uneven layer splits, 16K prompt chunks, longer drafts (`--spec 6`) | no gain (stages balanced; less overlap; acceptance drops) |
| Peer-to-peer stage hand-offs, KV cache near each GPU, on-device planner E-6 | no change |
| `cudaHostRegister` / `cudaMallocManaged` for experts | registered memory reads at 0-48 GB/s with UVM migration on |
| Drafter at FP16 or with Q4_K_M / Q8_0 experts | acceptance unchanged (76-77%) |
| CPU share of misses on 4 GPUs | within noise: too few misses |

## Hardware rules learned on the AC922

- **Page-locked memory that GPUs read must come from `cudaHostAlloc`**, allocated by a thread on the GPU's NUMA node
  (it ignores `numactl --membind`). Pinned copies: 68-72 GB/s from the local socket, 40 from the other.
- **Fill each node's memory from that node's CPUs.** Local pages last written by the other socket read at 45 GB/s.
- **Leave UVM access-counter migration on** (the driver default): with it on, `cudaHostRegister`'d memory is slow,
  `cudaHostAlloc` memory is not.
- **Threads inherit CPU pins**: anything spawned by a pinned host thread must release the pin, or it shares one core.
- GPUs 0-1 sit on node 0, GPUs 2-3 on node 8; each GPU's partner is the other GPU on its socket (no link between pairs).

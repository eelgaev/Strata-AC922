# Experimental: IBM Power System AC922 (POWER9 + 4x V100, NVLink 2.0)

This branch (`ac922`) runs Strata on an IBM AC922: two POWER9 sockets (20 cores each, SMT4, ppc64le) and four
NVIDIA Tesla V100-SXM2 16 GB (sm_70), each GPU joined to its socket by NVLink 2.0 (~72 GB/s one way) and to its
partner GPU by a second NVLink. The CPU and the GPUs share one coherent, unified memory (ATS over NVLink): a GPU
reads the sockets' RAM directly. Tested with Qwen3.8-Flash-Next **UD-Q4_K_XL** (Unsloth 4-bit, 71.7 GiB of experts)
and **IQ2_XS**, on 2 and 4 GPUs.

Everything here was measured on one AC922 (RHEL 8, driver 550.54.15 - ppc64le's last - CUDA 12.4, gcc-toolset-12).
It is not an upstream-supported platform.

- [At a glance](#at-a-glance)
- [Build](#build)
- [Run](#run)
- [Speed](#speed)
- [What this branch adds](#what-this-branch-adds)
- [Quality](#quality)
- [Tried and dropped](#tried-and-dropped)
- [Hardware rules learned on the AC922](#hardware-rules-learned-on-the-ac922)

## At a glance

UD-Q4_K_XL on 4x V100 unless noted. **Bitwise** = greedy output identical to the code before the change;
**FP32-level** = another summation order, error vs FP64 at FP32 level (outputs can differ late in a greedy answer);
**shifts** = changes the numbers more than that (still within the quality noise band, see [Quality](#quality)).

| Optimization | Where | Measured | Numerics | On by |
|---|---|---|---|---|
| Per-socket page-locked arena: one `cudaHostAlloc` per NUMA node, allocated and filled on that node | `expert_source.cpp`, `pinned.cu`, `generate.cpp` | decode 37-55 -> **67-74 tok/s**; experts load 27.7 GiB/s (registered: 2.4) | GPU instead of CPU kernels for the misses | default with `--resident-budget-gib` |
| Helper threads release the host thread's CPU pin | `thread_affinity.hpp`, `prefill.cpp`, `pool.cpp` | 10K prompt on 1 GPU **87 -> 894 tok/s**; 4 GPUs 64K prefill +21-26% | bitwise | default |
| NVLink: >= 60 GB/s links take every missed expert | `generate.cpp` | 1 GPU decode 33.7 -> 38.7 tok/s | GPU instead of CPU kernels | default |
| Missed-expert fetch beside the cached experts (own stream) | `verify.cpp`, `verify_kernels.cu` | 2 GPUs decode +5% | bitwise | default (`STRATA_FETCH_OVERLAP=0` off) |
| The idle peer GPU fetches half of the misses over its NVLink | `verify.cpp`, `verify_kernels.cu` | link 55 -> 104 GB/s; 2 GPUs decode **51-58 -> 60-68 tok/s** | bitwise | default (`STRATA_FETCH_PARTNER=0` off) |
| Prompt chunks pipelined through every layer-split stage | `prefill.cpp` | 4 GPUs 64K prefill **+56-64%** | bitwise | default |
| Drafter prompt pass batched on a layer split | `prefill.cpp`, `generate.cpp` | draft layer 325 -> 37 ms per chunk; prefill +2-6% | bitwise | default |
| BF16-form weights as FP16 at load (tensor cores) | `load_converted.cpp`, `gemm.cu`, `fused_gr.cu`, `fused_gdn.cu`, ... | prefill **+14-17%**, decode equal | shifts (FP16 rounding) | `STRATA_FP16=load STRATA_FP16_GATES=1` |
| Decode hc read per token count and weight form, loads a row ahead, bank padding | `fused_gr.cu` | down 38 -> 18 us; window 32.1 -> 29.9 ms (**+7% decode**) | bitwise | default |
| `gdn_ab_multi` per token count | `verify_kernels.cu` | window 29.85 -> 29.66 ms | bitwise | default |
| **`--prefill-ring 320`** on 4 GPUs (new flag): the prompt path's streamed-expert ring - the engine picked 96 slots here (its pinned share counts the experts already in the GPU caches), about a third of a layer, so the GPU waited on each layer's expert burst | `generate.cpp`, `prefill.cpp` | copy wait 8.5% -> 1.8% of the GPU timeline; prompts 7.8K 2,658 -> 2,953, 18K 4,036 -> 4,566, 65K 5,700 -> **6,386 tok/s** (+11-13%; ring sweep 96..512 peaks at 320) | bitwise (teacher-forced log-probs byte-identical) | `--prefill-ring 320` |
| `--prefill 4096` on 4 GPUs (config): a mid-size prompt becomes several chunks, so the layer-split stages overlap them, and the prompt buffers borrow fewer expert-cache slots | server config | prompts 7.8K **+34%** (2,021 -> 2,713), 18K **+22%** (3,294 -> 4,009), 2K and 65K the same | chunk boundaries differ (KL across chunk sizes 2048-16384: 0.018-0.025, noise) | `--prefill 4096` |
| Shared expert: dead BF16 copy dropped, gate sigmoid fused | `shared_expert.cu`, `verify.cpp` | window -0.75% | bitwise | default |
| Prompt attention on Volta `wmma` (int8 KV) | `qsa_prompt_attn.cu` | attention 2.4-2.7x (405 -> 166 ms per 8K chunk); 65K prompt **+17%** | FP32-level | `STRATA_ATTN_WMMA=1` |
| Fused W4A16 prompt experts (Q4_K / Q5_1 / Q8_0 dequantized in shared memory, `mma.m8n8k4` on a swizzled tile, SwiGLU epilogue, 32 experts per launch) | `fused_expert.cu`, `prefill.cpp` | 5.5x / 2.7x / 1.7x vs dequant + cuBLAS (40 / 160 / 320 tokens per expert); prompts **+14-18%**, then v2 another 6-13% per layer (+2-3% prompts) | FP32-level (more accurate than cuBLAS) | `STRATA_FUSED_EXPERTS=1` |
| QSA block selection scores on `wmma` (FP16 hi + lo split) | `qsa_select.cu` | scores 8x at 57K context; 65K prompt **+9%** | FP32-level | `STRATA_SELECT_VOLTA=1` |
| VSX multi-token CPU expert kernels (Q4_K, Q5_1) | `kq_vsx.cpp`, `native_expert.cpp` | per core vs ggml: Q4_K 1.7-2.3x, Q5_1 **4.3-7.7x** | FP32-level vs ggml (2e-7) | `STRATA_VSX_EXPERTS=1` |
| SMT2 CPU expert pool | `pool.cpp` | per-core throughput 1.95x | bitwise vs 1 thread per core | `STRATA_POOL_SMT=2` |
| One CPU expert pool per NUMA node | `generate.cpp` | CPU cost per expert 250-440 -> 105 us (4 GPUs) | bitwise | `STRATA_POOL_PER_NODE=1` |
| CPU share of decode misses on 2 GPUs (the three rows above + `--pcie-frac 0.6`) | | 2 GPUs window 35.7 -> **32.5 ms (-9%)** | CPU instead of GPU kernels | flags above |
| Skip quantizing activations when no expert goes to the CPU | `expert_source.cpp` | 2 GPUs decode -3% window | bitwise | default |
| VSX BF16 router dot | `portable.cpp` | the router lookahead on POWER | FP32-level | default |

**Long prompts now (4 GPUs, best mode + the opt-in prompt kernels + `--prefill 4096 --prefill-ring 320`, one prompt each, 2026-10-04):** 7.8K 3,295, 18K 4,799, 65K 6,742, **123K 7,020 tok/s**. 2 GPUs: 7.8K 2,061, 18K 2,702, 65K 3,603, 123K 3,642.

**Totals (4 GPUs, llama-benchy, every optimization on, 2026-10-04):** prefill 659-1,265 -> **1,882-5,958 tok/s**,
decode mean 69.5 -> **79.0 tok/s**. 2 GPUs: prefill **1,417-3,564**, decode **69.0**. Against the first working
port: prefill 426-1,121 and decode ~50 tok/s.

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

On 4 GPUs use `"--prefill", "4096", "--prefill-ring", "320"` instead of `"--prefill", "auto"`: mid-size prompts read 22-34% faster and every prompt from 8K up another 11-13% (see At a glance).

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

| Test | First port (`--mmap-experts`) | Per-socket arena, BF16 default (2026-10-02) | Best mode after the 0.1.38 merge | **Every optimization on (2026-10-04)** |
|---|---:|---:|---:|---:|
| Prefill pp2048 | 426 | 659 | 812 | **1,882** |
| Prefill pp8192 | 755 | 1,106 | 1,583 | **3,228** |
| Prefill pp2048 @ 16K | 864 | 1,098 | 1,981 | **3,857** |
| Prefill pp8192 @ 16K | 952 | 1,145 | 2,034 | **4,012** |
| Prefill pp2048 @ 64K | 1,106 | 1,265 | 3,601 | **5,958** |
| Prefill pp8192 @ 64K | 1,121 | 1,262 | 3,320 | **5,768** |
| Decode (mean of the 6 rows) | 50 | 69.5 | 76.7 | **79.0** |

"Every optimization on" is the launch configuration: best mode, `STRATA_FUSED_EXPERTS=1 STRATA_SELECT_VOLTA=1`,
`--prefill 4096 --prefill-ring 320`, with the default-on kernel fixes of 2026-10-04 (fused experts v2, the coalesced
Q8_0 and expert dequantizers). The third column is best mode without the opt-in prompt kernels.

### 2 GPUs, UD-Q4_K_XL, every optimization on (2026-10-04)

`--prefill auto --pcie-frac 0.6 --pool-workers 39`, `STRATA_VSX_EXPERTS=1 STRATA_POOL_SMT=2` and the prompt kernels
above:

| Test | Prefill | Decode |
|---|---:|---:|
| pp2048 | 1,417 | 63.4 |
| pp8192 | 2,227 | 67.7 |
| pp2048 @ 16K | 2,570 | 72.1 |
| pp8192 @ 16K | 2,619 | 70.1 |
| pp2048 @ 64K | 3,564 | 72.4 |
| pp8192 @ 64K | 3,474 | 68.5 |
| **Mean decode** | | **69.0** | The first
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
| **`STRATA_FUSED_EXPERTS=1`**: fused W4A16 prompt experts - Q4_K gate/up and Q5_1/Q8_0 down dequantized tile by tile into shared memory, tensor cores, SwiGLU in the epilogue, 32 experts per launch pair, no FP16 copy of the expert | per expert 5.5x / 2.7x / 1.7x vs dequant + cuBLAS at 40 / 160 / 320 tokens; prompts **+14-18%** |
| Fused experts v2 (same outputs, bit for bit): Q4_K scales read from registers (the byte-indexed header went through local memory and stalled each K step on the prefetch), raw `mma.m8n8k4` on an XOR-swizzled tile (sm_70 `wmma` loads have 8-way bank conflicts at any legal stride), independent accumulators issued back to back, MMAs skipped for rows past an expert's last token | per layer at 40 / 80 / 160 tokens per expert: -9% / -13% / -6%; prompts 18K 4,414 -> 4,537, 65K 6,404 -> 6,540, 123K 6,604 -> 6,783 tok/s |
| Coalesced Q8_0 -> FP16 dequant (default; same values bit for bit): the prompt path dequantizes every dense Q8_0 projection (GDN qkv/gate/out, QSA q/k/v/out, shared expert, hc) before its cuBLAS GEMM, and the generic kernel's thread per 32-value block wrote 2-byte values 64 bytes apart (~107 GB/s). Four threads per block, one 16-byte store each: ~725 GB/s | attn_qkv 0.747 -> 0.110 ms; the model's dense dequant per 4,096-token chunk 106 -> 18 ms; prompts 18K 4,537 -> 4,708, 65K 6,540 -> 6,766, 123K 6,783 -> 6,985 tok/s |
| Expert dequant (`iq_dequant_*`, default; same values bit for bit): 8 superblocks per 256-thread block and 16/8-byte stores from registers instead of 32-thread blocks with 2-byte stores. The prompt path without `STRATA_FUSED_EXPERTS` dequantizes every streamed expert before cuBLAS | per expert Q4_K gate/up 29.4 -> 14.9 us, Q5_1 down 15.5 -> 5.9 us, Q5_K 23.5 -> 19.0, Q8_0 23.5 -> 13.5; default-mode prompts (4 GPUs) 7.8K 2,124 -> 2,380, 18K 3,189 -> 3,573, 65K 4,240 -> 4,758 tok/s (+11-12%) |
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
| A per-prompt chunk size (about 1.5 chunks per stage) with 8K buffers | below a fixed `--prefill 4096`: the 8K buffers borrow more cache slots |
| Merged host hand-shake kernels in the verify window (wait + copy plan, wait + CPU rows + hits, wait + rebase) | bit-identical, but 29.0 -> 29.1 ms/window: the graph nodes were not on the critical path |
| Q8_0 decode GEMVs with 2 or 4 rows per block (bit-identical) | slower than 1 row per block (fewer blocks in flight) |
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

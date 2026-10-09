# Experimental: IBM Power System AC922 (POWER9 + 4x V100, NVLink 2.0)

This branch (`ac922`) runs Strata on an IBM AC922: two POWER9 sockets (20 cores each, SMT4, ppc64le) and four
NVIDIA Tesla V100-SXM2 16 GB (sm_70), each GPU joined to its socket by NVLink 2.0 (~72 GB/s one way) and to its
partner GPU by a second NVLink. The CPU and the GPUs share one coherent, unified memory (ATS over NVLink): a GPU
reads the sockets' RAM directly. Tested with Qwen3.8-Flash-Next **UD-Q4_K_XL** (Unsloth 4-bit, 71.7 GiB of experts)
and **IQ2_XS**, on 2 and 4 GPUs.

Everything here was measured on one AC922 (RHEL 8, driver 550.54.15 - ppc64le's last - CUDA 12.4, gcc-toolset-12).
It is not an upstream-supported platform. The branch tracks upstream: last merged v0.1.41 (2026-10-09, see
[Merging upstream](#merging-upstream)).

- [At a glance](#at-a-glance)
- [Build](#build)
- [Run](#run)
- [Speed](#speed)
- [What this branch adds](#what-this-branch-adds)
- [Quality](#quality)
- [Tried and dropped](#tried-and-dropped)
- [Merging upstream](#merging-upstream)
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
| Prompt attention on `mma.m8n8k4`: Klaus Friedel's (fks) kernel from PR #600 ([commit ee0843d](https://github.com/fks/Strata/commit/ee0843da07ee8da96fa0b5afebfdd8dde6c0b587)), with the hi / lo halves of its split operands in separate accumulator chains | `qsa_prompt_attn.cu` | 12.4 vs 14.2 ms per 2,048-query chunk (32K); prompts 7.8K +2.0%, 18K +2.7%, 65K +1.4%, 123K +1.1% | error vs FP64 2.3e-6 (wmma 6.6e-6); KL 32K 0.0232 -> 0.0220, 2K 0.0144 -> 0.0154 | default on V100 since the v0.1.39 merge (upstream took #600; `STRATA_ATTN_WMMA=1` without `STRATA_ATTN_M884=1` picks the wmma kernel) |
| `STRATA_PREFILL_ADAPT=F` (per-prompt chunk on a layer split: about F chunks per stage, at least `STRATA_PREFILL_ADAPT_MIN`, at most `--prefill`'s chunk) | `prefill.cpp` | 4 GPUs (F 1.5, min 2048): 4K +16%, 7.8K +6%. 2 GPUs (F 2.0, min 4096): 7.8K +21%, 18K +13%. Longer prompts the same | other chunk boundaries (KL within noise) | opt-in |
| `--prefill 3072` on 4 GPUs, `--prefill-ring 480` on 2 GPUs (config, re-swept 2026-10-04 after the kernel work) | server config | 4 GPUs vs 4096: 7.8K +9%, 18K +11%, 65K same, 123K -1%. 2 GPUs (auto = 8,192-token chunks) with ring 480: 2K +29%, 7.8K-123K +8-9% | same outputs (ring); chunk size as before | config |
| PLE rows uploaded at a stage's first layer (they queued behind layer 1's expert copies), the batch gather on 8 threads (`STRATA_PLE_GATHER_THREADS`) | `prefill.cpp`, `ngram.cpp` | layer 1 41 -> 28 ms per chunk; gather 27-40 -> 4.5 ms; prompts +1% (the stages are level now) | bitwise | default |
| Prompt path: the host grouping waits for the router's ids (an event), not the whole stream | `prefill.cpp` | ~1 ms a layer of GPU idle gone; prompts +0.2-0.7% | bitwise | default |
| Async commit on a layer split (each stage launches its commit graph on its own stream; the next window follows on it) | `verify.cpp`, `generate.cpp` | window 29.0 -> 28.25 ms (**-2.6%**) | bitwise | default (`STRATA_COMMIT_SYNC=1` off) |
| Upstream v0.1.39 merged (#646 decode kernels and zero-doorbell verify graph, #603 long-context top-k, ...) | | window 28.25 -> **26.9 ms** in the decode test; prompts +1% | bitwise vs pre-merge | default |
| A verify window commits only the tokens it hands out (upstream PR #652 by anon761, cherry-picked) | `serve` | an answer cut by `max_tokens` or an end of turn inside a window no longer loses the live session: 8 of 8 follow-ups resume (before 7 of 8; a miss re-read the turn) | bitwise | default |
| A layer split's last stage books the MTP draft head (276 MiB) when it sizes its expert cache; the start-up VRAM check reads every stage | `generate.cpp` | 2 GPUs: the last card served with 140 MiB free, and a verify-window graph (~13 MiB each on a 24-layer stage) first needed after a long prompt failed to instantiate - the engine exited twice in one benchy run. Now 280 MiB left at the end of the run, no exits | cache 92 slots smaller on the last stage (outputs can differ) | default |
| Shared expert: dead BF16 copy dropped, gate sigmoid fused | `shared_expert.cu`, `verify.cpp` | window -0.75% | bitwise | default |
| Prompt attention on Volta `wmma` (int8 KV; superseded by the m8n8k4 kernel above) | `qsa_prompt_attn.cu` | attention 2.4-2.7x (405 -> 166 ms per 8K chunk); 65K prompt **+17%** | FP32-level | `STRATA_ATTN_WMMA=1` |
| Fused W4A16 prompt experts (Q4_K / Q5_1 / Q8_0 dequantized in shared memory, `mma.m8n8k4` on a swizzled tile, SwiGLU epilogue, 32 experts per launch) | `fused_expert.cu`, `prefill.cpp` | 5.5x / 2.7x / 1.7x vs dequant + cuBLAS (40 / 160 / 320 tokens per expert); prompts **+14-18%**, then v2 another 6-13% per layer (+2-3% prompts), Q5_K gate/up +1-3% | FP32-level (more accurate than cuBLAS) | `STRATA_FUSED_EXPERTS=1` |
| Coalesced dequant: dense Q8_0 projections (4 threads per block, 16-byte stores) and streamed experts (8 superblocks per block) | `dequant_bf16.cu`, `iq_kernels.cu` | Q8_0 ~107 -> ~725 GB/s; prompts +3-4%, default mode (no fused experts) +11-12% | bitwise | default |
| QSA block selection scores on `wmma` (FP16 hi + lo split) | `qsa_select.cu` | scores 8x at 57K context; 65K prompt **+9%** | FP32-level | `STRATA_SELECT_VOLTA=1` |
| VSX multi-token CPU expert kernels (Q4_K, Q5_1) | `kq_vsx.cpp`, `native_expert.cpp` | per core vs ggml: Q4_K 1.7-2.3x, Q5_1 **4.3-7.7x** | FP32-level vs ggml (2e-7) | `STRATA_VSX_EXPERTS=1` |
| SMT2 CPU expert pool | `pool.cpp` | per-core throughput 1.95x | bitwise vs 1 thread per core | `STRATA_POOL_SMT=2` |
| One CPU expert pool per NUMA node | `generate.cpp` | CPU cost per expert 250-440 -> 105 us (4 GPUs) | bitwise | `STRATA_POOL_PER_NODE=1` |
| CPU share of decode misses on 2 GPUs (the three rows above + `--pcie-frac 0.6`) | | 2 GPUs window 35.7 -> **32.5 ms (-9%)** | CPU instead of GPU kernels | flags above |
| Skip quantizing activations when no expert goes to the CPU | `expert_source.cpp` | 2 GPUs decode -3% window | bitwise | default |
| VSX BF16 router dot | `portable.cpp` | the router lookahead on POWER | FP32-level | default |

**Long prompts now (launch configuration, one prompt each, 2026-10-04):** 7.8K 4,000, 18K 5,551, 65K 7,161, 123K 7,332, **249K 7,299 tok/s** (4 GPUs); 2 GPUs: 7.8K 2,820, 18K 3,486, 65K 4,033, 123K 4,027.

**Totals (llama-benchy, launch configuration, 2026-10-04):** 4 GPUs prefill 659-1,265 -> **1,967-6,353 tok/s**,
decode mean 69.5 -> **83.4 tok/s**; 2 GPUs prefill **1,845-3,785**, decode **73.4**. Against the first working
port: prefill 426-1,121 and decode ~50 tok/s. Greedy decode (3 answers of 384 tokens): **85-87 tok/s** on 4 GPUs,
72-75 on 2.

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
          "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
          "--prefill", "3072", "--prefill-ring", "320",
          "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/path/to/mtp/rt",
          "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768", "--ple-io", "ram",
          "--resident-budget-gib", "72"],
 "gpu": [0, 1, 2, 3],
 "layer_split": "auto"
}
```

Settings re-swept 2026-10-04 after the kernel work: `--prefill 3072 --prefill-ring 320` on 4 GPUs (over 4096: 7.8K
+9%, 18K +11%); `--spec 4 --spec-min-p 0.5` stays best over spec 3-6 x min-p 0.3-0.7 on essay, code and Q&A text.

The environment of the server:

```sh
# best mode: default outputs change slightly (FP16 instead of BF16 weights), quality the same (see Quality)
export STRATA_FP16=load STRATA_FP16_GATES=1
# opt-in, faster prompts (another summation order; quality checked)
export STRATA_FUSED_EXPERTS=1 STRATA_SELECT_VOLTA=1
# opt-in: a per-prompt chunk size on a layer split (4K-18K prompts +6-16%)
export STRATA_PREFILL_ADAPT=1.5 STRATA_PREFILL_ADAPT_MIN=2048
# since the v0.1.40 merge: keep the 12/24/36 layer split and the old short-prompt upload schedule (below)
export STRATA_SPLIT_CURVE=0139 STRATA_SPLIT_BALANCE=1 STRATA_PREFILL_STREAM_AHEAD=0
```

The last three keep upstream's defaults in the code and set what is faster on this machine:

- `STRATA_SPLIT_CURVE=0139` + `STRATA_SPLIT_BALANCE=1`: v0.1.40's four-way placement scorer picks 11/24/36 here (a
  13-layer first stage that paces every prompt: 65K prompts 6,400 instead of 7,150 tok/s, -10%). The 0.1.39 curve,
  plus "a placement within 1% of the best predicted decode window goes to the most balanced one", keeps 12/24/36.
  Check the start-up line `layer split auto: K=12,24,36`.
- `STRATA_PREFILL_STREAM_AHEAD=0`: v0.1.40's routed-only upload schedule for chunks below ~1K tokens costs ~110 ms on
  an 80-token prompt on 2 GPUs (450 tokens: 620 instead of 830 ms; 4 GPUs: -4% to -13%). Off, the timings are the
  pre-merge ones; prompts of 1K tokens and more do not use it.
- Not set: `STRATA_SM70_TABLE=1` (v0.1.41, PR 1401's Volta decode kernels) is bitwise the default here, but does not
  change the speed of UD-Q4_K_XL: its matvec table covers Q4_K/Q5_K/Q6_K/IQ4_XS dense weights (this model's are all
  Q8_0), expert mode 8 covers IQ2 experts, and its fast norm/up reads BF16-form weights only (it stays off with
  `STRATA_FP16=load`).

Prompt attention on the V100s takes the m8n8k4 kernel by default (see At a glance); `STRATA_ATTN_WMMA=1` picks the
older wmma kernel instead, unless `STRATA_ATTN_M884=1` is also set.

On **2 GPUs** (`"gpu": [0, 1]`, one socket) keep `"--prefill", "auto"` (8,192-token chunks), and add the CPU share of
the decode misses:

```sh
export STRATA_VSX_EXPERTS=1 STRATA_POOL_SMT=2
export STRATA_PREFILL_ADAPT=2.0 STRATA_PREFILL_ADAPT_MIN=4096
# and in args: "--prefill", "auto", "--prefill-ring", "480", "--pcie-frac", "0.6", "--pool-workers", "39"
```

**Long context:** `"--max-context", "262144"` works on 4 GPUs with the same `--kv-resident 32768` (the KV past it
streams from RAM). A 248,934-token prompt reads at 7,299 tok/s (34.1 s); a follow-up at that depth reuses all
248,865 tokens and answers after 0.3 s; decode at that depth ~65 tok/s (85-87 short).

Startup takes 3-5 minutes (the 72 GiB arena is copied in, the 27 GiB PLE table locked).

## Speed

Method: [llama-benchy](https://github.com/eugr/llama-benchy), `--pp 2048 8192 --tg 128 --depth 0 16384 65536 --runs 3
--no-cache`; prefill and decode taken from the server's log (`strata serve: prompt ... read in ... generated in`),
3 runs averaged after a warm-up request. Decode is with MTP speculative decoding (`--spec 4`, 73-82% accepted).

### 4 GPUs, UD-Q4_K_XL: where it started and where it is

| Test | First port (`--mmap-experts`) | Per-socket arena, BF16 default (2026-10-02) | Best mode after the 0.1.38 merge | Opt-in prompt kernels, `--prefill 4096 --prefill-ring 320` (2026-10-04 morning) | **Launch config, v0.1.39 (2026-10-04)** |
|---|---:|---:|---:|---:|---:|
| Prefill pp2048 | 426 | 659 | 812 | 1,882 | **1,967** |
| Prefill pp8192 | 755 | 1,106 | 1,583 | 3,228 | **4,079** |
| Prefill pp2048 @ 16K | 864 | 1,098 | 1,981 | 3,857 | **4,269** |
| Prefill pp8192 @ 16K | 952 | 1,145 | 2,034 | 4,012 | **4,719** |
| Prefill pp2048 @ 64K | 1,106 | 1,265 | 3,601 | 5,958 | **6,351** |
| Prefill pp8192 @ 64K | 1,121 | 1,262 | 3,320 | 5,768 | **6,353** |
| Decode (mean of the 6 rows) | 50 | 69.5 | 76.7 | 79.0 | **83.4** |

The launch configuration is [Run](#run)'s: best mode, `STRATA_FUSED_EXPERTS=1 STRATA_SELECT_VOLTA=1`,
`STRATA_PREFILL_ADAPT=1.5`, `--prefill 3072 --prefill-ring 320`, the m8n8k4 prompt attention, and everything on by
default. The third column is best mode without the opt-in prompt kernels. Decode in benchy's runs averages short
answers with a prompt still settling; greedy answers of 384 tokens run at 85-87 tok/s.

### Single prompts (4 and 2 GPUs, launch configuration, server log)

| Prompt | 4 GPUs | 2 GPUs |
|---|---:|---:|
| 7,879 tokens | 4,000 tok/s | 2,820 tok/s |
| 18,268 tokens | 5,551 tok/s | 3,486 tok/s |
| 64,998 tokens | 7,161 tok/s | 4,033 tok/s |
| 123,263 tokens | 7,332 tok/s | 4,027 tok/s |
| 248,934 tokens (`--max-context 262144`) | **7,299 tok/s** | |

### 2 GPUs, UD-Q4_K_XL, launch configuration

[Run](#run)'s 2-GPU settings: `--prefill auto --prefill-ring 480 --pcie-frac 0.6 --pool-workers 39`,
`STRATA_VSX_EXPERTS=1 STRATA_POOL_SMT=2 STRATA_PREFILL_ADAPT=2.0` and the prompt kernels above.

| Test | Prefill (2026-10-04 morning) | Decode | **Prefill (v0.1.39)** | **Decode** |
|---|---:|---:|---:|---:|
| pp2048 | 1,417 | 63.4 | **1,845** | **72.7** |
| pp8192 | 2,227 | 67.7 | **2,784** | **77.5** |
| pp2048 @ 16K | 2,570 | 72.1 | **2,955** | **73.9** |
| pp8192 @ 16K | 2,619 | 70.1 | **3,099** | **73.6** |
| pp2048 @ 64K | 3,564 | 72.4 | **3,785** | **71.2** |
| pp8192 @ 64K | 3,474 | 68.5 | **3,774** | **71.2** |
| **Mean decode** | | 69.0 | | **73.4** |

### Long prompts with the opt-in prompt kernels (4 GPUs, one 64,907-token prompt, server log)

| Mode | Read |
|---|---:|
| best mode | 4,558 tok/s |
| + `STRATA_FUSED_EXPERTS=1` | 5,206 tok/s (+14%) |
| + `STRATA_SELECT_VOLTA=1` | **5,673 tok/s (+24%)** |

A 7,846-token prompt: 1,675 -> 1,951 tok/s with the fused experts (+16%; selection does not apply below 2K cells).

### 2 GPUs (one socket, GPUs 0-1), UD-Q4_K_XL

| Test | Earlier on this branch (`94d7556`, BF16) | Best mode (no opt-in prompt kernels) |
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
| **`--prefill-ring N`** (new flag): the streamed-expert ring of the prompt path. The engine sized it from its pinned share, which counts the experts already in the GPU caches (96 slots on 4 GPUs, a third of a layer) | copy wait 8.5% -> 1.8% of the GPU timeline; 4 GPUs prompts +11-13% at 320, 2 GPUs +8-29% at 480; same outputs |
| `STRATA_PREFILL_ADAPT=F`: a per-prompt chunk size on a layer split. The 4-stage pipeline fills and drains for ~3 chunk-times a prompt (~32% of an 18K prompt), so mid-size prompts take smaller chunks | 4 GPUs 4K +16%, 7.8K +6%; 2 GPUs 7.8K +21%, 18K +13% |
| PLE rows uploaded at a stage's first layer, not at layer 1 behind its expert copies; the batch gather on 8 threads | stage 0's layer 1 41 -> 28 ms per chunk (the stages level), prompts +1%, same outputs |
| The host grouping of a layer's (token, expert) pairs waits for an event on the router's ids, not the whole stream (the shared expert now runs while the host sorts) | ~1 ms GPU idle a layer gone, same outputs |
| Async commit on a layer split: each stage launches its commit graph on its own stream instead of four round trips a window (`STRATA_COMMIT_SYNC=1`: the old wait) | decode window 29.0 -> 28.25 ms, same outputs |
| The last stage of a layer split books the MTP draft head (276 MiB) in its cache sizing, and the start-up VRAM check reads every stage: on 2 GPUs the second card served with 140 MiB free and a verify-window graph needed after a long prompt failed to instantiate ("verify: instantiate: out of memory", the engine restarted) | 2 GPUs: a full benchy run without an engine exit; 280 MiB free at its end |
| A verify window commits only the tokens it hands out (upstream PR #652 by [anon761](https://github.com/Niko1221/Strata/pull/652), cherry-picked) | a follow-up to an answer cut inside a window resumes the live session (8 of 8, ~180 ms) instead of re-reading the turn |

### Volta tensor cores (V100 has FP16 tensor cores only: no BF16, TF32 or INT8 MMA, no `cp.async`)

| Change | Effect |
|---|---|
| **`STRATA_FP16=load`** (+ `STRATA_FP16_GATES=1`): the 483 BF16-form tensors read from the GGUF at load as FP16 (Q8_0/BF16 -> FP16), so every GEMM runs on the tensor cores | prefill **+14-17%**, decode unchanged |
| **`STRATA_ATTN_WMMA=1`**: int8-KV prompt attention on Volta `wmma` | attention per 8K chunk 405 -> 166 ms; 65K prompt **+17%** |
| Prompt attention on `mma.m8n8k4`: fks's PR #600 kernel ([commit ee0843d](https://github.com/fks/Strata/commit/ee0843da07ee8da96fa0b5afebfdd8dde6c0b587)), now upstream's Volta path, with the hi and lo halves of the split q (in q.k) and p (in p.v) in their own accumulator chains. In one chain Volta's truncating accumulation dropped most of the lo half (error vs FP64 5.1e-6) | 12.4 vs 14.2 ms (wmma) per 2,048-query chunk at 32K; error vs FP64 **2.3e-6** (the FP32 kernel: 2.4e-6); prompts +1-3% |
| Decode hc read compiled per token count and weight form (`fused_gr`), loads a row ahead, bank-conflict padding | decode window 32.1 -> 29.9 ms (**+7% decode**), bit-identical |
| `gdn_ab_multi` per token count | 29.85 -> 29.66 ms/window, bit-identical |
| **`STRATA_FUSED_EXPERTS=1`**: fused W4A16 prompt experts - Q4_K gate/up and Q5_1/Q8_0 down dequantized tile by tile into shared memory, tensor cores, SwiGLU in the epilogue, 32 experts per launch pair, no FP16 copy of the expert | per expert 5.5x / 2.7x / 1.7x vs dequant + cuBLAS at 40 / 160 / 320 tokens; prompts **+14-18%** |
| Fused experts v2 (same outputs, bit for bit): Q4_K scales read from registers (the byte-indexed header went through local memory and stalled each K step on the prefetch), raw `mma.m8n8k4` on an XOR-swizzled tile (sm_70 `wmma` loads have 8-way bank conflicts at any legal stride), independent accumulators issued back to back, MMAs skipped for rows past an expert's last token | per layer at 40 / 80 / 160 tokens per expert: -9% / -13% / -6%; prompts 18K 4,414 -> 4,537, 65K 6,404 -> 6,540, 123K 6,604 -> 6,783 tok/s |
| Fused experts: Q5_K gate/up: UD-Q4_K_XL's one Q5_K expert layer (layer 2) ran on dequant + cuBLAS on the first pipeline stage, already the slowest (513 vs 470-476 ms per 4,096-token chunk) | stage 0 513 -> 498 ms per chunk; prompts 18K 4,843 -> 4,902, 65K 6,816 -> 6,978, 123K 7,048 -> 7,255 tok/s; KL 32K 0.0219 -> 0.0232, 2K 0.0139 -> 0.0144 (PPL slightly better on both) |
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
| + Q5_K fused gate/up | 0.0144 | 4.483 | 0.0232 | 4.585 |
| + m8n8k4 attention (**launch configuration**; v0.1.39 merge byte-identical) | 0.0154 | 4.509 | 0.0220 | 4.556 |
| 2 GPUs, CPU share (VSX, `--pcie-frac 0.6`) | 0.0165 | 4.502 | | |

The launch configuration's worst position on the 32K text is KL 0.58 (wmma attention: 1.80), top-1 agreement
95.3%. All within one text's noise band: changing only the decode weights' format (BF16 -> FP16 from Q8_0) moves KL by 0.013
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
| 2 stages of 2 GPUs (each pair splitting a layer's experts) instead of 4 stages | not built: modeled from per-layer times, the dense ~70% of a layer stays on one GPU; break-even near 6K tokens, a 65K prompt ~40% slower |
| The MoE combine folded into the fused down kernel | no gain, reverted |
| hc read: up-projection and mix fused into one kernel (float4 epilogue, `__ldcs` streaming loads) | correct, no gain end to end |
| Fused experts: registers capped at 128, double-buffered tiles | spills (2x slower); double buffering slower (occupancy) |
| BF16 cuBLAS GEMMs | 11.8 TFLOPS against FP16's 98 on V100 (no BF16 tensor cores): the reason for `STRATA_FP16=load` |

## Merging upstream

The branch merges upstream's `main` (never rebases: the fork's history is public). v0.1.39 (2026-10-04, 217 commits,
23 files with conflicts) was resolved hunk by hand, then checked:

- every line either side added since the last merge is in the result or replaced on purpose (a script over all 45
  files both sides changed);
- all parity tests (iq, native experts on the real shards, mmvq, gr, gdn, qsa, top-k, shared expert, kv, sampler, ...);
- teacher-forced 2K and 32K logs byte-identical to the build before the merge;
- decode texts identical on 4 and 2 GPUs.

Rules worth keeping for the next merge:

- Upstream's new x86 intrinsics go through the portable helpers (`strata_store_fence`, `strata_cpu_pause`); its AVX2
  start-up checks and CPU probes behind x86 guards or with non-x86 stubs; the router lookahead keeps `portable.cpp`'s
  VSX dot on POWER (upstream's dispatch falls back to scalar code there).
- `STRATA_FP16=load` stores some weights as FP16: any new upstream kernel that assumes BF16 weights needs the
  form-aware read (`wform`).
- The 2-GPU fetch overlap and the partner GPU's fetch live in verify.cpp's non-resident branch; mapped buffers stay
  `cudaHostAllocPortable` (the partner reads them).
- Thread pools and helper threads keep `release_inherited_pin` / `note_host_pin` (a POWER thread starts on its
  creator's pinned core).
- README.md: the fork's text sits above upstream's, which stays unchanged below the line, so it merges cleanly.

v0.1.40 (2026-10-07..09, 274 + 25 first-parent merges, in 14 parts) and v0.1.41 (2026-10-09, 128 commits, 4 parts) were
merged in small parts, each one checked the same way (lost lines, ctest, teacher-forced 2K and 32K byte-identical at
equal cache slots, 4- and 2-GPU speeds up to 123K tokens, short prompts on 2 GPUs). Upstream rewrote its history once
(the commit trailers removed, same trees): the rewritten twin of the last merged commit was recorded with
`git merge -s ours`, then merging went on normally. More rules from those merges:

- Check the layer split after a merge that touches the placement search (`layer split auto: K=12,24,36`).
- Check short prompts on 2 GPUs: a schedule change for chunks below 1K tokens only shows there.
- Upstream code that assumes one contiguous resident arena goes through `complement_blob` / `complement_at` /
  `complement_dev_at` (the arena is one segment per NUMA node here).
- Upstream's prefill now and then adds work right after the router readback; here the router ids are copied early
  with an event the host waits on, so anything the host reads after that wait must be queued before the event.
- Greedy outputs depend on the cache slot count of every card: match the 4 per-card `cache N slots` lines before
  comparing (`--vram-reserve-mib` for the first card, `--vram-reserve-later-mib` for the others). Decode texts of
  served answers are not reproducible start to start even then (the cache adapts in the background); the
  teacher-forced logs are, and are the check.

## Hardware rules learned on the AC922

- **Page-locked memory that GPUs read must come from `cudaHostAlloc`**, allocated by a thread on the GPU's NUMA node
  (it ignores `numactl --membind`). Pinned copies: 68-72 GB/s from the local socket, 40 from the other.
- **Fill each node's memory from that node's CPUs.** Local pages last written by the other socket read at 45 GB/s.
- **Leave UVM access-counter migration on** (the driver default): with it on, `cudaHostRegister`'d memory is slow,
  `cudaHostAlloc` memory is not.
- **Threads inherit CPU pins**: anything spawned by a pinned host thread must release the pin, or it shares one core.
- GPUs 0-1 sit on node 0, GPUs 2-3 on node 8; each GPU's partner is the other GPU on its socket (no link between pairs).

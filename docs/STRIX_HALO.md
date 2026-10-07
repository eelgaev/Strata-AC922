# Strix Halo (Ryzen AI Max, gfx1151): build from source on Linux

Strata runs on the Ryzen AI Max "Strix Halo" APUs (Radeon 8060S / 8050S, RDNA3.5, `gfx1151`, one pool of LPDDR5X shared by the
CPU and the GPU) through the same HIP backend as the Radeon cards in [AMD_HIP.md](AMD_HIP.md). **Status: experimental.**
Everything here was built and measured on one machine, the maintainers' Strix Halo box (Ryzen AI Max+ 395, 128 GB,
Ubuntu 26.04, kernel 7.0, ROCm 7.14.1). There is no ready-made engine for it and setup (`setup.sh`) does not offer it:
it lists integrated Radeon GPUs as not supported. You build the engine yourself, as below.

What the engine does differently on this chip:

- **Unified memory.** The GPU's "VRAM" is the GTT pool of the same RAM. `device_free_bytes()` (the figure `--expert-cache auto` and
  the memory plan use) counts what Linux can give back (`MemAvailable` less 6 GiB for the system, `STRATA_UMA_HEADROOM_GIB`
  changes the 6) instead of only `MemFree`, per device, for any integrated GPU on Linux (this applies to a DGX Spark too).
- **gfx11 matrix cores.** The prompt attention, the block scorer, the prompt GEMMs and the prompt experts have WMMA kernels for
  gfx1100 / 1101 / 1102 / 1150 / 1151. A gfx11 part outside that list (gfx1103, gfx1152) is not given them: it takes the portable path.
- **The hipBLASLt table** `tools/hip/gfx1151-hipblaslt-100401.txt` (ROCm 7.14.1's hipBLASLt 1.x `100401`), including the prompt shapes
  at `--prefill 16384` (T 16383 / 16384).

## 1. The toolchain (no root needed)

ROCm 7.14.1 for gfx1151 from AMD's TheRock builds, extracted into a folder of your own:

```sh
mkdir -p ~/rocm && cd ~/rocm
curl -LO https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz   # 1.7 GB
sha256sum therock-dist-linux-gfx1151-7.14.1.tar.gz   # c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00
mkdir install && tar -xzf therock-dist-linux-gfx1151-7.14.1.tar.gz -C install
export SDK=$HOME/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
```

You also need CMake 3.24+, Ninja, git and a C++ host toolchain with libstdc++ headers (`sudo apt install build-essential cmake ninja-build git`).
The maintainers' box has no system compiler: they point ROCm's clang at a user-space copy of gcc 15 with
`--gcc-toolchain=<dir>/usr` (added to `CMAKE_HIP_FLAGS` below; leave it out when the system has one).

The kernel's amdgpu driver and `/dev/kfd` access (your user in the `render` and `video` groups) are all the GPU side needs.
To let the GPU use most of a 128 GB machine the maintainers' box boots with `ttm.pages_limit=29360128 ttm.page_pool_size=29360128`
(a 112 GiB GTT; the default is about half of the RAM) and `amd_iommu=off`; both are host settings the engine does not need to run,
only to have the room.

## 2. The engine

llama.cpp's `ggml` sources are pinned by this repository: CMake fetches the revision it needs. For an offline build, point
`STRATA_GGML_DIR` at a checkout of exactly that revision.

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
cmake -S . -B build-halo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-halo --target strata -j 8        # about 25 minutes the first time
```

`gfx1151` is "unvalidated" in the build's own wording (it warns). `STRATA_PREFILL_MMQ=ON` builds ggml's MMQ kernels the prompt
experts need. The tests: `cmake --build build-halo -j 8` builds them all, `ctest --test-dir build-halo` runs them;
`hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm` need the table (next section) and skip without it.

## 3. The prompt table, and running

Hand the engine the table (`setup` does it for the supported cards; for gfx1151 you do it): without it the prompt's dense matrix
products use plain hipBLAS and are slower.

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1151-hipblaslt-100401.txt
```

The engine accepts it only for the architecture and the hipBLASLt version named on the table's second line
(`STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401`: ROCm 7.14.1) and says so at start-up (`prefill gemm: hipBLASLt tuning enabled (90 rows,
gfx1151, version 100401)`). Then run `strata` with the same arguments as on any other card (the model packs, `--serve`, the
settings of [DETAILS.md](DETAILS.md)); a pack of the Unsloth UD-IQ4_XS or UD-Q4_K_XL GGUFs is what the numbers below used.

## 4. What is on by default on gfx1151

At start the engine looks at GPU 0; on a gfx1151 it sets the switches below unless you set them yourself
(`STRATA_GFX1151_DEFAULTS=0` turns the whole table off, and the start-up line names what was set). **Every one of them gave
byte-identical output on the maintainers' box** (the same token ids, 4K / 32K / 64K and four models, and a bitwise harness per kernel):
they change speed, not answers. Nothing here applies to CUDA or to another AMD architecture.

| Switch | What it does |
|---|---|
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY` | the prompt's gated-delta-net recurrence: four lanes per column, two columns per lane, the convolution and the norms fused |
| `STRATA_HCD_EXACT` | the hyper-connection down projection (N 320, K 10240) in a WMMA kernel that reproduces hipBLASLt solution 1176 / 1177's summation order (`hip_prefill_hcd_exact_parity` checks it bit for bit); it runs only where the table makes hipBLASLt pick one of those two, and says once if it cannot |
| `STRATA_CVEC_FUSE`, `STRATA_PF_PAD` | a steered layer's write, control vector and next norm in one pass; padded GEMM row strides |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE` | the grouped decode experts (IQ3_S / IQ4_NL), single-butterfly warp sums, fewer launches around the shared expert and the KV append |
| `STRATA_GDN_SPLIT`, `STRATA_QFUSE`, `STRATA_PLE_BATCH` | the GDN step over four blocks per head, activation images written by their producers, the PLE key / value projections of a verify window at once |

## 5. What is not on by default (it changes bits)

These are faster on gfx1151 and **round differently** from the default path (a different summation order or a half-precision
intermediate), so they are opt-in and the maintainers gate them with KL against the default (`STRATA_PF_SWITCH_MIN_T=4096` keeps
the short prompts on the default numerics, since these pay on long ones). The maintainers' fast configuration sets all of
them:

```sh
export STRATA_PF_FUSED=1        # the prompt experts on the matrix cores (IQ2 / IQ3 / IQ4_XS gate/up, Q2_0 / IQ4_NL / Q8_0 down)
export STRATA_PF_GEMM=1         # the prompt's FP16 projections on the 128 x 256 WMMA GEMM
export STRATA_HC_UPMIX=1        # the hyper-connection up projection with its epilogue
export STRATA_PA_FAST=1         # the prompt attention with single FP16 q and p
export STRATA_HIP_WMMA=1        # the prompt attention on matrix cores
export STRATA_SELECT_WMMA=1     # the block scorer on matrix cores
export STRATA_HC_Q8=1           # the hyper-connection read from the GGUF's own Q8_0 projections
export STRATA_PF_SWITCH_MIN_T=4096
```

Two engine flags help on long contexts and are not Strix-specific: `--mtp-window 8192` (the draft layer attends to the last 8,192
cells: 64K output +3.5%, 32K -0.4%) and `--spec`, `--lookup-chain`, `--mtp-q4` (see [DETAILS.md](DETAILS.md)).

## 6. Measured

Ryzen AI Max+ 395 (128 GB, gfx1151), the maintainers' fast configuration of section 5 on top of the defaults of section 4, `--spec 4`,
`--mtp`, `--lookup-chain 3`, `--mtp-q4 all`, `--prefill 16384`, int8 KV, all experts in the unified memory. Prompt tok/s is
(prompt tokens - 1) / prompt time, output tok/s is the tokens / decode time; medians of 3 runs, one fresh process per run, the two
engines interleaved. "Before" is the maintainers' earlier engine branch (the same switches set by hand), "now" is this release.
Every pair printed the same token ids.

| Model | Context | Prompt before -> now | Output before -> now |
|---|---|---|---|
| UD-IQ4_XS | 8K | 1,212 -> 1,293 (+6.7%) | 51.7 -> 53.8 (+4.0%) |
| UD-IQ4_XS | 128K | 1,241 -> 1,320 (+6.4%) | 49.3 -> 51.4 (+4.2%) |
| UD-Q4_K_XL | 8K | 1,136 -> 1,169 (+2.9%) | 50.6 -> 52.6 (+3.9%) |
| UD-Q4_K_XL | 128K | 1,220 -> 1,248 (+2.4%) | 41.6 -> 43.3 (+3.9%) |
| IQ3_S | 8K | 1,202 -> 1,242 (+3.4%) | 56.2 -> 59.7 (+6.3%) |
| IQ3_S | 128K | 1,254 -> 1,312 (+4.6%) | 40.6 -> 42.8 (+5.5%) |
| IQ3_XXS | 8K | 1,237 -> 1,238 (+0.1%) | 48.8 -> 51.5 (+5.6%) |
| IQ3_XXS | 128K | 1,259 -> 1,252 (-0.6%) | 48.4 -> 50.9 (+5.1%) |

UD-IQ4_XS at 4K / 32K / 64K (5-6 interleaved pairs): prompt +9.2% / +6.0% / +5.3%, output +5.0% / +3.9% / +4.6%.
IQ3_XXS at 8K was measured again with 6 interleaved pairs (an earlier 3-run set had read 5% low while other jobs shared the GPU): prompt equal, output +5.6%. With the defaults off the same build's prompt is 8% lower (1,137), so the table helps there too.

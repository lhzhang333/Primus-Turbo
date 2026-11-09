# gfx1250 attention on HipKittens kernels

Flash attention for Llama-style GQA (head dim 128, bf16, bottom-right causal) on HIP kernels for gfx1250 (MI455X):
the HIPKITTENS backend of `flash_attn_func` there. The Python side is
`primus_turbo/pytorch/kernels/attention/attention_hipkittens_gfx1250_impl.py`.

## Use

`PRIMUS_TURBO_ATTN_BACKEND=HIPKITTENS` (or `GlobalBackendManager.set_attn_backend(BackendType.HIPKITTENS, ...)`). On
gfx1250 the backend accepts every call the FlyDSL head-dim-128 kernels (`primus_turbo/flydsl/attention/gfx1250`)
take; each direction of a call these kernels cannot take runs on those FlyDSL kernels, with one warning per reason
(the LSE convention is shared, so the directions mix):

* forward: G = Hq / Hkv in {4, 8, 16}, causal, Sq <= Skv, Sq and Skv multiples of 64;
* backward: G = 4, causal, Sq <= Skv, Sq a multiple of 64 and Skv of 128, `deterministic=False` (dQ is summed with
  fp32 atomics);
* both: bf16, 16-byte aligned tensors, byte offsets within int32, and the code objects built.

q / k / v may each be contiguous `[b, s, h, d]` or a `[b, s, h, d]` view of contiguous `[s, b, h, d]` storage
(Megatron's layout). The kernels read them in place and write out / dq / dk / dv in the layouts of q / q / k / v, so
no layout copy is made.

## Kernels

| code object | kernel | work |
|---|---|---|
| `k_attn_fwd` | `k_attn_fwd` | O and LSE; 64 queries x 4 q heads of one kv head per work item, a persistent grid of one workgroup per CU (zig-zag pairs of q blocks), K / V through a 4-deep TDM ring, lazy softmax rescale |
| `k_attn_bwd` | `k_attn_bwd_prep` | the per-row terms -LSE log2(e) and rowsum(dO * O), and a zeroed fp32 dQ workspace |
| | `k_attn_bwd_main` | dK and dV of 128 kv rows per workgroup; dQ = dS K added to the workspace with fp32 atomics |
| | `k_attn_bwd_dq` | dQ = bf16(scale * workspace) |

`include/` holds the primitive layer (`hk_attn/`: WMMA, LDS loaders, TDM descriptors, buffer accesses, split
barriers) and the index formulas of each direction (`fwd_idx.h`, `bwd_idx.h`); `src/` the two kernel sources.

**Numerics.** Against FlyDSL at the Llama-3.1-8B shape: O to 49.8 dB SNR and LSE to 5e-3; on the same forward
output, dQ / dK / dV to 97.7 / 92.7 / 94.4 dB. The forward and dK / dV are bitwise reproducible; dQ is summed with fp32
atomics in no fixed order (98 dB from run to run).

## Build

`setup.py` runs `hsaco.build()` when `GPU_ARCHS` contains gfx1250 (`PRIMUS_TURBO_BUILD_HK_ATTN=0` skips it); the code
objects go to `primus_turbo/lib/hk_gfx1250` and ship in the wheel. By hand:

```
python csrc/kernels/attention/hipkittens/gfx1250/hsaco.py [--check] [--force] [--strict] [--cxx PATH] [--bin-dir DIR]
```

It builds into this checkout's `primus_turbo/lib/hk_gfx1250`, which an in-place or editable build uses; for an
installed package, pass that package's `primus_turbo/lib/hk_gfx1250` as `--bin-dir`. The compiler is `--cxx`,
`$PRIMUS_TURBO_HK_CXX`, `amdclang++` on `PATH`, or `$ROCM_PATH` / `$ROCM_HOME` / `/opt/rocm` `llvm/bin/amdclang++`.
Each entry of `manifest.json` is one device-only code object built with exactly the manifest's flags from this
directory alone; `-cuid=<name>` makes the build byte-identical in any directory.

**Fail-closed.** `manifest.json` pins the sha256 of every code object for its `toolchain` (the
`amdprimus/amdprimus:tas-20261019` image). Only a code object with the pinned bytes is installed, one without them is
removed, and another compiler builds nothing. The runtime loads a code object only when its sha256 is the pin in
`_CODE_OBJECTS` of `attention_hipkittens_gfx1250_impl.py` (kept equal to the manifest by a test); without one the
HIPKITTENS backend runs the FlyDSL kernels, and a failed build never fails the install.

## Changing a kernel

1. Edit `src/` or `include/`, or a code object's flags in `manifest.json` (for a new compiler, set `toolchain` to its
   `--version` line first).
2. `hsaco.py --force` reports the sha256 of every code object that changed. Set it as the entry's `hsaco_sha256` in
   `manifest.json` and in `_CODE_OBJECTS` of `attention_hipkittens_gfx1250_impl.py` (with the workgroup sizes and
   argument layouts if an interface changed); `hsaco.py --force --strict` must then pass.
3. On an MI455X, `hsaco.py --check` must print `ok`, then run

   ```
   PRIMUS_TURBO_TEST_LARGE=1 pytest tests/pytorch/ops/test_attention_hipkittens_gfx1250.py tests/pytorch/ops/test_attention_flydsl_gfx1250.py
   ```

## Performance

MI455X (the test node's GPU clock was held at about 1.3 GHz: compare the ratios), Llama-3.1-8B attention (b4 s8192,
32 / 8 heads, causal, bf16), one process, medians of 10 interleaved rounds (20 back-to-back calls each), two runs:

| | FlyDSL (ms) | HipKittens (ms) | speedup |
|---|---|---|---|
| forward | 1.724 | 1.363 | 1.26x |
| backward | 5.898 | 5.162 | 1.14x |
| forward + backward (autograd, Megatron's SBHD layout) | 8.272 | 6.640 | 1.25x |

With the L2 / MALL flushed before every call the ratios are the same. The backward's gain depends on how fast the
GPU's fp32 atomics are: the same backward measured 1.29x on another MI455X.

Llama-3.1-8B pretraining on one MI455X (Megatron, MBS 4, seq 8192, 20 iterations, mean of iterations 3-20, median of
3 runs): 1253.3 -> 1175.9 ms per iteration (-6.2%) and 345 -> 337 GB peak memory (no layout copies); the loss of every
iteration is within FlyDSL's run-to-run spread.

# cuda-oxide kernels: how a lane adds a GPU kernel

Why this track: our pods (driver 580.x = CUDA 13.0, vllm/vllm-openai:v0.30.0)
have **no PTX JIT library** and reject SASS from newer toolchains (silicon,
2026-09-25). Every kernel we ship is therefore **sm_120 SASS from ptxas 13.0**,
built in CI, loaded with `cuModuleLoadData`, never JIT-compiled on a pod.

```
kernels-oxide/<name>/            standalone cargo crate (own [workspace])
  -> cargo +nightly-2026-08-28 oxide build --arch sm_120   (PTX, ISA <= 9.0)
  -> ptxas 13.0.88 -arch=sm_120 -O3                          (cubin, SASS only)
  -> cuobjdump --list-elf  must show sm_120                  (CI assertion)
  -> suffix_hybrid/oxide_cubins/{<name>.cubin, manifest.json} in the bundle
  -> suffix_hybrid.oxide_kernels.ensure_loaded(name)         (pod: sha256 + driver load)
  -> src/oxide.rs launch(...)                                (cuLaunchKernel, torch stream)
```

CI (`.github/workflows/native.yml`, owned by the K2 lane — other lanes do
NOT edit it) runs `scripts/oxide_build.py`, which builds **every** directory
under `kernels-oxide/` automatically. Adding a kernel needs no CI change.

## 1. Add the kernel crate

`kernels-oxide/<name>/Cargo.toml` — copy `kernels-oxide/probe/Cargo.toml`
(same pinned cuda-oxide rev) and change `name`. The crate name is the kernel
family name everywhere (cubin file, manifest, loader).

`kernels-oxide/<name>/src/main.rs`:

```rust
use cuda_device::{DynamicSharedArray, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn my_entry(x: *const u16, out: *mut f32, n: u32, scale: f32) { ... }
}
fn main() {}
```

ABI rules (so the host launch stays a flat list of 8-byte slots):
- parameters are **raw pointers** (`*const T` / `*mut T`) or 32-bit scalars
  (`u32`, `i32`, `f32`). Avoid slices / `DisjointSlice` in shipped kernels:
  they lower to `(ptr, len)` pairs (the probe uses them on purpose, once).
- check the generated `<name>.ptx` `.entry` param list once; it is the ABI.
- tensor cores: `cuda_device::wmma::mma_m16n8k16_f32_bf16` (and `_f16`,
  tf32, s8) emit `mma.sync.aligned.m16n8k16...` — valid on sm_120 (see
  kernels-oxide/k2_nvfp4_attn). Special functions via `ptx_asm!`
  (`ex2.approx.ftz.f32`, `lg2.approx.f32`), NOT libdevice (not in CI).
- dynamic shared memory: `DynamicSharedArray::<T>::get()`; the loader opts
  every entry into 99 KB.

Local check (macOS works up to PTX; the host-embed step then fails with
"unsupported host object target", which is harmless — the `.ptx` is written):

```
rustup toolchain install nightly-2026-08-28 -c rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git \
    --rev ec4aa4797956534578a1af010f86252a0b6d8626 cargo-oxide
cd kernels-oxide/<name> && CUDA_TOOLKIT_PATH=<dir with include/cuda.h> \
    cargo +nightly-2026-08-28 oxide build --arch sm_120
grep -E '^\.(version|target)' <name>.ptx      # .version <= 9.0, .target sm_120
```

(`cargo add cuda-oxide` is an unrelated crates.io crate — never use it.)

### Variants (one crate, several cubins)

`kernels-oxide/<name>/oxide-variants.json` (optional) builds the crate once
per entry with its cargo features and extra ptxas flags; each result is its
own cubin/family `<name><suffix>`:

```json
[{"suffix": "_w1", "features": "w1", "ptxas": ["-maxrregcount=128"]},
 {"suffix": "_w3", "features": "w3", "ptxas": []}]
```

Use it for compile-time sizes (register-array lengths, launch bounds) the
host selects per launch — see k2_nvfp4_attn (MTW/MTS consts per feature).

## 2. Host side (plugin crate, feature `oxide-kernels`)

```rust
use crate::oxide::{function, launch, Arg};
let f = function("<name>", "my_entry", device_ordinal)?;   // loaded earlier
launch(f, grid, block, dyn_smem_bytes, stream_ptr,
       &[Arg::Ptr(x_ptr), Arg::Ptr(out_ptr), Arg::U32(n), Arg::F32(scale)])?;
```

- `stream_ptr` = `torch.cuda.current_stream(dev).cuda_stream` (0 = legacy
  default stream is valid). Launches are graph-capturable (no sync/alloc).
- memory stays torch-owned; pass `data_ptr()`s; validate shapes/dtypes/strides
  in the pyfunction and error loudly (see `src/nvfp4_attn_oxide.rs`).
- expose the pyfunction under `#[cfg(feature = "oxide-kernels")]` in
  `src/lib.rs`; keep work behind `guard_py` and `py.detach`.

## 3. Python side

```python
from suffix_hybrid import oxide_kernels
oxide_kernels.ensure_loaded("<name>")   # sha256 + cuModuleLoadData; raises on driver rejection
```

Call it at your lane's startup gate (outside CUDA-graph capture), before the
first launch. `oxide_kernels.probe()` (also run at pod boot with
`SUFFIX_OXIDE_PROBE=1`) loads every manifest cubin and prints
`[suffix oxide] OXIDE-PROBE PASS ...` — the per-roll proof that the toolchain
output loads and computes on the node.

## 4. What is checked where

| check | where |
|---|---|
| ptxas is release 13.0 | CI (`oxide_build.py`, fatal) |
| PTX `.target sm_120`, `.version <= 9.0`, >= 1 entry | CI |
| cubin contains sm_120 SASS (`cuobjdump --list-elf`) | CI |
| cubin sha256 == manifest | bundle build + pod loader |
| driver accepts the cubin | pod (`ensure_loaded`, boot probe) |
| probe computes correctly | pod boot (`SUFFIX_OXIDE_PROBE=1`) |

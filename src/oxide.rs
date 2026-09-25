// SPDX-License-Identifier: Apache-2.0
//! Shared host plumbing for our cuda-oxide kernels (feature `oxide-kernels`).
//!
//! Chain (docs/oxide-kernels.md): kernels-oxide/<name>/ -> cargo-oxide ->
//! PTX (ISA <= 9.0) -> ptxas 13.0 -> sm_120 cubin (SASS only) ->
//! suffix_hybrid/oxide_cubins/ -> `oxide_load_cubin` (cuModuleLoadData on
//! torch's PRIMARY context, never a second context) -> cuLaunchKernel on
//! torch's current stream. No JIT anywhere: a cubin the driver rejects is a
//! startup error with the driver's own message.

use cuda_core::{Device, Module};
use std::sync::{Arc, Mutex};

pub struct Loaded {
    pub name: String,
    pub ordinal: usize,
    _module: Arc<Module>,
    /// (entry, CUfunction as usize)
    functions: Vec<(String, usize)>,
}

static LOADED: Mutex<Vec<Loaded>> = Mutex::new(Vec::new());
static DEVICES: Mutex<Vec<(usize, Arc<Device>)>> = Mutex::new(Vec::new());

/// Largest dynamic shared memory any of our kernels may request (SM120
/// opt-in limit is 99 KB per block).
pub const MAX_DYN_SMEM: i32 = 99 * 1024;

pub fn device(ordinal: usize) -> Result<Arc<Device>, String> {
    let mut g = DEVICES.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, d)) = g.iter().find(|(o, _)| *o == ordinal) {
        return Ok(d.clone());
    }
    // Device::new retains the device's PRIMARY context — the one torch uses.
    let d = Device::new(ordinal).map_err(|e| format!("Device::new({ordinal}): {e:?}"))?;
    g.push((ordinal, d.clone()));
    Ok(d)
}

/// Load `cubin` for kernel family `name` on `ordinal` and resolve `entries`
/// (opting each into MAX_DYN_SMEM dynamic shared memory). Idempotent.
pub fn load(name: &str, cubin: &[u8], ordinal: usize, entries: &[String]) -> Result<usize, String> {
    let mut g = LOADED.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(l) = g.iter().find(|l| l.name == name && l.ordinal == ordinal) {
        return Ok(l.functions.len());
    }
    let dev = device(ordinal)?;
    // SAFETY: `cubin` is a complete image whose sha256 the Python loader
    // verified against the bundle manifest.
    let module = unsafe { dev.load_module_from_bytes(cubin) }.map_err(|e| {
        format!(
            "CUDA driver rejects oxide cubin {name} ({} bytes): {e:?} — the \
             cubin must be sm_120 SASS from ptxas 13.0 (no PTX JIT on pods)",
            cubin.len()
        )
    })?;
    let mut functions = Vec::new();
    for entry in entries {
        let f = module
            .load_function(entry)
            .map_err(|e| format!("{name}: entry {entry} not in cubin: {e:?}"))?;
        // SAFETY: valid function handle from the module just loaded.
        let cu = unsafe { f.cu_function() };
        unsafe {
            cuda_core::sys::cuFuncSetAttribute(
                cu,
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                MAX_DYN_SMEM,
            );
        }
        functions.push((entry.clone(), cu as usize));
    }
    let n = functions.len();
    g.push(Loaded {
        name: name.to_string(),
        ordinal,
        _module: module,
        functions,
    });
    Ok(n)
}

/// Resolved CUfunction for (family, entry, device), or a loud error.
pub fn function(name: &str, entry: &str, ordinal: usize) -> Result<usize, String> {
    let g = LOADED.lock().unwrap_or_else(|p| p.into_inner());
    g.iter()
        .find(|l| l.name == name && l.ordinal == ordinal)
        .and_then(|l| {
            l.functions
                .iter()
                .find(|(e, _)| e == entry)
                .map(|(_, f)| *f)
        })
        .ok_or_else(|| {
            format!(
                "oxide kernel {name}::{entry} not loaded on cuda:{ordinal} \
                 (suffix_hybrid.oxide_kernels.ensure_loaded must run first)"
            )
        })
}

/// One kernel parameter, stored so cuLaunchKernel can take its address.
pub enum Arg {
    Ptr(u64),
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Each argument widened into an 8-byte slot (u32/i32/f32 in the low 4
/// bytes): cuLaunchKernel reads `sizeof(.param)` bytes through each pointer.
pub fn pack(args: &[Arg]) -> Vec<[u8; 8]> {
    args.iter()
        .map(|a| {
            let mut b = [0u8; 8];
            match a {
                Arg::Ptr(p) => b.copy_from_slice(&p.to_ne_bytes()),
                Arg::U32(v) => b[..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::I32(v) => b[..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::F32(v) => b[..4].copy_from_slice(&v.to_ne_bytes()),
            }
            b
        })
        .collect()
}

/// cuLaunchKernel on `stream_ptr` (0 = legacy default stream, what torch
/// reports for its default stream). Graph-capturable: no sync, no alloc.
pub fn launch(
    func: usize,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    stream_ptr: usize,
    args: &[Arg],
) -> Result<(), String> {
    let mut storage = pack(args);
    let mut params: Vec<*mut std::ffi::c_void> = storage
        .iter_mut()
        .map(|b| b.as_mut_ptr() as *mut std::ffi::c_void)
        .collect();
    // SAFETY: `func` came from `load` (live module), params point at
    // correctly sized values matching the kernel's .param list, the stream
    // is torch's (or the legacy default), all device pointers are live
    // torch allocations validated by the caller.
    unsafe {
        cuda_core::launch_kernel(
            func as cuda_core::sys::CUfunction,
            grid,
            block,
            smem,
            stream_ptr as cuda_core::sys::CUstream,
            &mut params,
        )
    }
    .map_err(|e| format!("cuLaunchKernel: {e:?}"))
}

// ---- Python surface --------------------------------------------------------
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Load one bundle cubin (sha-verified by the caller) and resolve entries.
#[pyfunction]
pub fn oxide_load_cubin(
    py: Python<'_>,
    name: &str,
    cubin: &[u8],
    device_ordinal: usize,
    entries: Vec<String>,
) -> PyResult<usize> {
    let cubin = cubin.to_vec();
    let name = name.to_string();
    py.detach(move || {
        crate::guard_py("oxide_load_cubin", move || {
            load(&name, &cubin, device_ordinal, &entries).map_err(PyRuntimeError::new_err)
        })
    })
}

/// Toolchain probe: out[i] = a[i] + i (u32, n elements) with the `probe`
/// cubin. The caller checks the result and prints the boot marker.
#[pyfunction]
pub fn oxide_probe_launch(
    py: Python<'_>,
    a_ptr: u64,
    out_ptr: u64,
    n: u32,
    device_ordinal: usize,
    stream_ptr: usize,
) -> PyResult<()> {
    py.detach(move || {
        crate::guard_py("oxide_probe_launch", move || {
            let f = function("probe", "oxide_probe", device_ordinal)
                .map_err(PyRuntimeError::new_err)?;
            // Slices lower to (ptr, len) pairs in cuda-oxide's ABI.
            launch(
                f,
                (n.div_ceil(128), 1, 1),
                (128, 1, 1),
                0,
                stream_ptr,
                &[
                    Arg::Ptr(a_ptr),
                    Arg::Ptr(n as u64),
                    Arg::Ptr(out_ptr),
                    Arg::Ptr(n as u64),
                ],
            )
            .map_err(PyRuntimeError::new_err)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_args_are_packed_at_their_abi_width() {
        // u32/i32/f32 occupy the low 4 bytes, pointers 8 — the layout
        // cuLaunchKernel reads through each params[i] pointer.
        let a = [
            Arg::Ptr(0x1122_3344_5566_7788),
            Arg::U32(7),
            Arg::I32(-1),
            Arg::F32(1.5),
        ];
        let b = pack(&a);
        assert_eq!(u64::from_ne_bytes(b[0]), 0x1122_3344_5566_7788);
        assert_eq!(u32::from_ne_bytes(b[1][..4].try_into().unwrap()), 7);
        assert_eq!(i32::from_ne_bytes(b[2][..4].try_into().unwrap()), -1);
        assert_eq!(f32::from_ne_bytes(b[3][..4].try_into().unwrap()), 1.5);
    }
}

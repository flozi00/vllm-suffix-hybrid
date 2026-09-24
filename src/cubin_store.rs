// SPDX-License-Identifier: Apache-2.0
//! Process-global in-memory cutile JIT store for CI-prebuilt cubins.
//!
//! `cutile::jit_cache::enable` installs ONE store per process (no getter), so
//! every kernel lane that ships prebuilt cubins (K-GDN1, K2-NVFP4) must put
//! its entries into this shared store — a second `enable` would silently
//! drop the first lane's cubins and send its launches to `tileiras` (absent
//! on pods -> hard error).

use cutile::cutile_compiler::cuda_tile_runtime_utils::{tileiras_fingerprint, TileirasOptions};
use cutile::cutile_compiler::jit_cache::{encode_entry, EntryParams, JitStore};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

struct MemStore(Mutex<HashMap<String, Vec<u8>>>);

impl JitStore for MemStore {
    fn get(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        Ok(self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned())
    }
    fn put(&self, key: &str, value: &[u8]) -> std::io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key.to_string(), value.to_vec());
        Ok(())
    }
    fn delete(&self, key: &str) -> std::io::Result<()> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).remove(key);
        Ok(())
    }
    fn clear(&self) -> std::io::Result<()> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clear();
        Ok(())
    }
}

static STORE: OnceLock<Arc<MemStore>> = OnceLock::new();
/// Raw cubins by L2 key, for the startup driver-load self-check.
static CUBINS: Mutex<Vec<(String, Vec<u8>)>> = Mutex::new(Vec::new());

/// Serve `cubin` (compiled by CI from `bytecode` for `gpu_name`) under the
/// L2 key this process's launcher will look up.
pub fn install(key: &str, bytecode: &[u8], gpu_name: &str, cubin: &[u8]) -> Result<(), String> {
    if cubin.is_empty() {
        return Err("empty cubin".into());
    }
    let opts = TileirasOptions::default();
    let params = EntryParams {
        bc_sha256: Sha256::digest(bytecode).into(),
        gpu_name,
        opt_level: opts.opt_level,
        flags: opts.flags_byte(),
        tileiras_fp: tileiras_fingerprint(),
    };
    let entry = encode_entry(&params, cubin).ok_or("cubin entry encode failed")?;
    let store = STORE.get_or_init(|| {
        let s = Arc::new(MemStore(Mutex::new(HashMap::new())));
        cutile::jit_cache::enable(s.clone());
        s
    });
    store.put(key, &entry).map_err(|e| e.to_string())?;
    let mut c = CUBINS.lock().unwrap_or_else(|p| p.into_inner());
    if !c.iter().any(|(k, _)| k == key) {
        c.push((key.to_string(), cubin.to_vec()));
    }
    Ok(())
}

/// Is `key` served by the shared store?
pub fn contains(key: &str) -> bool {
    STORE
        .get()
        .and_then(|s| s.get(key).ok().flatten())
        .is_some()
}

/// (hits, misses) of cutile's process-wide cubin cache.
pub fn jit_snapshot() -> (u64, u64) {
    let s = cutile::jit_cache::stats();
    (s.hits, s.misses)
}

/// Turn a failed launch into OUR error. cutile falls through to `tileiras`
/// (absent on pods) in exactly two ways, told apart by its cache counters
/// (cutile-compiler cuda_tile_runtime_utils.rs compile_bytecode_cached +
/// cutile tile_kernel.rs driver-rejection retry):
///   misses grew -> the launch-time key is not installed (key drift);
///   hits grew   -> the cubin WAS served, the CUDA driver rejected it at
///                  cuModuleLoadData, cutile evicted it and tried to recompile.
pub fn explain_launch_failure(kernel: &str, before: (u64, u64), err: String) -> String {
    let (hits, misses) = jit_snapshot();
    let why = if misses > before.1 {
        "prebuilt-cubin STORE MISS: the launch-time cutile key is not among the \
         installed keys (key derivation drift) — pods never JIT"
    } else if hits > before.0 {
        "prebuilt cubin was served but the CUDA driver REJECTED it at module load \
         (toolchain/driver skew); cutile then tried to recompile with tileiras"
    } else {
        "launch failed"
    };
    format!("{kernel}: {why}. cutile said: {err}")
}

/// Load every installed cubin through the driver (cuModuleLoadData) so a
/// driver/toolchain skew fails at startup with the driver's own error, not
/// as a JIT attempt on the first launch. Returns the number loaded.
pub fn driver_load_all(ordinal: usize) -> Result<usize, String> {
    let device = cutile::cuda_core::Device::new(ordinal)
        .map_err(|e| format!("Device::new({ordinal}): {e:?}"))?;
    let c = CUBINS.lock().unwrap_or_else(|p| p.into_inner());
    for (key, cubin) in c.iter() {
        // SAFETY: `cubin` is a complete image whose sha256 was verified
        // against the bundle manifest before install.
        unsafe { device.load_module_from_bytes(cubin) }.map_err(|e| {
            format!(
                "CUDA driver rejects prebuilt cubin (key {key}, {} bytes): {e:?} — \
                 tileiras/driver skew; rebuild the cubins for this driver",
                cubin.len()
            )
        })?;
    }
    Ok(c.len())
}

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
    store.put(key, &entry).map_err(|e| e.to_string())
}

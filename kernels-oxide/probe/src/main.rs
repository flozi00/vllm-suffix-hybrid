// SPDX-License-Identifier: Apache-2.0
//! Toolchain probe: out[i] = a[i] + i  (u32). Shipped as a cubin and
//! launched once at boot by the driver self-check (marker
//! "OXIDE-PROBE PASS"): if this does not load and compute on the pod, no
//! heavy kernel is trusted.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn oxide_probe(a: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i].wrapping_add(i as u32);
        }
    }
}

fn main() {}

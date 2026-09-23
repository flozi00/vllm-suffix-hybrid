# SPDX-License-Identifier: Apache-2.0
"""Hardware-tuned Triton kernels for the vLLM IR op dispatch.

The vLLM IR layer (vllm.ir) is the official, CUDA-graph-safe plugin surface:
each op (rms_norm, fused_add_rms_norm, gelu_and_mul_sparse, ...) dispatches to
the first registered provider in its priority list whose supports_args guard
accepts the call. We register our kernels under the provider name
"suffix_kernels"; registration itself is INERT for serving — dispatch follows
priority, which only --kernel-config sets, e.g. (as a dev_args.json entry, so
enabling a kernel costs no pod roll on a dev pool):

  ["--kernel-config",
   {"ir_op_priority": {"rms_norm": ["suffix_kernels", "vllm_c", "native"],
                       "fused_add_rms_norm": ["suffix_kernels", "vllm_c", "native"]}}]

Kernels are Triton, so they JIT-compile natively for the host arch (SM120 RTX
PRO 6000, SM120/121 GB10, SM120 RTX 5090) instead of shipping pre-SM120 PTX —
that is the hardware-agnostic-for-our-fleet property: one source, per-card
SASS. Every wrapper declares a strict supports_args guard and declines what
it has not validated; the dispatcher then falls to the next provider. An
unsupported shape must degrade, never serve wrong numbers.
"""

"""Lazy-state KDA speculative verify for GLM-5.3-Flash (vLLM glm5next KDA layer, `SUFFIX_KDA_LAZY=1`).

The stock layer verifies a speculative window with `fused_recurrent_kda`, which writes a [H, V, K] fp32 state snapshot per verified token (so the sampler can
roll back): 1 MiB per token, stream and layer, 288 MiB per layer at 32 streams x 8 tokens, 34 layers, every step. This patch replaces that one call on the
spec path (and only there) by the kernel in kda_lazy_kernels.py, which keeps ONE state per request one window behind plus compact records of the last window
and replays the accepted prefix at the start of the next step, bit-identical to the stock snapshot (see that module for the protocol and the proof).
5090 microbench, 34 layers per step: 32 streams x 8 tokens 7.22 -> 2.15 ms, 8 x 8 1.70 -> 0.75 ms, 1 x 8 0.43 -> 0.26 ms.

What it touches in vLLM (anchors pinned, refuses to start on drift):
  * `fused_recurrent_kda` in vllm.models.glm5next.common.kda: spec calls (`num_accepted_tokens` set, 2-D `ssm_state_indices`) go to the lazy kernel, every other
    call (plain decode, prefill never uses it) to the original;
  * `scatter_states` in the same module: after the prefill state is written, the header of those blocks is cleared (a new request must not replay the records a
    previous owner of the block left behind);
  * `Glm5NextLinearAttention.__init__`: refuses unsupported configurations (prefill backend other than triton, mamba_cache_mode other than none).
Nothing else changes; with `SUFFIX_KDA_LAZY` unset the module is inert. Works in eager steps and inside the full CUDA graph of a decode step (no host state, the
row counters live on the device).

Check on a GPU (pod or notebook with the target vLLM): `python -m suffix_hybrid.kda_lazy oracle` (stock against lazy over many cycles, eager and graph replay)
and `... bench` (34 layers per step against stock)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

ENV = "SUFFIX_KDA_LAZY"
TARGET = "vllm.models.glm5next.common.kda"
TAG = "suffix kda_lazy"
# sha256 of the source of Glm5NextLinearAttention._forward (def line .. last line of the method) per audited vLLM release
PINNED_FORWARD = {"0.33.0": "1029f36a731bd5540e52eec8811b861a3cb1db6692780f8838094f843a2cc34a"}
ANCHOR_SPEC = "            core_attn_out_spec, _ = fused_recurrent_kda(\n"
ANCHOR_SPEC_ARGS = ("                ssm_state_indices=spec_state_indices_tensor,\n"
                    "                num_accepted_tokens=num_accepted_tokens,\n")
ANCHOR_SCATTER = ("            scatter_states(\n"
                  "                recurrent_state,\n"
                  "                last_recurrent_state,\n"
                  "                non_spec_state_indices_tensor,\n"
                  "            )\n")
ANCHOR_PLAIN = "            core_attn_out_non_spec, _ = fused_recurrent_kda(\n"

_STATE = {"layers": 0, "spec_len": 0, "engaged": 0, "orig": None, "ctx": {}}


class PatchDriftError(RuntimeError):
    pass


def _say(msg: str) -> None:
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def forward_source(text: str) -> str:
    """Source of Glm5NextLinearAttention._forward (decorators excluded) cut out of the module text."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise PatchDriftError(f"module source does not parse: {exc}") from exc
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextLinearAttention"), None)
    fn = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef) and n.name == "_forward"), None)
    if fn is None:
        raise PatchDriftError("Glm5NextLinearAttention._forward missing")
    return "".join(text.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])


def check_source(text: str) -> str:
    """Anchors and pin on the module source. Returns the matching vLLM release key; PatchDriftError otherwise."""
    body = forward_source(text)
    for name, anchor in (("spec call", ANCHOR_SPEC), ("spec call arguments", ANCHOR_SPEC_ARGS), ("scatter_states", ANCHOR_SCATTER),
                         ("plain decode call", ANCHOR_PLAIN)):
        if body.count(anchor) != 1:
            raise PatchDriftError(f"{name} anchor: expected 1, found {body.count(anchor)}")
    spec = body[body.index(ANCHOR_SPEC):]
    spec = spec[:spec.index("            )\n") + len("            )\n")]
    if ANCHOR_SPEC_ARGS not in spec:
        raise PatchDriftError("the spec call no longer passes ssm_state_indices/num_accepted_tokens in the audited form")
    digest = hashlib.sha256(body.encode()).hexdigest()
    for release, pinned in PINNED_FORWARD.items():
        if digest == pinned:
            return release
    raise PatchDriftError(f"Glm5NextLinearAttention._forward sha256 {digest} matches no audited release {sorted(PINNED_FORWARD)}: re-audit the "
                          f"spec path against kda_lazy_kernels.py and pin the new hash")


def _ctx(state, create: bool = True):
    """The per-layer lazy state (header, counters) of this layer's recurrent-state tensor; allocated on first use (warmup/prefill, before any graph capture)."""
    key = state.data_ptr()
    c = _STATE["ctx"].get(key)
    if c is None and create:
        from suffix_hybrid.kda_lazy_kernels import LazyKDA
        c = _STATE["ctx"][key] = LazyKDA(state, spec_len=_STATE["spec_len"])
    return c


def lazy_frk(**kw):
    """Replacement of the module's fused_recurrent_kda: spec-path calls go to the lazy kernel, everything else to the original."""
    orig = _STATE["orig"]
    if kw.get("num_accepted_tokens") is None or kw["ssm_state_indices"].dim() != 2:
        return orig(**kw)
    want = {"q", "k", "v", "g", "beta", "initial_state", "use_qk_l2norm_in_kernel", "cu_seqlens", "ssm_state_indices", "num_accepted_tokens", "out",
            "sigmoid_beta", "a_log", "g_bias", "compute_gate", "lower_bound"}
    if set(kw) != want or not (kw["use_qk_l2norm_in_kernel"] and kw["sigmoid_beta"] and kw["compute_gate"]) or kw["lower_bound"] is None:
        raise PatchDriftError(f"spec call of fused_recurrent_kda not in the audited form: {sorted(set(kw) ^ want)} / flags")
    state = kw["initial_state"]
    first = state.data_ptr() not in _STATE["ctx"]
    ctx = _ctx(state)
    out = ctx.run(kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], kw["cu_seqlens"], kw["ssm_state_indices"], kw["num_accepted_tokens"], kw["a_log"],
                  kw["g_bias"], kw["lower_bound"], out=kw["out"])
    if first:
        _STATE["engaged"] += 1
        _say(f"ENGAGED layer {_STATE['engaged']}/{_STATE['layers']}: lazy spec verify state {tuple(state.shape)} {state.dtype} rows<={kw['ssm_state_indices'].shape[0]} "
             f"window<={ctx.spec_len} (stock snapshots off)")
    return out, state


class _ClearingScatter:
    """The module's scatter_states with a header clear for the blocks it just wrote (prefill); every other attribute (register_warmup, ...) is the original's."""

    def __init__(self, orig):
        self._orig = orig

    def __call__(self, recurrent_state, last_recurrent_state, indices):
        self._orig(recurrent_state, last_recurrent_state, indices)
        if _STATE["spec_len"] >= 2:
            _ctx(recurrent_state).clear(indices)

    def __getattr__(self, name):
        return getattr(self._orig, name)


def apply(module) -> None:
    if _env(ENV) != "1":
        return
    release = check_source(Path(module.__file__).read_text())
    cls = module.Glm5NextLinearAttention
    orig_init = cls.__init__

    def __init__(self, config, vllm_config, prefix: str = "", *a, **kw):
        orig_init(self, config, vllm_config, prefix, *a, **kw)
        mode = getattr(vllm_config.cache_config, "mamba_cache_mode", None)
        if mode != "none":
            raise PatchDriftError(f"{prefix}: mamba_cache_mode={mode!r} (the lazy state needs the static block rows of 'none')")
        if getattr(self, "kda_prefill_backend", None) != "triton":
            raise PatchDriftError(f"{prefix}: kda_prefill_backend={getattr(self, 'kda_prefill_backend', None)!r} (prefill checkpoint blocks change the block row)")
        _STATE["layers"] += 1
        _STATE["spec_len"] = max(_STATE["spec_len"], int(self.num_spec) + 1 if self.num_spec else 0)

    cls.__init__ = __init__
    _STATE["orig"] = module.fused_recurrent_kda
    module.fused_recurrent_kda = lazy_frk
    module.scatter_states = _ClearingScatter(module.scatter_states)
    _say(f"ACTIVE: vLLM {release} glm5next KDA spec verify without per-token state snapshots (records in the column-1 block, replay at the next step)")


def install_post_import_hook() -> None:
    """sitecustomize entry; fail closed (SystemExit) on drift while enabled."""
    if _env(ENV) != "1":
        return

    def run(module):
        try:
            apply(module)
        except Exception as exc:
            raise SystemExit(f"[{TAG}] enabled but installation FAILED: {exc}") from exc

    if TARGET in sys.modules:
        run(sys.modules[TARGET])
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != TARGET:
                return None
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                run(module)

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _Finder())


if __name__ == "__main__":
    from suffix_hybrid.kda_lazy_oracle import main
    sys.exit(main(sys.argv[1:]))

"""GLM-5.3-Flash speculative verify through upstream RecoverSSM (`SUFFIX_GLM_RECOVERSSM=1`): vLLM 0.33.0 ships replay-based KDA state recovery for Kimi-K3 only; this
patch turns it on for the glm5next KDA layer, with prefix caching (mamba_cache_mode 'align') and the V2 model runner the pool runs.

Stock: the KDA layer's spec verify (`fused_recurrent_kda`) writes a [H, V, K] fp32 state snapshot per verified token, stream and layer (288 MiB per layer at 32 streams x 8 tokens),
the vLLM mamba cache keeps 1 + num_speculative_tokens state blocks per request and layer for them, and the V2 runner copies the accepted snapshot column at block boundaries.
RecoverSSM: the verify (`kda_recoverssm_verify`) reads ONE checkpoint state, writes small per-token records (outside the KV cache) and the outputs; after sampling the runner
calls one commit launch for all layers of a group (`KDARecoverSSMCommitContext.commit`: accepted tokens folded into the state, conv state compacted, block-boundary states written
for prefix caching) and resets num_accepted_tokens to 1. The per-draft-token state blocks are gone (`cache_config.use_replayssm` -> num_speculative_blocks = 0).

What this module does (every anchor pinned to vLLM 0.33.0, refuses to start on drift; unset = inert):
  0. vllm.config.vllm is compiled from a source with ONE rewrite: the `kda_architectures` tuple of VllmConfig.validate_mamba_cached_kernel gets the glm5next architectures (the validator
     re-runs on every dataclasses.replace of the config, e.g. in the model constructor, so setting the flags once is not enough); ModelConfig.supports_replayssm is true for them;
  1. EngineArgs.create_engine_config: for a glm5next target with speculative decoding it sets cache_config.use_replayssm / use_kda_recoverssm after vLLM's own validation
     and checks the V2 runner, the triton mamba backend and pp = 1;
  2. Glm5NextLinearAttention.__init__: allocates the layer's RecoverSSM records (any KDA prefill backend: flashkda's prefill checkpoint blocks sit behind the state columns,
     Kimi-K3 runs RecoverSSM with them too);
  3. the module's fused_recurrent_kda: spec calls go to kda_recoverssm_verify, everything else to the original;
  4. GDNAttentionMetadataBuilder.build (glm5next groups only): every active decode row takes the spec path, the metadata gets a neutral num_accepted_tokens, the commit
     metadata (state rows, query starts, request indices, block table + computed tokens for align mode) and the RecoverSSMMetadata class the V2 runner hooks look for.
Numerics: upstream's kernels, equal to the stock snapshots up to fp32 rounding (5090, 24 cycles: state max rel diff 3e-7, outputs within one bf16 ulp), not bit-identical.

Check on a GPU: `python -m suffix_hybrid.glm_recoverssm oracle` (upstream verify + commit against the stock spec kernel over cycles, and timing per step)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

from suffix_hybrid.kda_lazy import PatchDriftError, check_source

ENV = "SUFFIX_GLM_RECOVERSSM"
TAG = "suffix glm_recoverssm"
KDA_TARGET = "vllm.models.glm5next.common.kda"
GDN_TARGET = "vllm.v1.attention.backends.gdn_attn"
ARGS_TARGET = "vllm.engine.arg_utils"
CONFIG_TARGET = "vllm.config.vllm"
MODEL_CONFIG_TARGET = "vllm.config.model"
ARCHS = ("Glm5NextForCausalLM", "Glm5NextForConditionalGeneration")
# sha256 of GDNAttentionMetadataBuilder.build per audited vLLM release
PINNED_BUILD = {"0.33.0": "e2e6c60cbf1dedd7e2aedf82b7ea3b71e781c9b23cf0c31147c3fa2db979ecd1"}
# sha256 of VllmConfig.validate_mamba_cached_kernel per audited release (the validator whose architecture whitelist is rewritten)
PINNED_VALIDATOR = {"0.33.0": "c7eaec1f5ba39a88b164fc78236d28f9b8e313e776dfdb3daf92085eb8bedef3"}
OLD_KDA_ARCHS = '''        kda_architectures = (
            "KimiLinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        )
'''
NEW_KDA_ARCHS = '''        kda_architectures = (
            "KimiLinearForCausalLM",
            "KimiK3ForConditionalGeneration",
            "Glm5NextForCausalLM",
            "Glm5NextForConditionalGeneration",
        )
'''

_STATE = {"layers": [], "by_state": {}, "ctx": {}, "engaged": 0, "orig_frk": None, "ones": {}}


def _say(msg: str) -> None:
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _method_source(text: str, cls_name: str, name: str) -> str:
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise PatchDriftError(f"module source does not parse: {exc}") from exc
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    fn = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        raise PatchDriftError(f"{cls_name}.{name} missing")
    return "".join(text.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])


def check_builder_source(text: str) -> str:
    """Pin of GDNAttentionMetadataBuilder.build: the post-processing below reads its locals' meaning (spec rows = num_decode_draft_tokens >= 0, spec_state_indices_tensor, num_accepted_tokens)."""
    digest = hashlib.sha256(_method_source(text, "GDNAttentionMetadataBuilder", "build").encode()).hexdigest()
    for release, pinned in PINNED_BUILD.items():
        if digest == pinned:
            return release
    raise PatchDriftError(f"GDNAttentionMetadataBuilder.build sha256 {digest} matches no audited release {sorted(PINNED_BUILD)}")


# --------------------------------------------------------------------------------------------------------------------------------------------------- config
def rewrite_config_source(text: str) -> str:
    """Source of vllm.config.vllm with the glm5next architectures added to the validator's KDA whitelist (pinned: the validator's sha256 and exactly one anchor)."""
    body = _method_source(text, "VllmConfig", "validate_mamba_cached_kernel")
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest not in PINNED_VALIDATOR.values():
        raise PatchDriftError(f"VllmConfig.validate_mamba_cached_kernel sha256 {digest} matches no audited release {sorted(PINNED_VALIDATOR)}")
    if body.count(OLD_KDA_ARCHS) != 1 or text.count(OLD_KDA_ARCHS) != 1:
        raise PatchDriftError(f"kda_architectures anchor: expected 1, found {text.count(OLD_KDA_ARCHS)}")
    return text.replace(OLD_KDA_ARCHS, NEW_KDA_ARCHS)


def patch_model_config(module) -> None:
    """ModelConfig.supports_replayssm (the validator's gate for --use-replayssm) is true for glm5next; the registry's answer stays what it was for everything else."""
    cls = module.ModelConfig
    prop = cls.__dict__.get("supports_replayssm")
    if not isinstance(prop, property):
        raise PatchDriftError("ModelConfig.supports_replayssm is not a property any more")

    def supports_replayssm(self) -> bool:
        return bool(prop.fget(self)) or self.architecture in ARCHS

    cls.supports_replayssm = property(supports_replayssm)


def enable_config(vc) -> bool:
    """Called on every VllmConfig EngineArgs.create_engine_config returns. Sets the two RecoverSSM flags for a glm5next target with speculative decoding."""
    mc, cc = vc.model_config, vc.cache_config
    if mc is None or mc.architecture not in ARCHS or not getattr(vc, "num_speculative_tokens", 0):
        return False
    if not vc.use_v2_model_runner:
        raise PatchDriftError("RecoverSSM for glm5next needs the V2 model runner (the pool runs it; VLLM_USE_V2_MODEL_RUNNER=0 is not supported)")
    if vc.parallel_config.pipeline_parallel_size > 1:
        raise PatchDriftError("RecoverSSM needs pipeline_parallel_size = 1")
    backend = str(getattr(vc.mamba_config, "backend", "")).lower()
    if "triton" not in backend:
        raise PatchDriftError(f"RecoverSSM needs --mamba-backend triton (got {backend!r})")
    if cc.mamba_cache_mode not in ("none", "align"):
        raise PatchDriftError(f"unexpected mamba_cache_mode {cc.mamba_cache_mode!r}")
    if getattr(vc.mamba_config, "enable_stochastic_rounding", False):
        raise PatchDriftError("RecoverSSM does not support stochastic-rounding mamba caches")
    cc.use_replayssm = True        # MambaBase.get_kv_cache_spec: num_speculative_blocks = 0 (one state block per request and layer)
    cc.use_kda_recoverssm = True   # V2 MambaHybridModelState creates RecoverSSMState, the cudagraph sizes respect the record capacity
    _say(f"config: {mc.architecture} num_speculative_tokens={vc.num_speculative_tokens} mamba_cache_mode={cc.mamba_cache_mode}: use_replayssm + use_kda_recoverssm on")
    return True


def patch_args(module) -> None:
    cls = module.EngineArgs
    orig = cls.create_engine_config

    def create_engine_config(self, *a, **kw):
        vc = orig(self, *a, **kw)
        enable_config(vc)
        return vc

    cls.create_engine_config = create_engine_config


# --------------------------------------------------------------------------------------------------------------------------------------------------- layer
def _layer_for_state(state):
    """The KDA layer whose recurrent-state tensor OBJECT this is (id(), NOT data_ptr: the layers of different KV groups are views into the same raw tensors and share it; the call passes
    the very object the layer holds, `recurrent_state = self.kv_cache[1]`). The map is rebuilt on a miss: layers constructed later and KV caches bound later are picked up."""
    m = _STATE["by_state"]
    layer = m.get(id(state))
    if layer is None:
        m.clear()
        for cand in _STATE["layers"]:
            kv = getattr(cand, "kv_cache", None)
            if kv is not None and len(kv) > 1:
                m[id(kv[1])] = cand
        layer = m.get(id(state))
        if layer is None:
            raise PatchDriftError(f"spec call for a recurrent state {tuple(state.shape)} {state.dtype} (id {id(state)}) that belongs to none of the {len(_STATE['layers'])} registered glm5next "
                                  f"KDA layers ({len(m)} with a bound KV cache: {sorted(getattr(c, 'prefix', '?') for c in m.values())})")
    return layer


def recoverssm_frk(**kw):
    """Replacement of the module's fused_recurrent_kda: spec calls (num_accepted_tokens set, 2-D indices) go to the RecoverSSM verify, everything else to the original."""
    orig = _STATE["orig_frk"]
    if kw.get("num_accepted_tokens") is None or kw["ssm_state_indices"].dim() != 2:
        return orig(**kw)
    want = {"q", "k", "v", "g", "beta", "initial_state", "use_qk_l2norm_in_kernel", "cu_seqlens", "ssm_state_indices", "num_accepted_tokens", "out",
            "sigmoid_beta", "a_log", "g_bias", "compute_gate", "lower_bound"}
    if set(kw) != want or not (kw["use_qk_l2norm_in_kernel"] and kw["sigmoid_beta"] and kw["compute_gate"]) or kw["lower_bound"] is None:
        raise PatchDriftError(f"spec call of fused_recurrent_kda not in the audited form: {sorted(set(kw) ^ want)} / flags")
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify
    state = kw["initial_state"]
    layer = _layer_for_state(state)
    idx = kw["ssm_state_indices"]
    out = kda_recoverssm_verify(kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], kw["a_log"].reshape(-1), kw["g_bias"], float(kw["lower_bound"]), state,
                                layer.recoverssm_records, kw["cu_seqlens"], idx[:, 0], idx.shape[1], out=kw["out"])
    if not getattr(layer, "_recoverssm_logged", False):
        layer._recoverssm_logged = True
        _STATE["engaged"] += 1
        _say(f"ENGAGED layer {_STATE['engaged']}/{len(_STATE['layers'])}: RecoverSSM verify state {tuple(state.shape)} {state.dtype} window<={idx.shape[1]} (no per-token snapshots)")
    return out, state


def patch_kda(module) -> None:
    check_source(Path(module.__file__).read_text())
    cls = module.Glm5NextLinearAttention
    orig_init = cls.__init__

    def __init__(self, config, vllm_config, prefix: str = "", *a, **kw):
        orig_init(self, config, vllm_config, prefix, *a, **kw)
        cc = vllm_config.cache_config
        if not (cc.use_replayssm and cc.use_kda_recoverssm):
            raise PatchDriftError(f"{prefix}: RecoverSSM flags are not set on the cache config (EngineArgs hook missing, or no speculative decoding)")
        import torch
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords
        self.recoverssm_records = KDARecoverSSMRecords.allocate(
            vllm_config.scheduler_config.max_num_seqs, self.local_num_heads, 1 + vllm_config.num_speculative_tokens, self.head_dim,
            vllm_config.model_config.dtype, torch.device("cuda"))
        _STATE["layers"].append(self)

    cls.__init__ = __init__
    _STATE["orig_frk"] = module.fused_recurrent_kda
    module.fused_recurrent_kda = recoverssm_frk


# --------------------------------------------------------------------------------------------------------------------------------------------------- metadata
def _commit_context(builder):
    """One KDARecoverSSMCommitContext per KDA attention group (the builder's layers), created on the first step that has spec rows (the KV cache is bound then)."""
    ctx = _STATE["ctx"].get(id(builder))
    if ctx is not None:
        return ctx
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext
    fc = builder.vllm_config.compilation_config.static_forward_context
    layers = [fc[name] for name in builder.layer_names]
    adapters = [types.SimpleNamespace(kv_cache=layer.kv_cache, recoverssm_records=layer.recoverssm_records, A_log=layer.A_log.view(layer.local_num_heads),
                                      dt_bias=layer.dt_bias.view(layer.local_num_heads, layer.head_dim), gate_lower_bound=layer.kda_lower_bound,
                                      local_num_heads=layer.local_num_heads, head_dim=layer.head_dim) for layer in layers]
    ctx = KDARecoverSSMCommitContext.create(adapters, spec_query_len=1 + builder.vllm_config.num_speculative_tokens,
                                            max_num_reqs=builder.vllm_config.scheduler_config.max_num_seqs, backend="triton")
    _STATE["ctx"][id(builder)] = ctx
    return ctx


def _metadata_class():
    """GDNAttentionMetadata + RecoverSSMMetadata (the V2 runner's RecoverSSMState collects instances of the latter from the attention metadata)."""
    cls = _STATE.get("md_cls")
    if cls is not None:
        return cls
    import dataclasses
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata, RecoverSSMPostprocessMetadata

    @dataclasses.dataclass
    class GlmKDAMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
        # dataclass fields (not plain attributes): dataclasses.replace() of the metadata must keep the commit plan, a silent None would skip the commit and leave stale states
        recoverssm_commit: tuple | None = None
        recoverssm_context: object | None = dataclasses.field(default=None, repr=False, compare=False)

        def commit_recoverssm_state(self, num_accepted_tokens):
            commit = self.recoverssm_commit
            if commit is None:
                return None
            request_indices, align = commit
            _STATE["commits"] = _STATE.get("commits", 0) + 1
            if _STATE["commits"] in (1, 2, 3, 10, 100, 1000, 10000, 100000):
                _say(f"COMMIT #{_STATE['commits']}: {self.num_spec_decodes} spec rows, {'align' if align is not None else 'none'} mode, mixed={request_indices is not None}")
            # the CURRENT spec rows of this metadata (update_block_table re-gathers them after build)
            self.recoverssm_context.commit(
                num_accepted_tokens, self.spec_state_indices_tensor[: self.num_spec_decodes, 0], self.spec_query_start_loc[: self.num_spec_decodes + 1], request_indices=request_indices,
                block_table=align[0] if align is not None else None, num_computed_tokens=align[1] if align is not None else None,
                mamba_block_size=align[2] if align is not None else None)
            if align is None:
                return None
            return RecoverSSMPostprocessMetadata(num_spec_decodes=self.num_spec_decodes, request_indices=request_indices, block_table=align[0],
                                                 num_computed_tokens=align[1], block_size=align[2])

    _STATE["md_cls"] = GlmKDAMetadata
    return GlmKDAMetadata


def _is_glm_group(builder) -> bool:
    fc = builder.vllm_config.compilation_config.static_forward_context
    return bool(builder.layer_names) and all(type(fc.get(n)).__name__ == "Glm5NextLinearAttention" for n in builder.layer_names)


def patch_builder(module) -> None:
    check_builder_source(Path(module.__file__).read_text())
    cls = module.GDNAttentionMetadataBuilder
    orig_build = cls.build

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None, num_decode_draft_tokens_cpu=None, fast_build=False):
        import torch
        if not self.use_spec_decode or not _is_glm_group(self):
            return orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
        m = common_attn_metadata
        draft = num_decode_draft_tokens_cpu
        if draft is not None and m.is_prefilling is not None:
            # every active decode row takes the spec path: only that path writes the verify records the commit after sampling folds in
            ql = m.query_start_loc_cpu.diff()
            active_decode = (~m.is_prefilling) & (ql > 0) & (ql <= self.num_spec + 1)
            draft = draft.clone()
            draft[active_decode & (draft < 0)] = 0
        md = orig_build(self, common_prefix_len, m, num_accepted_tokens, draft, fast_build)
        n = md.num_spec_decodes
        if n == 0:
            return md
        window = self.num_spec + 1
        # the layer reads column 0 (conv state, verify) and the window length from the width of the row: one block per request, a stride-0 view keeps the width
        md.spec_state_indices_tensor = md.spec_state_indices_tensor[:, :1].expand(-1, window)
        ones = _STATE["ones"].get(id(self))
        if ones is None:
            ones = _STATE["ones"][id(self)] = torch.ones(self.vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=m.query_start_loc.device)
        md.num_accepted_tokens = ones[:n]                      # the commit already applied the accepted tokens
        request_indices = None
        if md.num_prefills > 0 or md.num_decodes > 0:          # mixed step: the spec rows are a subset of the batch
            from vllm.utils.torch_utils import async_tensor_h2d
            request_indices = async_tensor_h2d(md.spec_sequence_masks_cpu.nonzero(as_tuple=True)[0], dtype=torch.int32, device=m.query_start_loc.device)
        align = None
        if self.kv_cache_spec.mamba_cache_mode == "align":
            align = (m.block_table_tensor, m.compute_num_computed_tokens(), self.kv_cache_spec.block_size)
        md.__class__ = _metadata_class()
        md.recoverssm_commit = (request_indices, align)
        md.recoverssm_context = _commit_context(self)
        return md

    cls.build = build

    orig_update = cls.update_block_table

    def update_block_table(self, metadata, blk_table, slot_mapping):
        md = orig_update(self, metadata, blk_table, slot_mapping)       # dataclasses.replace: keeps the class and the commit plan, re-slices the state rows to the width of the row
        if getattr(md, "recoverssm_commit", None) is not None and md.spec_state_indices_tensor is not None:
            md.spec_state_indices_tensor = md.spec_state_indices_tensor[:, :1].expand(-1, self.num_spec + 1)
            md.recoverssm_context = _commit_context(self)         # THIS group's layers: the replace() copied group 1's context
        return md

    cls.update_block_table = update_block_table


# --------------------------------------------------------------------------------------------------------------------------------------------------- install
def _patchers():
    return {ARGS_TARGET: patch_args, KDA_TARGET: patch_kda, GDN_TARGET: patch_builder, MODEL_CONFIG_TARGET: patch_model_config}


def _source_patchers():
    """Modules compiled from a rewritten source (they define pydantic validators, so a post-import patch is too late)."""
    return {CONFIG_TARGET: rewrite_config_source}


def install_post_import_hook() -> None:
    """sitecustomize entry; fail closed (SystemExit) on drift while enabled."""
    if _env(ENV) != "1":
        return
    targets = _patchers()
    sources = _source_patchers()

    def run(fullname, module):
        try:
            targets[fullname](module)
            _say(f"ACTIVE: {fullname} patched")
        except Exception as exc:
            raise SystemExit(f"[{TAG}] enabled but installation of {fullname} FAILED: {exc}") from exc

    for name in list(targets):
        if name in sys.modules:
            run(name, sys.modules[name])
            targets.pop(name)
    for name in sources:
        if name in sys.modules:
            raise SystemExit(f"[{TAG}] {name} was imported before the hook: its validator can no longer be rewritten")

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname not in targets and fullname not in sources:
                return None
            patcher = targets.pop(fullname, None)    # first: the find_spec below asks this finder again
            rewrite = sources.pop(fullname, None)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module
            loader = spec.loader

            def exec_module(module, *a, **kw):
                try:
                    if rewrite is not None:
                        text = loader.get_source(fullname)
                        if text is None:
                            raise PatchDriftError(f"no source for {fullname}")
                        exec(compile(rewrite(text), spec.origin, "exec", dont_inherit=True), module.__dict__)
                    else:
                        real_exec(module, *a, **kw)
                    if patcher is not None:
                        patcher(module)
                    _say(f"ACTIVE: {fullname} {'rewritten' if rewrite is not None else 'patched'}")
                except Exception as exc:
                    raise SystemExit(f"[{TAG}] enabled but installation of {fullname} FAILED: {exc}") from exc
                if not targets and not sources:
                    sys.meta_path[:] = [f for f in sys.meta_path if f is not self]

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    if targets or sources:
        sys.meta_path.insert(0, _Finder())


if __name__ == "__main__":
    from suffix_hybrid.glm_recoverssm_oracle import main
    sys.exit(main(sys.argv[1:]))

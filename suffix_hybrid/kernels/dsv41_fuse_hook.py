# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 launch-fusion hooks (``SUFFIX_DSV41_FUSE=1``), SM120 only.

Same shape as dsv41_block64.py / dsv41_engram.py: sys.meta_path post-import hooks
(nvfp4_kv_patch._PostImportFinder), pinned source-hash anchors that refuse on drift, env
gates that are OFF by default, fail closed, no vLLM fork.

Master gate      SUFFIX_DSV41_FUSE=1       arms the hooks below (otherwise this module does nothing)
Family knobs     oxide (default = stock vLLM, family not patched) | auto | helion
                   auto    the Helion kernel serves a decode bucket only where a lab-tuned config is
                           shipped (suffix_hybrid/configs/helion/<kernel>/<platform>.json holds only
                           lab WINS against the stock launches); stock elsewhere
                   helion  every decode bucket (<= 192 tokens) on Helion; a bucket without a config
                           raises at its first call (vLLM's eager warm-up pass runs before capture)
  SUFFIX_HELION_DSV41_HC      mHC boundary (dsv41_hc.py)             SUFFIX_DSV41_HC_BOUNDARY=split2|resident
  SUFFIX_HELION_DSV41_ROUTER  gate GEMV (dsv41_router.py)
  SUFFIX_HELION_DSV41_SWIGLU  shared-expert SwiGLU + MXFP8 quant (dsv41_swiglu.py)
  SUFFIX_DSV41_HC_QUANT=1     the HC epilogue also emits the MXFP8 activation; the attention input
                              projection and the shared-expert gate_up consume it through
                              QuantizedActivation (skips their quant launches). Needs HC on.
Calls above 192 tokens (prefill chunks, mixed batches) always run stock vLLM; the count is part of the
exit summary. Any OTHER reason to leave a fused path (dtype, layout, hidden size, hc_mult != 4, a
linear that cannot take MXFP8 input, non-SM120 device, a quant activation that never arrives) RAISES:
there is no silent fallback. Nothing here fuses across an all-reduce.

Anchors (nightly 39983a5e7; every file hashed, a drifted file refuses to start):
  hc      vllm.models.deepseek_v41.nvidia.ops.mhc: ``mhc_shifted_post_pre`` resolves the global
          ``mhc_fused_post_pre_delayed_tilelang`` at call time, so replacing that global swaps exactly the
          SM120 fall-through of the shifted post+pre boundary (the MoEOutput / overlap-stream / mega-mHC
          branches are SM100-only; layer 0's broadcast pre and the Engram layers' pre use other entry
          points and are untouched)
  attn_q  vllm.models.deepseek_v41.attention: DeepseekV4Attention._fused_wqa_wkv_gemm
  mlp     vllm.models.deepseek_v4.nvidia.model: DeepseekV4MLP.forward (shared experts)
  gate    vllm.model_executor.layers.fused_moe.router.gate_linear: GateLinear.forward
"""
import atexit
import hashlib
import os
import sys
from pathlib import Path

GATE_ENV = "SUFFIX_DSV41_FUSE"
QUANT_ENV = "SUFFIX_DSV41_HC_QUANT"
HC_ENV = "SUFFIX_HELION_DSV41_HC"
ROUTER_ENV = "SUFFIX_HELION_DSV41_ROUTER"
SWIGLU_ENV = "SUFFIX_HELION_DSV41_SWIGLU"
TAG = "[suffix dsv41-fuse]"

HC_TARGET = "vllm.models.deepseek_v41.nvidia.ops.mhc"
ATTN_TARGET = "vllm.models.deepseek_v41.attention"
MLP_TARGET = "vllm.models.deepseek_v4.nvidia.model"
GATE_TARGET = "vllm.model_executor.layers.fused_moe.router.gate_linear"
SHA256 = {
    HC_TARGET: "c8233a70004719017ff6a5826ca538c92ea6319a9c7c0090dc8fc2e603db25e7",
    ATTN_TARGET: "1b22884bc309f89c3a9eaa217dcfc3ef85cc781ef767c14758c235631effef1c",
    MLP_TARGET: "570289501d37477ebd7b87952b7dc030a2a19b5ae9cf573a4cebb6da0ee124c3",
    GATE_TARGET: "f82a00c110dc2d15fade20d820f88e8eb22fe9bed4d649b13a67cb2ce9dc37b8",
}
HC_STOCK_NAME = "mhc_fused_post_pre_delayed_tilelang"
HC_CALL_SITE = "mhc_fused_post_pre_delayed_tilelang("      # the call site the global swap relies on
ATTN_MIN_SHARD_TOKENS = 2048    # attention._SHARD_FUSED_WQA_WKV_MIN_TOKENS (an SM90-only token-sharded path)
ATTR = "_suffix_mxfp8"          # the QuantizedActivation riding on a layer-input tensor
FIRST_QUANT_CHECK = 80          # emitted quant activations before an attention miss-only run is an error


def _fresh_state() -> dict:
    return {"hc": {"served": {}, "stock": {}}, "router": {"served": {}, "stock": {}},
            "swiglu": {"served": {}, "stock": {}}, "q": {"emitted": 0, "attn_hit": 0, "attn_miss": 0,
                                                          "ffn_hit": 0, "ffn_miss": 0},
            "modes": {}, "logged": set(), "checked_sm": False, "quant": False}


_STATE = _fresh_state()


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def _once(key: str, msg: str) -> None:
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        _log(msg)


def _count(family: str, kind: str, key) -> None:
    table = _STATE[family][kind]
    table[key] = table.get(key, 0) + 1


def summary() -> str:
    """K5: which path every call class took (printed at exit by each worker)."""
    parts = []
    for fam in ("hc", "router", "swiglu"):
        if fam in _STATE["modes"]:
            s = _STATE[fam]
            parts.append(f"{fam}[{_STATE['modes'][fam]}] helion by bucket {dict(sorted(s['served'].items()))}, "
                         f"stock by reason {dict(sorted(s['stock'].items()))}")
    if _STATE["quant"]:
        parts.append(f"quant {_STATE['q']}")
    return "; ".join(parts) or "no family armed"


def _check_sm120() -> None:
    if _STATE["checked_sm"]:
        return
    import torch
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(f"{TAG} device is sm_{major}{minor}: these kernels are SM120-only (SM100 runs "
                           "vLLM's mega-mHC / overlap path); unset SUFFIX_DSV41_FUSE")
    _STATE["checked_sm"] = True
    _log(f"device sm_{major}{minor} (family 12): ok")


def _verify(module, target: str) -> Path:
    src_path = Path(module.__file__)
    digest = hashlib.sha256(src_path.read_bytes()).hexdigest()
    if digest != SHA256[target]:
        raise RuntimeError(f"{src_path} sha256 {digest} != pinned {SHA256[target]}: re-check the hooked "
                           "call sites against this vLLM and re-pin")
    return src_path


def _vllm_qa():
    """(QuantizedActivation, kMxfp8Dynamic): imported lazily, patched out in the unit tests."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    return QuantizedActivation, kMxfp8Dynamic


def _can_take_mxfp8(linear) -> bool:
    """Stock helper: both projections use the same MXFP8 activation ABI (FlashInfer F8_128x4)."""
    from vllm.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant
    return bool(can_fuse_query_quant([linear]))


def _kernel_mod(name: str):
    import importlib
    return importlib.import_module(f"suffix_hybrid.helion_kernels.{name}")


def _decide(mod, env_mode: str, kernels: tuple, case_, what: str):
    """-> ('helion', bucket) | ('stock', reason). missing configs: auto -> stock, helion -> raise."""
    from suffix_hybrid import helion_kernels as hk
    miss = [k for k in kernels if hk.config(k, case_) is None]
    if miss:
        if env_mode == "auto":
            return "stock", "no-config"
        raise RuntimeError(f"{TAG} {what}=helion but no tuned {'/'.join(miss)} config for "
                           f"{hk.case_key(case_)} on {hk.platform()} (tune it in the lab and ship "
                           "configs/helion/<kernel>/<platform>.json)")
    return "helion", case_["numtokens"]


# ---------------------------------------------------------------------------
# hc
# ---------------------------------------------------------------------------
def make_hc_wrapper(stock, mode: str):
    """The replacement for the module global ``mhc_fused_post_pre_delayed_tilelang``."""
    decisions: dict = {}

    def wrapper(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None,
                norm_weight=None, norm_eps=1e-6, capture_aux=False):
        t, h = x.shape[0], x.shape[1]
        key = (t, h)
        d = decisions.get(key)
        if d is None:
            hc = _kernel_mod("dsv41_hc")
            if t > hc.MAX_TOKENS:
                d = ("stock", f"tokens>{hc.MAX_TOKENS}")
            else:
                d = _decide(hc, mode, hc.kernels_of(hc.boundary_knob()), hc.serving_case(t, h), HC_ENV)
            decisions[key] = d
        path, what = d
        if path == "stock":
            _count("hc", "stock", what)
            _once(f"hc-stock-{what}", f"hc: stock vLLM for {what} (first call tokens={t}, hidden={h})")
            return stock(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                         pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps,
                         capture_aux=capture_aux)
        # served by Helion: every other deviation from the contract is an error
        _check_sm120()
        import torch
        if (norm_weight is None or x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
                or residual.dim() != 3 or residual.shape[1] != 4 or h % 32 or fn.shape[0] != 24
                or fn.dtype != torch.float32 or not (x.is_contiguous() and residual.is_contiguous())):
            raise RuntimeError(
                f"{TAG} hc: unexpected call (norm_weight={'set' if norm_weight is not None else None}, "
                f"x {x.dtype} {tuple(x.shape)}, residual {tuple(residual.shape)} {residual.dtype}, "
                f"fn {tuple(fn.shape)} {fn.dtype}): the fused kernels serve bf16 [T,H] / [T,4,H] with "
                "the normalised input only")
        hc = _kernel_mod("dsv41_hc")
        want_q = _STATE["quant"]
        _count("hc", "served", what)
        _once(f"hc-helion-{what}", f"hc: Helion {hc.boundary_knob()} serves bucket {what} (first call "
                                    f"tokens={t}, hidden={h}, aux={capture_aux}, quant={want_q})")
        out = hc.hc_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                             hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                             pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps,
                             capture_aux=capture_aux, want_q=want_q)
        if not want_q:
            return out
        QuantizedActivation, key_mxfp8 = _vllm_qa()
        layer_in = out[3]
        setattr(layer_in, ATTR, QuantizedActivation(out[6], out[7], layer_in.dtype, layer_in.shape,
                                                    key_mxfp8))
        _STATE["q"]["emitted"] += 1
        return out[:6]

    wrapper.stock = stock
    return wrapper


def apply_hc(module, mode: str) -> None:
    src_path = _verify(module, HC_TARGET)
    stock = getattr(module, HC_STOCK_NAME, None)
    if not callable(stock):
        raise RuntimeError(f"{src_path} has no callable {HC_STOCK_NAME}")
    if HC_CALL_SITE not in src_path.read_text():
        raise RuntimeError(f"{src_path}: the {HC_STOCK_NAME} call site is gone")
    module.mhc_fused_post_pre_delayed_tilelang = make_hc_wrapper(stock, mode)
    _STATE["modes"]["hc"] = mode
    _log(f"ACTIVE: {HC_TARGET}.{HC_STOCK_NAME} -> Helion mHC boundary ({HC_ENV}={mode}, "
         f"quant={_STATE['quant']}; <=192 tokens, stock above)")


# ---------------------------------------------------------------------------
# attn_q: the attention input projection consumes the HC epilogue's MXFP8 activation
# ---------------------------------------------------------------------------
def apply_attn(module) -> None:
    src_path = _verify(module, ATTN_TARGET)
    cls = getattr(module, "DeepseekV4Attention", None)
    stock = cls.__dict__.get("_fused_wqa_wkv_gemm") if cls is not None else None
    if stock is None:
        raise RuntimeError(f"{src_path}: DeepseekV4Attention._fused_wqa_wkv_gemm is gone")
    eligible: dict = {}

    def _fused_wqa_wkv_gemm(self, hidden_states):
        qa = getattr(hidden_states, ATTR, None)
        q = _STATE["q"]
        if qa is None:
            q["attn_miss"] += 1
            if q["emitted"] >= FIRST_QUANT_CHECK and q["attn_hit"] == 0:
                raise RuntimeError(f"{TAG} {QUANT_ENV}=1 but {q['emitted']} MXFP8 activations were emitted "
                                   "and none reached the attention input projection: the layer-input "
                                   "tensor lost its attribute between the HC boundary and attn()")
            return stock(self, hidden_states)
        if getattr(self, "shard_fused_wqa_wkv", False) and hidden_states.shape[0] >= ATTN_MIN_SHARD_TOKENS:
            return stock(self, hidden_states)       # the SM90 token-sharded path: not ours
        ok = eligible.get(id(self))
        if ok is None:
            ok = eligible[id(self)] = _can_take_mxfp8(self.fused_wqa_wkv)
        if not ok:
            raise RuntimeError(f"{TAG} fused_wqa_wkv cannot take an MXFP8 activation "
                               f"({type(getattr(self.fused_wqa_wkv, 'quant_method', None)).__name__}): "
                               f"unset {QUANT_ENV}")
        q["attn_hit"] += 1
        qr_kv, _ = self.fused_wqa_wkv(qa)
        return qr_kv

    cls._fused_wqa_wkv_gemm = _fused_wqa_wkv_gemm
    _log(f"ACTIVE: {ATTN_TARGET}.DeepseekV4Attention._fused_wqa_wkv_gemm consumes the HC MXFP8 activation")


# ---------------------------------------------------------------------------
# mlp: shared experts - FFN-in prequant for gate_up, SwiGLU + quant for down
# ---------------------------------------------------------------------------
def apply_mlp(module, swiglu_mode: str, quant_on: bool) -> None:
    src_path = _verify(module, MLP_TARGET)
    cls = getattr(module, "DeepseekV4MLP", None)
    stock = cls.__dict__.get("forward") if cls is not None else None
    if stock is None:
        raise RuntimeError(f"{src_path}: DeepseekV4MLP.forward is gone")
    decisions: dict = {}
    eligible: dict = {}

    def _eligible(role: str, owner, linear) -> bool:
        key = (role, id(owner))
        if key not in eligible:
            eligible[key] = _can_take_mxfp8(linear)
        return eligible[key]

    def swiglu_decision(self, m: int, d: int):
        dkey = (m, d)
        dec = decisions.get(dkey)
        if dec is None:
            sw = _kernel_mod("dsv41_swiglu")
            if m > sw.MAX_TOKENS:
                dec = ("stock", f"tokens>{sw.MAX_TOKENS}")
            else:
                dec = _decide(sw, swiglu_mode, (sw.KERNEL,), sw.serving_case(m, d), SWIGLU_ENV)
            decisions[dkey] = dec
        return dec

    def forward(self, x):
        qa_in = getattr(x, ATTR, None) if quant_on else None
        if qa_in is not None:
            if not _eligible("gate_up", self, self.gate_up_proj):
                raise RuntimeError(f"{TAG} shared-expert gate_up cannot take an MXFP8 activation: unset {QUANT_ENV}")
            _STATE["q"]["ffn_hit"] += 1
        elif quant_on:
            _STATE["q"]["ffn_miss"] += 1
        gate_up, _ = self.gate_up_proj(qa_in if qa_in is not None else x)
        if swiglu_mode == "oxide":
            out, _ = self.down_proj(self.act_fn(gate_up))
            return out
        import torch
        m, d = gate_up.shape[0], gate_up.shape[-1] // 2
        act = self.act_fn
        if (type(act).__name__ != "SiluAndMulWithClamp" or getattr(act, "alpha", 1.0) != 1.0
                or getattr(act, "beta", 0.0) != 0.0 or gate_up.dtype != torch.bfloat16
                or gate_up.dim() != 2 or d % 32 or not gate_up.is_contiguous()):
            raise RuntimeError(f"{TAG} swiglu: unexpected shared-expert activation "
                               f"({type(act).__name__}, gate_up {gate_up.dtype} {tuple(gate_up.shape)}): "
                               f"the fused kernel serves SiluAndMulWithClamp on bf16 [M, 2d], d % 32 == 0")
        path, what = swiglu_decision(self, m, d)
        if path == "stock":
            _count("swiglu", "stock", what)
            _once(f"swiglu-stock-{what}", f"swiglu: stock vLLM for {what} (first call tokens={m}, d={d})")
            out, _ = self.down_proj(act(gate_up))
            return out
        _check_sm120()
        if not _eligible("down", self, self.down_proj):
            raise RuntimeError(f"{TAG} shared-expert down_proj cannot take an MXFP8 activation: "
                               f"unset {SWIGLU_ENV}")
        sw = _kernel_mod("dsv41_swiglu")
        _count("swiglu", "served", what)
        _once(f"swiglu-helion-{what}", f"swiglu: Helion {sw.KERNEL} serves bucket {what} (first call tokens={m}, d={d})")
        q, qs = sw.swiglu_quant(gate_up, act.swiglu_limit)
        QuantizedActivation, key_mxfp8 = _vllm_qa()
        out, _ = self.down_proj(QuantizedActivation(q, qs, gate_up.dtype, torch.Size((m, d)), key_mxfp8))
        return out

    cls.forward = forward
    _log(f"ACTIVE: {MLP_TARGET}.DeepseekV4MLP.forward (swiglu={swiglu_mode}, ffn-in prequant={quant_on})")


# ---------------------------------------------------------------------------
# gate: GateLinear GEMM -> Helion GEMV
# ---------------------------------------------------------------------------
def apply_gate(module, mode: str) -> None:
    src_path = _verify(module, GATE_TARGET)
    cls = getattr(module, "GateLinear", None)
    stock = cls.__dict__.get("forward") if cls is not None else None
    if stock is None:
        raise RuntimeError(f"{src_path}: GateLinear.forward is gone")
    decisions: dict = {}

    def forward(self, x):
        import torch
        w = self.weight
        if tuple(w.shape) != (384, 5120):          # not the V4.1 router (other gates stay stock)
            return stock(self, x)
        if (x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.dim() != 2
                or x.shape[1] != 5120 or not x.is_contiguous()
                or getattr(self, "out_dtype", None) != torch.float32):
            raise RuntimeError(f"{TAG} router: unexpected gate call (x {x.dtype} {tuple(x.shape)}, weight "
                               f"{w.dtype}, out_dtype {getattr(self, 'out_dtype', None)}): the GEMV serves "
                               "bf16 [M,5120] x bf16 [384,5120] -> f32")
        m = x.shape[0]
        dec = decisions.get(m)
        if dec is None:
            rt = _kernel_mod("dsv41_router")
            if m > rt.MAX_TOKENS:
                dec = ("stock", f"tokens>{rt.MAX_TOKENS}")
            else:
                dec = _decide(rt, mode, (rt.K_GEMV,), rt.serving_case(m), ROUTER_ENV)
            decisions[m] = dec
        path, what = dec
        if path == "stock":
            _count("router", "stock", what)
            _once(f"router-stock-{what}", f"router: stock GateLinear for {what} (first call tokens={m})")
            return stock(self, x)
        _check_sm120()
        rt = _kernel_mod("dsv41_router")
        _count("router", "served", what)
        _once(f"router-helion-{what}", f"router: Helion {rt.K_GEMV} serves bucket {what} (first call tokens={m})")
        return self._return(rt.gate_logits(x, w))

    cls.forward = forward
    _STATE["modes"]["router"] = mode
    _log(f"ACTIVE: {GATE_TARGET}.GateLinear.forward -> Helion GEMV for (384, 5120) gates ({ROUTER_ENV}={mode})")


# ---------------------------------------------------------------------------
# arming
# ---------------------------------------------------------------------------
def plan_from_env(environ=None) -> dict:
    """-> {'modes': {family: mode}, 'quant': bool, 'targets': [module names]}; SystemExit on a bad value
    or an inconsistent combination. Imports nothing heavy."""
    env = os.environ if environ is None else environ
    modes = {}
    for fam, name in (("hc", HC_ENV), ("router", ROUTER_ENV), ("swiglu", SWIGLU_ENV)):
        try:
            value = (env.get(name, "oxide").strip().lower() or "oxide")
            if value not in ("oxide", "helion", "auto"):
                raise RuntimeError(f"{name}={value!r}: expected oxide | helion | auto")
        except RuntimeError as exc:
            raise SystemExit(f"{TAG} {exc}") from exc
        modes[fam] = value
    quant = env.get(QUANT_ENV, "").strip() == "1"
    if quant and modes["hc"] == "oxide":
        raise SystemExit(f"{TAG} {QUANT_ENV}=1 needs {HC_ENV}=auto|helion (the HC epilogue emits the activation)")
    targets = []
    if modes["hc"] != "oxide":
        targets.append(HC_TARGET)
    if quant:
        targets.append(ATTN_TARGET)
    if quant or modes["swiglu"] != "oxide":
        targets.append(MLP_TARGET)
    if modes["router"] != "oxide":
        targets.append(GATE_TARGET)
    return {"modes": modes, "quant": quant, "targets": targets}


def _callback_for(target: str, plan: dict):
    def callback(module) -> None:
        try:
            if target == HC_TARGET:
                apply_hc(module, plan["modes"]["hc"])
            elif target == ATTN_TARGET:
                apply_attn(module)
            elif target == MLP_TARGET:
                apply_mlp(module, plan["modes"]["swiglu"], plan["quant"])
            elif target == GATE_TARGET:
                apply_gate(module, plan["modes"]["router"])
        except Exception as exc:
            raise SystemExit(f"{TAG} enabled but FAILED on {module.__name__}: {exc}") from exc
    return callback


def install_post_import_hook() -> bool:
    """sitecustomize entry. False (importing nothing heavy) unless the master gate is on AND at least
    one family knob is not 'oxide'."""
    if os.environ.get(GATE_ENV, "").strip() != "1":
        return False
    plan = plan_from_env()
    if not plan["targets"]:
        _log(f"{GATE_ENV}=1 but every family knob is 'oxide' ({HC_ENV}, {ROUTER_ENV}, {SWIGLU_ENV}): inert")
        return False
    early = [t for t in plan["targets"] if t in sys.modules]
    if early:
        raise SystemExit(f"{TAG} {early} imported before the hook armed")
    _STATE["quant"] = plan["quant"]
    for fam, mode in plan["modes"].items():
        if mode != "oxide":
            _STATE["modes"][fam] = mode
    from nvfp4_kv_patch import _PostImportFinder
    for target in plan["targets"]:
        sys.meta_path.insert(0, _PostImportFinder(target, _callback_for(target, plan)))
    atexit.register(lambda: _log(summary()))
    return True

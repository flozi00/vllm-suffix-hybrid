# SPDX-License-Identifier: Apache-2.0
"""CPU stand-in for the Triton subset our kernels use (CI has no triton): `install()` puts fake
`triton`, `triton.language`, `...extra.libdevice` and `...extra.cuda` modules in sys.modules, so a
kernel module's @triton.jit bodies run as plain Python on torch CPU tensors, one program at a
time (grid order). Pointers are (flat tensor, int64 offsets); loads / stores bounds-check every
unmasked lane, masked loads without `other` return poison (NaN / 0xA5) so a kernel that uses
them fails its numerics. Strict where Triton is: power-of-two aranges / tensor shapes, tl.dot
operands 2-D, >= 16 per dim, matching fp16 / bf16 dtypes. Layouts, warps, pipelining, PDL and
cache hints do not exist here (no-ops); inline asm only as an L2 prefetch."""
from __future__ import annotations

import contextlib
import sys
import types

import torch

_PID = [(0, 0, 0)]
_GRID = [(1, 1, 1)]
_LAUNCH_OPTS = {"num_warps", "num_stages", "num_ctas", "launch_pdl", "maxnreg", "enable_fp_fusion",
                "launch_cooperative_grid"}


def _pow2(n) -> bool:
    n = int(n)
    return n > 0 and n & (n - 1) == 0


class Ptr:
    __slots__ = ("buf", "off")

    def __init__(self, buf, off=0):
        self.buf, self.off = buf, off

    def __add__(self, o):
        if isinstance(o, torch.Tensor):
            o = o.to(torch.int64)
        return Ptr(self.buf, self.off + o)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-o)


def _off(ptr):
    return torch.as_tensor(ptr.off, dtype=torch.int64)


def _bounds(off, n):
    if off.numel() and (int(off.min()) < 0 or int(off.max()) >= n):
        raise IndexError(f"shim: access outside the tensor (offsets {int(off.min())}..{int(off.max())}, "
                         f"numel {n})")


def _poison(dtype):
    return float("nan") if dtype.is_floating_point else 0xA5


def load(ptr, mask=None, other=None, **_):
    off, n = _off(ptr), ptr.buf.numel()
    if mask is None:
        _bounds(off, n)
        return ptr.buf[off]
    m = torch.as_tensor(mask, dtype=torch.bool)
    off, m = torch.broadcast_tensors(off, m)
    _bounds(off[m], n)
    v = ptr.buf[torch.where(m, off, 0)]
    fill = _poison(v.dtype) if other is None else other
    return torch.where(m, v, torch.as_tensor(fill).to(v.dtype))


def store(ptr, val, mask=None, **_):
    off, n = _off(ptr), ptr.buf.numel()
    val = torch.as_tensor(val)
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        off, val, m = torch.broadcast_tensors(off, val, m)
        off, val = off[m], val[m]
    else:
        off, val = torch.broadcast_tensors(off, val)
    _bounds(off, n)
    off, val = off.reshape(-1), val.reshape(-1)
    if off.numel() != torch.unique(off).numel():
        raise AssertionError("shim: one store writes the same address twice")
    ptr.buf[off] = val.to(ptr.buf.dtype)


def program_id(axis):
    return torch.tensor(_PID[0][axis], dtype=torch.int32)


def num_programs(axis):
    return torch.tensor(_GRID[0][axis], dtype=torch.int32)


def arange(a, b):
    assert _pow2(b - a), f"tl.arange({a}, {b}): size must be a power of two"
    return torch.arange(int(a), int(b), dtype=torch.int32)


def _shape(shape):
    shape = [int(s) for s in (shape if isinstance(shape, (tuple, list)) else [shape])]
    assert all(_pow2(s) for s in shape), f"tensor shape {shape}: powers of two only"
    return shape


def zeros(shape, dtype):
    return torch.zeros(_shape(shape), dtype=dtype)


def full(shape, value, dtype):
    return torch.full(_shape(shape), float(value) if dtype.is_floating_point else int(value),
                      dtype=dtype)


def zeros_like(x):
    return torch.zeros_like(x)


def where(c, a, b):
    c = torch.as_tensor(c, dtype=torch.bool)
    if not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32 if isinstance(a, float) or isinstance(b, float)
                         else torch.int32)
    return torch.where(c, a, b)


def _red(fn, x, axis, keep_int=True):
    x = torch.as_tensor(x)
    r = fn(x) if axis is None else fn(x, int(axis))
    return r.to(x.dtype) if keep_int and not x.dtype.is_floating_point and x.dtype != torch.bool else r


def sum(x, axis=None, **_):  # noqa: A001
    x = torch.as_tensor(x)
    assert x.dtype != torch.bool, "tl.sum of int1: cast first"
    return _red(lambda t, *a: torch.sum(t, *a), x, axis)


def max(x, axis=None, **_):  # noqa: A001
    return _red(lambda t, *a: torch.amax(t, *a), x, axis)


def min(x, axis=None, **_):  # noqa: A001
    return _red(lambda t, *a: torch.amin(t, *a), x, axis)


def cumsum(x, axis=0, **_):
    x = torch.as_tensor(x)
    return torch.cumsum(x, int(axis)).to(x.dtype)


def histogram(x, num_bins, mask=None):
    x = torch.as_tensor(x).reshape(-1)
    if mask is not None:
        x = x[torch.as_tensor(mask, dtype=torch.bool).reshape(-1)]
    assert x.numel() == 0 or (int(x.min()) >= 0 and int(x.max()) < num_bins), "histogram range"
    return torch.bincount(x.long(), minlength=int(num_bins))[:int(num_bins)].to(torch.int32)


def minimum(a, b, **_):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return torch.minimum(a, b.to(a.dtype) if b.dim() == 0 else b)


def maximum(a, b, **_):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return torch.maximum(a, b.to(a.dtype) if b.dim() == 0 else b)


def abs(x):  # noqa: A001
    return torch.abs(x)


def exp2(x):
    return torch.exp2(x)


def reshape(x, *shape, can_reorder=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return torch.as_tensor(x).reshape(_shape(shape))


def split(x):
    assert x.shape[-1] == 2, "tl.split: last dim must be 2"
    return x[..., 0], x[..., 1]


def join(a, b):
    return torch.stack(torch.broadcast_tensors(a, b), -1)


def permute(x, *dims):
    dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
    return x.permute(*dims)


def trans(x, *dims):
    return permute(x, *dims) if dims else x.t()


def expand_dims(x, axis):
    return torch.as_tensor(x).unsqueeze(int(axis))


def broadcast_to(x, *shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return x.expand(*shape)


def dot(a, b, acc=None, out_dtype=torch.float32, **_):
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], (a.shape, b.shape)
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16), (a.dtype, b.dtype)
    assert all(int(d) >= 16 for d in (*a.shape, b.shape[1])), f"tl.dot dims >= 16: {a.shape} {b.shape}"
    r = a.float() @ b.float()
    return r if acc is None else acc + r


def cast(x, dtype, bitcast=False, **_):
    x = torch.as_tensor(x)
    if bitcast:
        assert x.element_size() == torch.empty((), dtype=dtype).element_size(), "bitcast width"
        return x.view(dtype)
    if dtype == torch.float8_e4m3fn:
        assert bool(torch.isfinite(x).all()) and float(x.abs().max(), ) <= 448.0, "e4m3 range"
    return x.to(dtype)


def static_range(*a):
    return range(*[int(v) for v in a])


def range_(*a, **_):
    return range(*[int(v) for v in a])


def debug_barrier():
    return None


def multiple_of(x, _):
    return x


def max_contiguous(x, _):
    return x


def inline_asm_elementwise(asm, constraints, args, dtype, is_pure=True, pack=1):
    assert asm.startswith("prefetch.global.L2"), f"shim: unsupported inline asm {asm!r}"
    for a in args:
        if isinstance(a, Ptr):
            _bounds(_off(a), a.buf.numel())
    return torch.zeros(_off(args[0]).shape, dtype=dtype)


def div_rn(a, b):
    return torch.as_tensor(a, dtype=torch.float32) / torch.as_tensor(b, dtype=torch.float32)


def rint(x):
    return torch.round(x)


class constexpr:  # noqa: N801 - annotations only; constexpr(v) returns v
    def __new__(cls, v=None):
        return v


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__(self, *a, **k):  # @triton.jit helper called from a kernel: inline
        return self.fn(*a, **k)

    def __getitem__(self, grid):
        def launch(*args, **kw):
            kw = {k: v for k, v in kw.items() if k not in _LAUNCH_OPTS}
            g = grid(kw) if callable(grid) else grid
            g = tuple(int(v) for v in (g if isinstance(g, (tuple, list)) else (g,))) + (1, 1)
            g = g[:3]
            args = [Ptr(a.view(-1)) if isinstance(a, torch.Tensor) else a for a in args]
            for a in args:
                if isinstance(a, Ptr):
                    assert a.buf.is_contiguous()
            _GRID[0] = g
            for z in range(g[2]):
                for y in range(g[1]):
                    for x in range(g[0]):
                        _PID[0] = (x, y, z)
                        self.fn(*args, **kw)
        return launch


def jit(fn=None, **_):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


def _modules():
    tri = types.ModuleType("triton")
    tl = types.ModuleType("triton.language")
    extra = types.ModuleType("triton.language.extra")
    libdevice = types.ModuleType("triton.language.extra.libdevice")
    cuda = types.ModuleType("triton.language.extra.cuda")
    math = types.ModuleType("triton.language.math")
    tri.jit = jit
    tri.cdiv = lambda a, b: -(-a // b)
    tri.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    tri.language = tl
    for name in ("load", "store", "program_id", "num_programs", "arange", "zeros", "full",
                 "zeros_like", "where", "sum", "max", "min", "cumsum", "histogram", "minimum",
                 "maximum", "abs", "exp2", "reshape", "split", "join", "permute", "trans",
                 "expand_dims", "broadcast_to", "dot", "cast", "static_range", "debug_barrier",
                 "multiple_of", "max_contiguous", "inline_asm_elementwise", "constexpr"):
        setattr(tl, name, globals()[name])
    tl.range = range_
    tl.float32, tl.float16, tl.bfloat16 = torch.float32, torch.float16, torch.bfloat16
    tl.int64, tl.int32, tl.int16, tl.int8, tl.uint8 = (torch.int64, torch.int32, torch.int16,
                                                       torch.int8, torch.uint8)
    tl.int1, tl.float8e4nv = torch.bool, torch.float8_e4m3fn
    math.div_rn = div_rn
    tl.math = math
    libdevice.rint = rint
    extra.libdevice = libdevice
    extra.cuda = cuda
    cuda.gdc_wait = lambda: None
    cuda.gdc_launch_dependents = lambda: None
    cuda.libdevice = libdevice
    tl.extra = extra
    return {"triton": tri, "triton.language": tl, "triton.language.extra": extra,
            "triton.language.extra.libdevice": libdevice, "triton.language.extra.cuda": cuda,
            "triton.language.math": math}


@contextlib.contextmanager
def installed():
    """Fake triton in sys.modules for the duration (restores what was there)."""
    mods = _modules()
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield mods["triton.language"]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

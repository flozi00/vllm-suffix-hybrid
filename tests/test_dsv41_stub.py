# SPDX-License-Identifier: Apache-2.0
"""CPU stand-in for ``helion.language`` used by the dsv41 fusion tests.

Helion is not installable on a Mac (it imports Triton), so the Helion bodies of
suffix_hybrid/helion_kernels/dsv41_*.py run here through this reference-mode
emulation of the subset they use. Real Helion (HELION_INTERPRET=1) is used
instead whenever it is importable (CI image); see ``backends()``.

Semantics mirrored from Helion's ref mode:
  * ``hl.tile`` yields tile objects (.begin/.end/.index/.id/.block_size);
  * a subscript made of tiles / 1-d index tensors / slices is an OUTER
    PRODUCT (every non-int entry is its own axis, in order), ints are scalars;
  * programs execute sequentially, so ``hl.atomic_add`` returns the old value
    and the last program of a group sees ``old == n - 1``.
Tensors the bodies index with tiles must be ``_T`` (a Tensor subclass): the
tests wrap every kernel argument with ``wrap`` and the bodies allocate with
``x.new_empty`` / ``torch.empty_like`` (which keep the subclass).
"""
import itertools
import types

import pytest
import torch


class Tile:
    def __init__(self, begin, end, size):
        self.begin, self.end, self.block_size = torch.tensor(begin), torch.tensor(end), size
        self.index = torch.arange(begin, min(begin + size, end))
        self.id = torch.tensor(begin // size)

    def __len__(self):
        return len(self.index)


def _axes(idx, shape):
    """Subscript -> torch advanced index with Helion outer-product semantics."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = tuple(Ellipsis if i is Ellipsis else i for i in idx)
    if Ellipsis in idx:
        k = idx.index(Ellipsis)
        fill = len(shape) - (len(idx) - 1 - sum(i is None for i in idx))
        idx = idx[:k] + (slice(None),) * fill + idx[k + 1:]
    scal = lambda i: isinstance(i, int) or (isinstance(i, torch.Tensor) and i.dim() == 0)
    if len(idx) == 1 and isinstance(idx[0], torch.Tensor) and idx[0].dim() >= 2:
        return (idx[0].as_subclass(torch.Tensor).long(),)      # one n-d index tensor: torch semantics
    if all(scal(i) or isinstance(i, slice) or i is None for i in idx):
        return tuple(int(i) if scal(i) else i for i in idx)
    out, pos, n_arr = [], 0, 0
    arrs = []
    for i in idx:
        if i is None:
            arrs.append(("none", None))
        elif scal(i):
            arrs.append(("int", int(i)))
            pos += 1
        elif isinstance(i, slice):
            arrs.append(("arr", torch.arange(shape[pos])[i]))
            pos += 1
            n_arr += 1
        else:
            t = i.index if isinstance(i, Tile) else i
            arrs.append(("arr", t.as_subclass(torch.Tensor).long()))
            pos += 1
            n_arr += 1
    j = 0
    for kind, v in arrs:
        if kind == "int":
            out.append(v)
        elif kind == "arr":
            sh = [1] * n_arr
            sh[j] = -1
            out.append(v.reshape(sh))
            j += 1
        else:
            raise AssertionError("None in a tile subscript is not supported by the stub")
    return tuple(out)


class _T(torch.Tensor):
    def __getitem__(self, idx):
        return super().__getitem__(_axes(idx, self.shape))

    def __setitem__(self, idx, val):
        if isinstance(val, torch.Tensor):
            val = val.as_subclass(torch.Tensor)
        super().__setitem__(_axes(idx, self.shape), val)


def wrap(x):
    return x.as_subclass(_T) if isinstance(x, torch.Tensor) else x


def unwrap(x):
    if isinstance(x, tuple):
        return tuple(unwrap(i) for i in x)
    return x.as_subclass(torch.Tensor) if isinstance(x, torch.Tensor) else x


def make_hl(blocks=None):
    """blocks: {registration order index -> value}; default = the lower bound
    clamped up to ``default``. ``register_block_size(lo, hi)`` returns
    blocks[i] clamped into [lo, hi] (i = nth call in the body)."""
    state = {"n": 0}
    blocks = blocks or {}

    def register_block_size(lo, hi=None):
        i, state["n"] = state["n"], state["n"] + 1
        lo, hi = (lo, hi) if hi is not None else (1, lo)
        return int(max(lo, min(hi, blocks.get(i, hi))))

    def tile(begin, end=None, block_size=None):
        if end is None:
            begin, end = ([0] * len(begin) if isinstance(begin, list) else 0), begin
        multi = isinstance(end, list)
        ends = end if multi else [end]
        begins = begin if isinstance(begin, list) else [begin] * len(ends)
        sizes = block_size if isinstance(block_size, list) else [block_size] * len(ends)
        dims = []
        for b, e, s in zip(begins, ends, sizes):
            b, e = int(b), int(e)
            s = int(s) if s is not None else max(1, e - b)
            dims.append([Tile(i, e, s) for i in range(b, e, s)])
        for combo in itertools.product(*dims):
            yield combo if multi else combo[0]

    def grid(ends):
        return itertools.product(*(range(int(e)) for e in ends)) if isinstance(ends, list) \
            else range(int(ends))

    def atomic_add(target, index, value, sem="relaxed"):
        ix = tuple(int(i) for i in index)
        old = target.as_subclass(torch.Tensor)[ix].clone()
        target.as_subclass(torch.Tensor)[ix] = old + value
        return int(old)

    def dims(shape):
        return [len(s) if isinstance(s, Tile) else int(s) for s in shape]

    return types.SimpleNamespace(
        tile=tile, grid=grid, atomic_add=atomic_add, static_range=range,
        join=lambda a, b: torch.stack([a, b], -1),
        arange=lambda *a, **k: wrap(torch.arange(*[int(x) for x in a], **k)),
        pdl_wait=lambda: None, pdl_launch_dependents=lambda: None,
        specialize=lambda v: v if not isinstance(v, (list, tuple)) else None,
        register_block_size=register_block_size, constexpr=int,
        zeros=lambda shape, dtype=torch.float32: wrap(torch.zeros(dims(shape), dtype=dtype)),
        full=lambda shape, v, dtype=torch.float32: wrap(torch.full(dims(shape), v, dtype=dtype)),
        dot=lambda a, b, acc=None, out_dtype=None: (a.float() @ b.float()) + (0 if acc is None else acc))


def backends():
    """('stub',) always; ('stub', 'helion') when Helion imports (CI)."""
    try:
        import helion  # noqa: F401
        return ("stub", "helion")
    except Exception:  # noqa: BLE001
        return ("stub",)


class _TorchProxy:
    """``torch`` as the bodies see it: allocations come back as ``_T`` so tile subscripts work."""

    _ALLOC = ("empty", "zeros", "full", "empty_like", "zeros_like", "full_like")

    def __getattr__(self, name):
        attr = getattr(torch, name)
        if name in self._ALLOC:
            return lambda *a, **k: wrap(attr(*a, **k))
        return attr


def run_body(monkeypatch, module, body, args, blocks=None, backend="stub"):
    """Execute a Helion body on CPU tensors; returns plain tensors."""
    if backend == "helion":
        import helion
        kern = helion.kernel(body, static_shapes=False, config=helion.Config(**(blocks or {})))
        return unwrap(kern(*args))
    stub = make_hl(blocks)
    holders = [module] + [m for m in _helper_modules() if m is not module]
    saved = [(m, getattr(m, "hl", None), m.torch) for m in holders]
    for m in holders:
        m.hl, m.torch = stub, _TorchProxy()
    try:
        out = body(*[wrap(a) for a in args])
    finally:
        for m, hl_, torch_ in saved:
            m.hl, m.torch = hl_, torch_
    return unwrap(out)


def _helper_modules():
    """Modules whose helper functions are traced into the bodies (they carry their own hl)."""
    from suffix_hybrid.helion_kernels import dsv41_common
    return [dsv41_common]


@pytest.mark.parametrize("n", [1])
def test_stub_outer_product_semantics(n):
    x = wrap(torch.arange(2 * 3 * 4).reshape(2, 3, 4))
    t0 = Tile(0, 2, 2)
    t2 = Tile(0, 4, 2)
    got = x[t0, :, t2]
    assert got.shape == (2, 3, 2)
    assert torch.equal(got.as_subclass(torch.Tensor), x.as_subclass(torch.Tensor)[:, :, :2])
    y = wrap(torch.zeros(2, 3, 4))
    y[t0, 1, t2] = torch.ones(2, 2)
    assert y.sum() == 4 and y[0, 1, 0] == 1


def lower_to_triton(body, args, config=None):
    """Helion's compiler front end on CPU tensors (the K2 'does it lower' proof): returns the
    Triton source, or skips when Helion is not importable (CI image has it)."""
    helion = pytest.importorskip("helion")
    from helion.runtime.ref_mode import RefMode
    bound = helion.kernel(body, static_shapes=False, ref_mode=RefMode.OFF).bind(tuple(args))
    cfg = helion.Config(**config) if config else bound.config_spec.default_config()
    return bound.to_triton_code(cfg)

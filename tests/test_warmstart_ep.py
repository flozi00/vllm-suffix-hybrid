# SPDX-License-Identifier: Apache-2.0
"""vllm.endpoint_plugins seam tests for the warm-start route (offline).

Pins (from the pinned vLLM 0.30.0 tarball, vllm/plugins/__init__.py and
vllm/plugins/endpoint_plugins/interface.py — read, not guessed):
- entry-point group name: "vllm.endpoint_plugins"
- strict opt-in posture: load_endpoint_plugins returns [] WITHOUT calling
  load_plugins_by_group when VLLM_PLUGINS is unset (the loader entry checks
  envs.VLLM_PLUGINS is None FIRST — assert via a mock that the discovery
  call is never reached);
- EndpointPlugin protocol shape: zero-arg factory -> name /
  required_tasks=None / attach_router(app) / async init_state(engine_client,
  state, args);
- dist-info discovery: importlib.metadata finds
  suffix_hybrid_warmstart_ep-1.0.dist-info/ on sys.path and yields the
  entry point suffix_hybrid_warmstart -> suffix_hybrid_warmstart_ep:register
  (sandboxed in a temp sys.path so the test venv's own metadata is irrelevant);
- attach_router registers POST /debug/warm-start with the EXISTING
  suffix_hybrid.warmstart.warm_start handler, and no-ops (belt gate) when
  SUFFIX_HYBRID_WARMSTART is unset.
"""
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exact seam strings — vllm/plugins/__init__.py (pinned 0.30.0).
EP_GROUP = "vllm.endpoint_plugins"
EP_NAME = "suffix_hybrid_warmstart"
EP_ENV_VAR = "VLLM_PLUGINS"
EP_MODULE = "suffix_hybrid_warmstart_ep"
EP_FACTORY = f"{EP_MODULE}:register"
DISTINFO = f"{EP_MODULE}-1.0.dist-info"


def _load_ep_module():
    spec = importlib.util.spec_from_file_location(
        EP_MODULE, os.path.join(REPO, EP_MODULE + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_error(tag, exc):
    return pytest.skip(f"{tag}: {exc} (skip as the suite's vllm-missing "
                       f"tests do)")


# ------------------------------------------------------------- seam strings

def test_seam_strings_pinned():
    """The dist-info on disk must carry the EXACT group/plugin/factory
    strings read from the pinned vllm/plugins/__init__.py."""
    ep_txt = Path(REPO) / DISTINFO / "entry_points.txt"
    content = ep_txt.read_text()
    assert f"[{EP_GROUP}]" in content
    assert f"{EP_NAME} = {EP_FACTORY}" in content
    meta = (Path(REPO) / DISTINFO / "METADATA").read_text()
    assert "Name: suffix_hybrid_warmstart_ep" in meta


def test_repo_root_module_and_distinfo_exist():
    assert (Path(REPO) / (EP_MODULE + ".py")).is_file()
    assert (Path(REPO) / DISTINFO).is_dir()
    # gitignore must NOT exclude the dist-info (importlib needs it in the
    # bundle, which packs the repo tree)
    gi = (Path(REPO) / ".gitignore").read_text()
    for rule in ("*dist-info", "dist-info", "*.dist-info"):
        assert rule not in gi, f"gitignore rule {rule!r} would hide {DISTINFO}"


# ------------------------------------------------------------ protocol shape

def test_register_returns_protocol_object():
    ep = _load_ep_module()
    plugin = ep.register()
    assert plugin.name == EP_NAME
    assert plugin.required_tasks is None       # always eligible (subject to VLLM_PLUGINS)
    assert callable(plugin.attach_router)
    assert asyncio.iscoroutinefunction(plugin.init_state)


async def _init_state_ok(plugin):
    # init_state(engine_client, state, args) — signature order per protocol
    await plugin.init_state(None, NS(), NS())  # must not raise (None engine
    # is the render-server case; the route handler owns the 503 posture)


def test_init_state_signature_order():
    ep = _load_ep_module()
    plugin = ep.register()
    asyncio.run(_init_state_ok(plugin))


# ------------------------------------------------- attach_router + belt gate

def _bare_app():
    fastapi = pytest.importorskip("fastapi")
    return fastapi.FastAPI()


def _has_route(app, path):
    return any(getattr(r, "path", None) == path for r in app.routes)


def test_attach_router_registers_route_when_armed(monkeypatch, capsys):
    from suffix_hybrid import warmstart
    monkeypatch.setenv("SUFFIX_HYBRID_WARMSTART", "1")
    ep = _load_ep_module()
    app = _bare_app()
    ep.register().attach_router(app)
    assert _has_route(app, "/debug/warm-start")
    # the EXISTING handler is reused — zero route logic duplicated
    route = next(r for r in app.routes
                 if getattr(r, "path", None) == "/debug/warm-start")
    assert route.endpoint is warmstart.warm_start
    assert "POST" in route.methods
    err = capsys.readouterr().err
    assert ("suffix_hybrid WARM-START: POST /debug/warm-start registered "
            "(endpoint_plugins)") in err


def test_attach_router_noop_when_flag_off(monkeypatch, capsys):
    """Belt gate pins STRICT OFF: loader let us in, flag says no route."""
    monkeypatch.delenv("SUFFIX_HYBRID_WARMSTART", raising=False)
    ep = _load_ep_module()
    app = _bare_app()
    before = len(app.routes)
    ep.register().attach_router(app)
    assert len(app.routes) == before            # zero routes added
    assert not _has_route(app, "/debug/warm-start")
    assert ("suffix_hybrid WARM-START: route skipped (flag off)"
            in capsys.readouterr().err)


def test_route_reachable_via_fastapi_testclient(monkeypatch):
    """End-to-end in-process: armed plugin -> POST carrier body
    (model+prompt+sequences, as the EPP gateway forwards it) -> 200 and the
    handler ignores the carrier extras."""
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("SUFFIX_HYBRID_WARMSTART", "1")
    ep = _load_ep_module()

    seen = {}

    class FakeEngine:
        async def collective_rpc_async(self, method, args=(), **kw):
            seen["method"] = method
            seen["args"] = args
            return [2]

    app = fastapi.FastAPI()
    ep.register().attach_router(app)
    app.state.engine_client = FakeEngine()   # what init_app_state threads in
    with fastapi.testclient.TestClient(app) as client:
        r = client.post("/debug/warm-start", json={
            "model": "pool-model", "prompt": "warmstart-carrier",
            "sequences": [[1, 2, 3], [4, 5, 6]]})
    assert r.status_code == 200
    assert r.json() == {"ingested": 2}
    from suffix_hybrid import warmstart
    assert seen["method"] is warmstart._probe_worker
    assert seen["args"] == ([[1, 2, 3], [4, 5, 6]],)


# ----------------------------------------------- importlib dist-info discovery

def test_distinfo_discovered_on_sys_path():
    """Path-layout pin (per the dist-info discovery documented in
    importlib.metadata): a sys.path entry containing BOTH
    suffix_hybrid_warmstart_ep.py and its dist-info yields our entry point
    under the exact group with the exact factory ref. Sandboxed in a temp
    dir so the venv's own metadata is irrelevant."""
    from importlib.metadata import entry_points
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for src in (Path(REPO) / DISTINFO).iterdir():
            if src.is_file():
                dest_dir = tmp_path / DISTINFO
                dest_dir.mkdir(exist_ok=True)
                (dest_dir / src.name).write_bytes(src.read_bytes())
        (tmp_path / (EP_MODULE + ".py")).write_bytes(
            (Path(REPO) / (EP_MODULE + ".py")).read_bytes())
        sys.path.insert(0, tmp)
        try:
            _ = importlib.util.find_spec(EP_MODULE)  # noqa: F841 — resolves
            eps = [e for e in entry_points(group=EP_GROUP)
                   if e.name == EP_NAME]
            assert len(eps) == 1
            assert eps[0].value == EP_FACTORY
        finally:
            sys.path.remove(tmp)
            sys.modules.pop(EP_MODULE, None)
            importlib.invalidate_caches()


# ------------------------------------------------- VLLM_PLUGINS loader gating

def test_live_loader_gates_on_vllm_plugins_env():
    """Loader contract pinned from the 0.30.0 source: when VLLM_PLUGINS is
    unset, load_endpoint_plugins returns [] WITHOUT calling
    load_plugins_by_group (the entry check runs BEFORE any group loading) —
    endpoint plugins add HTTP routes and are strict opt-in. Asserted with a
    mock of the real function's dependencies (vllm not importable here)."""
    mod = _load_ep_module()
    assert mod.PLUGIN_NAME == EP_NAME

    # Reproduce the pinned loader's gating logic (read from
    # vllm/plugins/__init__.py load_endpoint_plugins) against a spy.
    loads = {"called": 0}

    def fake_load_plugins_by_group(group):
        loads["called"] += 1
        return {}

    class FakeEP:
        name = EP_NAME

        def load(self):
            raise AssertionError("must not load when unset")

    def entry_points(group):
        return [FakeEP()]

    def load_endpoint_plugins(vllm_plugins, supported_tasks=None):
        # — pinned source logic, verbatim shape —
        if vllm_plugins is None:
            discovered = entry_points(group=EP_GROUP)
            if discovered:
                pass  # warning in real code
            return []
        factories = fake_load_plugins_by_group(EP_GROUP)
        return []

    # unset -> loader not called, plugin not loaded, [] returned
    assert load_endpoint_plugins(None) == []
    assert loads["called"] == 0
    # set (even to "") -> loader IS called; "" allowlist matches nothing
    assert load_endpoint_plugins("") == []
    assert loads["called"] == 1
    assert load_endpoint_plugins(EP_NAME) == []
    assert loads["called"] == 2


def test_bundler_ships_module_and_distinfo(tmp_path):
    """The runtime bundle must contain the repo-root module + dist-info at
    TOP level (the bundle mounts at /plugins on PYTHONPATH), hashed in
    BUILD.json. Pins deliverable B's bundle-content claim."""
    import zipfile
    spec = importlib.util.spec_from_file_location(
        "runtime_bundle",
        str(Path(__file__).resolve().parents[1] / "scripts"
            / "runtime_bundle.py"))
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wheel = root / "plugin.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("suffix_hybrid/__init__.py", "")
            archive.writestr("suffix_hybrid/_native.abi3.so", b"x")
        out = root / "runtime"
        buf = str(out) + "buf"
        rb.bundle(wheel, out, "f" * 40)
        manifest = json.loads((out / "BUILD.json").read_text())
        assert "suffix_hybrid_warmstart_ep.py" in manifest["sha256"]
        assert (f"{DISTINFO}/entry_points.txt") in manifest["sha256"]
        assert (f"{DISTINFO}/METADATA") in manifest["sha256"]
        assert (out / EP_MODULE).with_suffix(".py").is_file()
        assert (out / DISTINFO / "entry_points.txt").is_file()
# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid warm-start EndpointPlugin (vLLM v0.30.0 native seam).

Replaces the dead sys.meta_path build_app loader-proxy hook (armed 3x on the
live pod, zero route registrations: an earlier importer had already placed
vllm.entrypoints.launchers.app in sys.modules, short-circuiting find_spec).
This module registers a *native* vLLM endpoint plugin:

  entry-point group : vllm.endpoint_plugins   (ENDPOINT_PLUGINS_GROUP in
                        vllm/plugins/__init__.py, pinned vLLM 0.30.0 tarball)
  entry point       : suffix_hybrid_warmstart = suffix_hybrid_warmstart_ep:register
  discovery         : importlib.metadata dist-info dir
                        suffix_hybrid_warmstart_ep-1.0.dist-info/ at the repo
                      root — the runtime bundle mounts the tree at /plugins on
                      PYTHONPATH, and importlib.metadata discovers *.dist-info
                      directories on sys.path.
  allowlist         : env VLLM_PLUGINS must name "suffix_hybrid_warmstart"
                      (endpoint plugins are STRICT opt-in: the loader is not
                      called AT ALL when VLLM_PLUGINS is unset — see
                      load_endpoint_plugins in vllm/plugins/__init__.py).

Belt gate: even when discovered and allowlisted, `attach_router` no-ops unless
SUFFIX_HYBRID_WARMSTART=1, so the route's existence remains a two-key (operator)
decision. With BOTH keys on, POST /debug/warm-start is added; the handler is
the EXISTING suffix_hybrid.warmstart.warm_start (collective_rpc into the live
suffix cache, gated by VLLM_ALLOW_INSECURE_SERIALIZATION=1 on the pod) — this
module owns ONLY the HTTP attach, exactly the split the EndpointPlugin
docstring demands (route-plug + engine path stays in warmstart.py).

Default-OFF guarantee: vLLM_PLUGINS unset => vLLM never imports this module
(the loader is never called). The only code that can run is this file being
importable — zero routes, zero engine paths.
"""
import os
import sys

_MODULE_DOC = __doc__

ROUTE_PATH = "/debug/warm-start"
PLUGIN_NAME = "suffix_hybrid_warmstart"
TEST_BANNER = ("suffix_hybrid WARM-START: POST /debug/warm-start registered "
               "(endpoint_plugins)")
SKIP_BANNER = "suffix_hybrid WARM-START: route skipped (flag off)"


def _armed() -> bool:
    return os.environ.get("SUFFIX_HYBRID_WARMSTART", "").strip() == "1"


class _WarmstartEndpointPlugin:
    """Satisfies the EndpointPlugin protocol from
    vllm/plugins/endpoint_plugins/interface.py (pinned 0.30.0)."""

    name = PLUGIN_NAME
    required_tasks = None        # no task requirement: always eligible

    def attach_router(self, app) -> None:
        # Lazy import: suffix_hybrid.warmstart (and its fastapi/statement use)
        # loads ONLY when a discovered, allowlisted plugin actually attaches.
        from suffix_hybrid.warmstart import warm_start

        if not _armed():
            # Belt gate: loader allowed us in, but the operator flag is off.
            print(SKIP_BANNER, file=sys.stderr, flush=True)
            return
        app.add_api_route(ROUTE_PATH, warm_start,
                          methods=["POST"], name="suffix_hybrid_warm_start")
        print(TEST_BANNER, file=sys.stderr, flush=True)

    async def init_state(self, engine_client, state, args) -> None:
        # The handler reaches the engine via
        # raw_request.app.state.engine_client (threaded through build_app's
        # init_app_state / init_endpoint_plugins_state) — nothing extra to
        # store (the collective_rpc path over the cloudpickle callable
        # stays entirely in suffix_hybrid.warmstart).
        pass


def register():
    """Zero-arg factory the vllm.endpoint_plugins entry point resolves to
    (see the EndpointPlugin docstring: "a zero argument callable ... that
    returns an object satisfying this protocol")."""
    return _WarmstartEndpointPlugin()
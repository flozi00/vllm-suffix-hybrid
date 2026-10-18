"""Multi-model serving launcher: `python3 -m suffix_hybrid.multimodel`.

Contract: dossiers/multimodel-serving.md. Wiring only. It validates
SUFFIX_MULTIMODEL_CONFIG, boots one `vllm serve` child per model in sequence
(each with --enable-sleep-mode on 127.0.0.1 and all but the last put to sleep),
supervises them, and hands the public port to the Rust router
(`_native.multimodel_serve`). Routing, swapping and metrics run in Rust. Any
child exit or a router /health 503 (a failed swap) kills every child and
exits 1, so the pod restarts. A partial model set is never served.
"""
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("suffix_hybrid.multimodel")

ENV = "SUFFIX_MULTIMODEL_CONFIG"
OWNED_FLAGS = ("--host", "--port", "--served-model-name",
               "--gpu-memory-utilization", "--enable-sleep-mode")
TOP_KEYS = {"listen", "gpu_budget", "sleep_level", "slice_ms", "boot_timeout_s", "models"}
MODEL_KEYS = {"served_as", "port", "gpu_memory_utilization", "hostnames", "argv", "env"}
DEFAULTS = {"listen": "0.0.0.0:8000", "sleep_level": 1, "slice_ms": 2000, "boot_timeout_s": 1800}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _owned(arg):
    """True when `arg` sets (or abbreviates, argparse allow_abbrev) a launcher-owned flag."""
    if not arg.startswith("--"):
        return False
    flag = arg.split("=", 1)[0].replace("_", "-")
    if flag.startswith("--no-"):
        flag = "--" + flag[5:]
    return len(flag) > 2 and any(owned.startswith(flag) for owned in OWNED_FLAGS)


def load_config(raw):
    """Inline JSON, or a file path when the value starts with `/`."""
    if raw.startswith("/"):
        with open(raw, encoding="utf-8") as handle:
            raw = handle.read()
    return validate(json.loads(raw))


def validate(config):
    """Validate per the contract and fill the documented defaults. Raises ValueError."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if unknown := set(config) - TOP_KEYS:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    config = DEFAULTS | config
    budget = config.get("gpu_budget")
    if not _number(budget) or not 0 < budget <= 0.95:
        raise ValueError("gpu_budget must be a number in (0, 0.95]")
    if config["sleep_level"] not in (1, 2) or isinstance(config["sleep_level"], bool):
        raise ValueError("sleep_level must be 1 or 2")
    for key in ("slice_ms", "boot_timeout_s"):
        if not _number(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative number")
    if not isinstance(config["slice_ms"], int):
        raise ValueError("slice_ms must be an integer")
    if not isinstance(config["listen"], str) or ":" not in config["listen"]:
        raise ValueError("listen must be host:port")
    models = config.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("models must list at least 2 models")
    names, ports = set(), set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("every model must be an object")
        if unknown := set(model) - MODEL_KEYS:
            raise ValueError(f"unknown model fields: {sorted(unknown)}")
        name = model.get("served_as")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"served_as must be a unique non-empty string: {name!r}")
        names.add(name)
        port = model.get("port")
        if type(port) is not int or not 8100 <= port <= 8199 or port in ports:
            raise ValueError(f"{name}: port must be a unique integer in 8100-8199")
        ports.add(port)
        util = model.get("gpu_memory_utilization")
        if not _number(util) or not 0 < util <= budget:
            raise ValueError(f"{name}: gpu_memory_utilization must be in (0, gpu_budget]")
        hostnames = model.setdefault("hostnames", [])
        if not isinstance(hostnames, list) or not all(isinstance(h, str) and h for h in hostnames):
            raise ValueError(f"{name}: hostnames must be a list of strings")
        argv = model.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"{name}: argv must be a non-empty list of strings")
        if argv[0].startswith("-"):
            raise ValueError(f"{name}: argv must start with the model path")
        if owned := [a for a in argv if _owned(a)]:
            raise ValueError(f"{name}: argv sets launcher-owned flags {owned}")
        env = model.setdefault("env", {})
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f"{name}: env must map strings to strings")
    return config


def child_command(model):
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *model["argv"],
            "--host", "127.0.0.1", "--port", str(model["port"]),
            "--served-model-name", model["served_as"],
            "--gpu-memory-utilization", str(model["gpu_memory_utilization"]),
            "--enable-sleep-mode"]


def child_env(model, base=None):
    return dict(os.environ if base is None else base) | model["env"] | {"VLLM_SERVER_DEV_MODE": "1"}


def _status(url, method="GET", timeout=10.0):
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def wait_ready(proc, port, deadline):
    """Poll the child's /health until 200; False on exit or deadline."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if _status(f"http://127.0.0.1:{port}/health", timeout=5) == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def kill_all(procs):
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 30
    for proc in procs:
        try:
            proc.wait(max(0.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:  # the group also holds vLLM's EngineCore processes
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def supervise(procs, models, router_port):
    """Any child exit or a router /health 503 (Failed swap) restarts the pod."""
    while True:
        for proc, model in zip(procs, models):
            if (code := proc.poll()) is not None:
                logger.error("SUFFIX_MULTIMODEL FATAL child %s exited with %s; "
                             "killing all children", model["served_as"], code)
                kill_all(procs)
                os._exit(1)
        try:
            unhealthy = _status(f"http://127.0.0.1:{router_port}/health", timeout=10) == 503
        except OSError:
            unhealthy = False  # router not bound yet; it logs its own bind errors
        if unhealthy:
            logger.error("SUFFIX_MULTIMODEL FATAL router /health 503; killing all children")
            kill_all(procs)
            os._exit(1)
        time.sleep(1)


def boot(config):
    """Spawn children one at a time; every child but the last is put to sleep."""
    procs, models = [], config["models"]
    try:
        for index, model in enumerate(models):
            start = time.monotonic()
            proc = subprocess.Popen(child_command(model), env=child_env(model),
                                    start_new_session=True)
            procs.append(proc)
            if not wait_ready(proc, model["port"], start + config["boot_timeout_s"]):
                raise RuntimeError(f"{model['served_as']} not ready within "
                                   f"{config['boot_timeout_s']}s (exit={proc.poll()})")
            logger.info("SUFFIX_MULTIMODEL BOOT %s ready in %.1fs",
                        model["served_as"], time.monotonic() - start)
            model["awake"] = index == len(models) - 1
            if not model["awake"]:
                url = (f"http://127.0.0.1:{model['port']}/sleep"
                       f"?level={config['sleep_level']}&mode=wait")
                if (code := _status(url, "POST", timeout=600)) != 200:
                    raise RuntimeError(f"{model['served_as']} /sleep returned {code}")
                logger.info("SUFFIX_MULTIMODEL BOOT %s asleep", model["served_as"])
    except BaseException:
        kill_all(procs)
        raise
    return procs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = os.environ.get(ENV)
    if not raw:
        logger.error("%s is not set; refusing to start the multimodel launcher", ENV)
        return 1
    try:
        config = load_config(raw)
    except (OSError, ValueError) as error:
        logger.error("SUFFIX_MULTIMODEL invalid config: %s", error)
        return 1
    from suffix_hybrid import _native
    if not hasattr(_native, "multimodel_serve"):
        logger.error("SUFFIX_MULTIMODEL _native lacks multimodel_serve; rebuild the plugin")
        return 1
    try:
        procs = boot(config)
    except (OSError, RuntimeError) as error:
        logger.error("SUFFIX_MULTIMODEL BOOT failed: %s", error)
        return 1
    router_port = int(config["listen"].rsplit(":", 1)[1])
    threading.Thread(target=supervise, args=(procs, config["models"], router_port),
                     name="mm-supervisor", daemon=True).start()
    try:
        _native.multimodel_serve(json.dumps(config))
    except RuntimeError as error:
        logger.error("%s", error)
        kill_all(procs)
        return 1
    kill_all(procs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

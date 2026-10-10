# SPDX-License-Identifier: Apache-2.0
"""GPU telemetry lines in the pod log (SUFFIX_GPU_MON=<seconds>, AMD).

One process per pod (O_EXCL marker in /tmp) runs a daemon thread that reads the
amdgpu hwmon files every <seconds>: temperatures (edge / junction / mem),
socket power, shader and memory clocks. Shows whether a sustained bench runs
into thermal or power throttling (the MI350P's c32 throughput drifted 13%
between a 60 s and a 120 s window with constant MTP acceptance). Stdlib only,
read-only sysfs, never raises into the host process.
"""
import glob
import os
import sys
import threading
import time

ENV = "SUFFIX_GPU_MON"
MARKER = "/tmp/suffix_gpu_mon.claimed"


def _read(path: str):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def hwmon_dirs() -> list:
    return sorted(d for d in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")
                  if _read(os.path.join(d, "name")) == "amdgpu")


def sample(d: str) -> str:
    parts = []
    for key in ("temp1", "temp2", "temp3"):
        label = _read(os.path.join(d, f"{key}_label")) or key
        val = _read(os.path.join(d, f"{key}_input"))
        if val is not None:
            parts.append(f"{label} {int(val) / 1000:.0f}C")
    for key in ("power1_average", "power1_input"):
        val = _read(os.path.join(d, key))
        if val is not None:
            parts.append(f"power {int(val) / 1e6:.0f}W")
            break
    cap = _read(os.path.join(d, "power1_cap"))
    if cap is not None:
        parts.append(f"cap {int(cap) / 1e6:.0f}W")
    for key in ("freq1", "freq2"):
        label = _read(os.path.join(d, f"{key}_label")) or key
        val = _read(os.path.join(d, f"{key}_input"))
        if val is not None:
            parts.append(f"{label} {int(val) / 1e6:.0f}MHz")
    return " ".join(parts) or "no readable sensors"


def _loop(period: float, dirs: list) -> None:
    while True:
        for i, d in enumerate(dirs):
            print(f"[suffix gpu-mon] gpu{i} {sample(d)}", file=sys.stderr, flush=True)
        time.sleep(period)


def start() -> bool:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return False
    try:
        period = float(raw)
        os.close(os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except (ValueError, OSError):
        return False  # bad value, or another process of this pod already monitors
    dirs = hwmon_dirs()
    print(f"[suffix gpu-mon] {len(dirs)} amdgpu hwmon dir(s), every {period:g}s",
          file=sys.stderr, flush=True)
    if dirs:
        threading.Thread(target=_loop, args=(period, dirs), daemon=True,
                         name="suffix-gpu-mon").start()
    return bool(dirs)

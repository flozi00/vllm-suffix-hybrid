#!/usr/bin/env python3
"""Dev-mode argv splice: iterate on serving flags WITHOUT a pod roll.

Gated by SUFFIX_HYBRID_DEV_ARGS=1 (dev pools only; prod never sets it). When
enabled and /plugins/dev_args.json exists, its entries are merged into
sys.argv before vllm's parser runs, then the file is archived to
dev_args.last.json so every process start logs exactly one effective delta.

Entry forms (JSON list):
  "--flag"              bare boolean flag (appended if absent)
  ["--flag", "value"]   flag with value; overrides an existing occurrence
  ["--flag", null]      removes every existing occurrence of the flag

Overrides preserve the first occurrence's position; new flags append at the
end. The positional model path and --model/--served-model-name are never
touched (a dev arg must not be able to swap the checkpoint).
"""
import json
import os
import sys

PATH = "/plugins/dev_args.json"
ARCHIVE = "/plugins/dev_args.last.json"
_RESERVED = {"--model", "--model-path", "-m", "--served-model-name"}


def _tokenize(argv):
    """[(flag, [flag, values...]) | (None, [positionals...])] preserving order."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            name = tok.split("=", 1)[0]
            group = [tok]
            i += 1
            # A value is the next token unless it is itself a flag.
            if "=" not in tok and i < len(argv) and not argv[i].startswith("--"):
                group.append(argv[i])
                i += 1
            out.append((name, group))
        else:
            out.append((None, [tok]))
            i += 1
    return out


def splice(argv, entries):
    tokens = _tokenize(argv)
    for entry in entries:
        # Rebuilt per entry: a removal shifts positions, so a cached index
        # would point at the wrong slot for every later entry.
        index = {name: pos for pos, (name, _) in enumerate(tokens) if name}
        if isinstance(entry, str):
            flag, value = entry, None
        elif isinstance(entry, list) and len(entry) == 2:
            flag, value = entry
        else:
            raise ValueError(f"bad dev_args entry: {entry!r}")
        if not flag.startswith("--"):
            raise ValueError(f"dev_args flag must start with --: {flag!r}")
        if flag in _RESERVED:
            raise ValueError(f"dev_args may not change {flag}")
        if value is None and isinstance(entry, list):
            tokens = [t for t in tokens if t[0] != flag]
            continue
        group = [flag] if value is None else [flag, str(value)]
        if flag in index:
            tokens[index[flag]] = (flag, group)
        else:
            tokens.append((flag, group))
            index[flag] = len(tokens) - 1
    return [tok for _, group in tokens for tok in group]


def apply_dev_args(argv=None):
    """Mutate sys.argv in place. Returns the applied entries, or None (no-op)."""
    argv = sys.argv if argv is None else argv
    if os.environ.get("SUFFIX_HYBRID_DEV_ARGS", "").strip() != "1":
        return None
    # sitecustomize runs in EVERY python process of the container (workers,
    # EngineCore, spawn children). Only the `vllm serve` entrypoint carries
    # serving flags; splicing a worker's argv would corrupt multiprocessing
    # bookkeeping. Identify the entrypoint by argv shape, not PID (a kill-1
    # restart keeps PID 1, but a shell-launched debug run must also work).
    if not (argv and "serve" in argv[:3]
            and os.path.basename(argv[0]).startswith("vllm")):
        return None
    if not os.path.exists(PATH):
        return None
    try:
        with open(PATH) as fh:
            entries = json.load(fh)
        if not isinstance(entries, list):
            raise ValueError("dev_args.json must be a JSON list")
        merged = splice(list(argv), entries)
    except Exception as exc:
        # Fail closed: an enabled dev pool with a broken dev_args must not
        # silently serve with stale flags.
        raise SystemExit(f"suffix_hybrid dev_args rejected: {exc}") from exc
    argv[:] = merged
    # Keep the file in place: a kill-PID1 container restart re-runs the
    # ORIGINAL deployment argv, so the dev delta must re-apply on every
    # process start. last.json is an audit copy of the most recent delta.
    with open(PATH) as fh:
        with open(ARCHIVE, "w") as out:
            out.write(fh.read())
    print(f"suffix_hybrid dev_args applied: {json.dumps(entries)}",
          file=sys.stderr, flush=True)
    return entries

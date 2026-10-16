# bench/

Served-model speed, quality and A/B gating live in the suite at
`inference-console/benchmarks` (see its README). Run it from the
inference-console checkout, A1 (reference) -> B (candidate) -> A2 on an idle
dev pool, never on production or `glm`:

```sh
uv run benchmarks/quality.py --url https://<pool>.pl-ai.net --arm A1             # quick tier
python3 benchmarks/speed.py  --url https://<pool>.pl-ai.net --arm A1 --rounds 5
# same two commands with --arm B (candidate rolled out), then --arm A2 (rolled back)
python3 benchmarks/compare.py --a runs/<A1-quality> runs/<A2-quality> --b runs/<B-quality>
python3 benchmarks/compare.py --a runs/<A1-speed> runs/<A2-speed> --b runs/<B-speed>
```

Removed on 2026-10-16 as superseded: `ovh/qgate.py`, `quality_probe.py` ->
`quality.py`; `ovh/abbench.py`, `bench.py`, `bench_reuse.py`, `compare.py`,
`gemma-research/bench_gemma.py` -> `speed.py`; `ovh/gatecmp.py`,
`golden_rule.py` -> `compare.py`; `scripts/ab_glm.sh` -> the suite. Do not
restore them. Branches cut earlier (e.g. `lfm2-moe`, which also adds
`ovh/lpdrift.py`, superseded by the `drift` task) drop them when merging.

What stays here (in-process tools, no served-model load):

| File | Purpose |
|---|---|
| `hook_microbench.py`, `test_hook_microbench.py` | CPU microbenchmark of the V2 hook's `propose()` hot path (real Rust mixer, faked runner) |
| `scale_bench.py` | `mix_numpy` / `Engine.propose` cost vs rows and context length |
| `host_path_bench.py` | host µs per call of per-step Python paths (`cpu`, or in-pod `gpu` / `embed`) |
| `spec_sim.py` | CPU trace-driven accepted-tokens/step simulator for the mixer |
| `parse_gating.py` | gating-counter parser for pod logs |
| `ovh/cpuprobe.py` | API-server CPU cores from `/metrics` (sends no requests) |
| `RUNBOOK-next-spec-dev.md`, `results.jsonl`, `quality/*.json` | history of the 2026-09 next-spec-dev campaign |

`bench/*` is gitignored: tracked files are force-added (`git add -f`). Local
untracked files such as `gemma-research/wait_for_ready.py` (the readiness
primitive plugin-harness calls) are not benchmarks and stay.

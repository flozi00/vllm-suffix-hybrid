# next-spec-dev execution runbook — suffix-hybrid golden-rule pipeline

Status: ALL offline prep complete (2026-09-23). Blocked ONLY on the 10.70/16 route.
Never touch `qwen-flash-next-fp4` (production). All work targets pool `next-spec-dev`,
namespace `maas-inference`. Baseline = plugin-inert p18 (~480 c1 / ~1880 c8).

## Preconditions (check each iteration)
- Route: `curl -s --max-time 8 http://10.70.0.45/api/state` — anything but 000 = GO.
- Bundle live on GitHub: orphan branch `runtime-a151b32d` (ref `2722c477`, 16 files,
  wrap_v2.py 49233 B, BUILD.json 2002 B). Carries p24 (`SUFFIX_HYBRID_VERIFIER_WIN=1`)
  AND p22 (`SUFFIX_HYBRID_ZERO_OP=1`) selectable by env; default = exact p17 semantics.
- Env patches staged: `~/.hermes/cache/scratch/nsd_env_patches.json`
  (p24-verifierwin 11-env, p22-zeroop 11-env, p23-inertcap 3-env, p21-floor, fp8_kv args patch).
- edit_pool env patch REPLACES the whole env list — always dry_run=true first,
  verify `resolved_spec` env list byte-exactly, then apply.

## Queue (in order)

### 1. p24-verifierwin (the c8 attack)
1. `download_weights` repo=flozi00/vllm-suffix-hybrid ref=runtime-a151b32d source=github
   (pad BUILD.json unique size if bucket sync skips same-size objects) → verify 16 objects.
2. edit_pool: apply staged `p24-verifierwin` env (11 rows, VERIFIER_WIN=1; dry-run first).
3. Wait for new pod ready (~15–20 min, 0 restarts expected). `get_logs` until `installed` line.
4. **BEFORE any bench counts**: run `get_logs` capture →
   `python3 bench/parse_gating.py --window <capture>` — the measured c8-window share
   is mandatory (parse_gating mandate; old parser undercounted 3–9×).
5. Bench c1×2 + c8×2:
   `python3 bench/bench.py --endpoint https://next-spec-dev.pl-ai.net/v1 --model next-spec-dev --num-prompts 16 --concurrency 1 --label p24-c1-N`
   (c8: `--num-prompts 64 --concurrency 8`).
6. Append rows to `bench/results.jsonl`; run `python3 bench/golden_rule.py`.
7. ACCEPTANCE (pre-registered): c8-window share 58% → ≤10–15%; c8 ≥ 1800 (baseline 1880).
   Fallback if share > 12%: env-only knob ladder hb=64/cd=1 (sim: 6.5% share), re-bench.
8. Rollback if acceptance collapse (p20 pattern): bucket → runtime-a4e8f002,
   env → p19-wraponly 10-env list.

### 2. p23-inertcap (same-hour inert reference + c1 H2 test)
Staged 3-env inert list WITH log capture this time (p4wvq hole). c1×2/c8×2.
Baseline refresh datum; also re-measures environment drift.

### 3. p22-zeroop (c1 discriminator: machinery vs presence)
Staged 11-env (ZERO_OP=1, same bundle). Verification signature: zero `skips_*` counters
EXPECTED (mixer never runs); armed = `installed` line + acceptance ~9.8.
Interpretation pre-registered: c1 ≈ 480 → frozen path wasn't actually taken (engine bug);
c1 ≈ 410 → residual is hook presence/env → engine-level fix (move mix out of propose).

### 4. bench_reuse A/B (the workload where suffix wins materialize)
`python3 bench/bench_reuse.py --endpoint ... --concurrency 1 --label reuse-plugin-N`
after p24 warm, then again on p23-inert. Unique-prompt bench structurally cannot show
suffix wins (~0.5% flat); corpus-reuse is where acceptance pays.

### 5. fp8-KV (capacity lever — pre-informed, user's call)
NOT a golden-rule path alone: pre-loop live revert recorded −27% c1 p50 throughput
(commit 600c97d1); quality verified neutral (bench/quality artifacts, wakeup #37).
Staged args patch: `{"args": {"--kv-cache-dtype": "fp8_e4m3"}}` (rollback: `"auto"`).
Measure: KV capacity/pool-size gain (~2× expected) vs throughput cost at bench scale.

## Closed levers (do NOT revisit for this QSA checkpoint)
- NVFP4-KV: QSA Triton attention never routes through FA2 (qsa.py:64,72–77,96–98);
  also live-reverted pre-loop (7c7a4ad7). Patch remains valid for standard FA2/GQA models.
- MoE backend: marlin is the only NVFP4 MoE path on SM120 at this rev.
- Draft width k: QSA ring caps k at 12 (k=16 infeasible, 3be90377); wrap k=4 is fine.
- Fused multi-step decode: QSA blocks it on all pods.
- Stale-draft staging (p20): FAILED, retired — never re-ship.
- Rust frontend confound: closed — console pins VLLM_USE_RUST_FRONTEND=0 when absent
  (manifests.py:921-922), so inert and armed both run Python frontend.

## Known numbers (true calls-based shares)
p17 26.4% share → c1 413 / c8 1659 (Δ −67/−221 vs baseline); stall fit
`step_ms = 23.34 + 2.83 × share`; c1-phase share ≈4.5% with ~3.0 ms/step residual
(machinery bounded 1.2–1.7 µs; frozen path source-audited clean — residual = hook
presence or env drift, discriminated by p22/p23). Churn sim: p17 semantics 100% share
under c8 churn vs p24 10.5% (9.5× collapse, /tmp/churn_sim.py pattern in skill ref).

## Unlock options (user-side, any one)
1. Office LAN. 2. Tailscale re-auth + re-advertise 10.70/16 subnet route.
3. Complete the interactive Tailscale-SSH browser check during a live SSH attempt to
   `primeline@100.92.182.59` → primeline-1 (GB300, reachable, ping ~24 ms) as jump host.
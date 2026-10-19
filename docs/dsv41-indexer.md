# DeepSeek-V4.1-Flash sparse-attention indexer on SM120 (blind lane C, 2026-10-19)

Branch `dsv41-indexer`, base d0fec38 / 8a75f0d. No GPU, no cluster, nothing committed. Everything below is
proven on CPU (torch 2.14 CPU, Helion 1.5.1 eager mode + Triton **source** generation, vLLM source at
`39983a5e7a5a575a7829f5d5ada50a33d9337ec1`, pinned in `tests/fixtures/vllm_39983a5e7/`). Anything about speed is an
**estimate** with its derivation; silicon decides (section 9). Owner direction applied (coordinator message):
Helion first and auto-tuned, oxide only where Helion has not won; no barriers / semaphores / last-arriver merges in
graph-captured paths; scratch only from torch's allocator; no host sync; everything warmed before capture.

## 1. What stock vLLM runs on SM120 for this model (source-anchored)

Model facts (config.json): 8 indexers on layers 2, 8, 14, 20, 24, 28, 32, 36 (`index_source_layer_ids`), 32 heads x 128,
top-512; compress ratio 2 on layers 2-19 and 1 on 20-39, so layers 2/8/14 scan **ctx/2** index positions and 20..36 scan
**ctx**; K comes from the kv-source layers' latent (4 caches: layers 2, 8, 14, 20; 24..36 read layer 20's). Candidate
source = layer 20 (writes the top 2048 blocks of 8 positions); layers 24/28/32/36 only rank inside them. The indexer is
replicated, every rank runs all of it.

| step (decode, per indexer layer) | what runs | evidence |
|---|---|---|
| rows | `use_flattening=True supports_varlen=False (next_n=6)`: SM120 only takes `next_n in (1, 2)` natively, so DSpark's 1+5 rows become **6 single-row requests, each with its own copy of the block-table row** ("re-reads the KV tile once per row") | boot5.log:352; `v1/attention/backends/mla/indexer.py` `_supports_native_decode`, `_prepare_decode_tensors` (fixture `indexer_backend.py.txt`) |
| logits | vendored DeepGEMM `fp8_fp4_paged_mqa_logits` (the shim is **inert** in boot5: `SUFFIX_SM120` unset), pages of 64 only (`get_paged_mqa_page_sizes`), fp32 `[rows, max_model_len]`; K re-read per row | boot5.log:1; `deep_gemm.py:37-40` |
| layer 20 (source) | `select_candidate_blocks`: `_block_scores_kernel` with **FULL_WIDTH** (SM120 has no DeepSelect) = reads the whole `max_model_len`-wide row whatever the context, then `torch.topk(k=2048)` over nblocks = 65536 per row | `candidate_blocks.py` |
| layers 24/28/32/36 (consumers) | dense logits over the whole context, then `apply_candidate_mask`: a flags kernel + a **full-width -inf rewrite of every row** (rows x 2 MB at 524288, at every context) | `candidate_blocks.py`, `sparse_attn_indexer.py:895-922` |
| top-k | `persistent_topk` (cooperative excludes the 12x family), radix select, ~3 row passes | `indexer_topk.py:173-276` |
| graph | decode is a FULL graph captured once with `max_seq_len = max_model_len` ("valid at any replay"), so the indexer is inside the capture; PIECEWISE (prefill / mixed) breaks around it | `model_states/default.py:198`, dossier `dsv41-speed-2026-10-19.md` finding 1 |
| prefill | `fp8_fp4_mqa_logits(..., max_seqlen_k)` returns COMPRESSED `[M, max_seqlen_k]`; candidate mask / selection on it; `top_k_per_row_prefill` | `sparse_attn_indexer.py:723-772` |

The sparse DeepGEMM kernels (`fp8_fp4_paged_sparse_mqa_logits`, MXFP4 cache) are SM100-only; nothing upstream restricts
the dense scan on SM120. Our fused candidate kernel is the SM120 analogue.

## 2. Per-step cost of the stock path (estimate, `kernels/dsa_indexer_cost.py`)

Bytes over the HBM roofline (1.79 TB/s), no L2 credit for stock (the 6 flattened rows of a request may share lines in L2
on silicon: that is exactly what the lab measures, so these are upper bounds for stock), full MTP-row sharing for ours.
`python3 -I suffix_hybrid/kernels/dsa_indexer_cost.py`:

```
ctx      reqs rows  stock MB   ours MB   stock us  ours us  launches stock/ours
32768    1    6        263.5      35.1     147.2     19.6   29/22
32768    8    48      2107.6     281.1    1177.5    157.1   29/22
32768    32   192     8430.6    1124.6    4709.8    628.3   29/22
131072   1    6        830.9      94.7     464.2     52.9   29/22
131072   8    48      6646.9     757.7    3713.4    423.3   29/22
131072   32   192    26587.7    3030.9   14853.5   1693.2   29/22
524288   1    6       3100.5     333.0    1732.1    186.0   29/22
524288   8    48     24804.1    2664.0   13857.0   1488.3   29/22
524288   32   192    99216.3   10656.2   55428.1   5953.2   29/22
```

One 512k request, per layer, MB: stock dense (ratio 2) 233, source 486, consumer 479 (K scan 415 = 6 rows x 524288 x 132 B,
logits write 12.6, **mask pass 13.4**, top-k reads 37.8); ours dense 60, source 139, consumer **3.8**. The K scan of the
four consumers (4 x 415 MB) disappears, the K scan of the dense/source layers drops 6x, and the mask pass and the
`torch.topk` become independent of the context. The coordinator's dossier estimates the stock indexer at ~0.85 ms per 512k
sequence-step (some L2 credit); against the model's 1.73 ms roofline that brackets the real stock cost between ~0.85 and
~1.7 ms and ours at ~0.1-0.2 ms.

## 3. What was built (files)

| file | what |
|---|---|
| `suffix_hybrid/helion_kernels/dsa_indexer.py` | `DeepseekV41ForCausalLM` registered (INDEX_HEADS 32; CAND (2048, 8)); bodies: per-row decode, **row-group decode** `dsa_group_logits`, prefill (compressed + absolute), fused candidates `dsa_cand_logits`, lab-only survivor select `dsa_group_select`; free page layout (P % 32, packed stride), serving entries (`group_logits`, `cand_logits`, `prefill_logits`, `warm_all`), lab ABI `KERNELS` (7 entries incl. the DeepGEMM baselines) |
| `suffix_hybrid/kernels/dsa_indexer.py` | routing `oxide | helion-group | helion | triton`, K5 `note_path`/`strict`, DSpark = 1+k rows, `capture_replay` CLI mode |
| `suffix_hybrid/kernels/dsa_candidates.py` | consumer-layer candidate fusion (patched `_apply_candidate_mask`/`get_indexer_topk`, anchored), candidate-source block selection (`CANDSEL`), one-launch Triton slot remap |
| `suffix_hybrid/kernels/dsa_capture.py` | capture discipline: `launch` guard, `serving_violation`, `search_filter` |
| `suffix_hybrid/kernels/dsa_select.py` | warm-start threshold select: exactness proof + CPU reference (design, not wired) |
| `suffix_hybrid/kernels/dsa_indexer_gate.py` | boot gate `dsa_indexer_capture_replay` |
| `suffix_hybrid/kernels/dsa_indexer_lab.py` | one-run silicon driver (oracle, gate, tune, per-layer benchmark, report) |
| `suffix_hybrid/kernels/dsa_indexer_cost.py` | byte/launch model above |
| `sm120/deep_gemm_shim/sm120_fallback.py` | Triton fallback honours the compressed prefill contract (6th arg int = `max_seqlen_k`) |
| `tests/test_dsa_*.py`, `test_helion_dsa_indexer.py`, `test_dsv41_indexer_anchors.py`, `tests/fixtures/vllm_39983a5e7/` | proofs below |

## 4. Design

**(a) Registration.** `DECODE_ROWS` 1..384 (6 DSpark rows per request, up to 32 requests = 192 rows + padding), `GROUP`
cases `{heads, rows, group: 6, ctx, numtokens}` for ctx buckets 32768 (1-32 requests) and **524288** (1, 2, 4 requests: KV
capacity bounds long-context concurrency), fused-candidate cases at 2048 x 8, `SELECT` cases (lab only). Prefill keeps M
buckets 256-8192 (V4.1's chunk logits are bounded by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, M x N <= 134M, so deep contexts run
few rows: the lab benchmarks (8192, 8192), (8192, 32768), (1024, 131072), (256, 524288)). The per-row decode body stays tuned at
ctx 32768 (its shipped GLM keys are unchanged); the row-group body supersedes it for DSpark.

**(b) Split-context decode that fills the machine.** One program = (group of G consecutive rows) x (one split of their union
window). The split range is computed on the device from the row lengths (`per = max(ceil(tiles / eff), min_tiles)` 64-token
tiles), the grid is `reqs x splits` with `splits` a host function of shapes only (`splits_for`: ~8 programs per SM over the
request groups, **rounded to a multiple of 16** so the grid width is one Triton specialisation class): c=1 at 512k launches
1504 programs of ~5 tiles over 188 SMs, c=32 launches ~48 per group. There is no split-K across programs: logits are
independent per position, so every program writes disjoint columns and nothing is merged (nothing to deadlock, nothing to
reset between replays). Tunables: `block_n` 16-256, `min_tiles`, `split_div`, num_warps, num_stages, eviction policy.
**Row group:** each row still addresses K through its own block-table row and masks its own length, so *any* grouping is exact
(adaptive verification, straddling groups, padding rows, `rows % 6 != 0` are all covered by tests); rows of one request
resolve to the same cache lines, so the 2nd-6th load of a K tile is an L1 hit and HBM sees K once (K / G). The row loop is a
rolled device loop: the config space stays that of the per-row body (8 eviction / 9 indexing entries, not 42 / 48 when
unrolled) and the compile is 2x shorter. Dead lanes issue no block-table read (`hl.load(extra_mask)`), read K from page 0 and
store nothing. K4 on silicon must show the L1 reuse; oxide's page-dedup group is the baseline where it can read the layout.

**(c) Fused top-k, three levers, decreasing certainty.**
1. *Candidate fusion* (consumers, built, graph-safe, env `..._CAND`): scores only the 16384 candidate positions straight into
   compressed `[rows, 16384]` logits, vLLM's persistent top-k ranks them with per-row lengths, one Triton launch maps slots to
   positions. Bytes per consumer layer no longer grow with the context; the full-width mask pass disappears. Decided from
   static facts only (a decode graph is captured once), so there is no context gate: the dense path pays rows x 2 MB at every
   context.
2. *Valid-block selection* (candidate source, built, env `..._CANDSEL`): vLLM's own `_block_scores_kernel` with FULL_WIDTH off
   (valid blocks only) + `persistent_topk(k=2048)` instead of `torch.topk` over 65536 block scores; same set up to ties.
3. *Warm-start threshold select* (dense layers, `dsa_select.py`, **design + exact reference + lab-only Helion body**, not
   wired). After row sharing the dense layers still write 4 B/position/row of logits and read them ~3x in the top-k (96 B vs 132 B
   of K per position at G = 6). Exact scheme: score only the previous step's top-512 plus 512 evenly spaced positions (distinct
   positions only, sorted), take their **k-th largest** as a bound thr; the k-th largest of a subset of the window never exceeds
   the true k-th largest, so every true top-k element has logit >= thr (any warm set is exact, a bad one only keeps more
   survivors); the scan writes only logit >= thr packed behind a per-row cursor; persistent_topk ranks the packed row (rows <= k
   return 0..len-1), a table lookup maps slots back. Tie rule = the stock path (arbitrary at the k-th value, same set above it);
   f32 slack 1e-5 relative on thr. Tests: random / stale / all-equal / all-zero-relu / boundary ties / ctx 0, 1, 7, 24, 25, 26 /
   ctx 524288 (`tests/test_dsa_select.py`, 20 cases); an unsorted/duplicated warm set was a real bug there (a repeated position
   lifts the k-th largest above the truth): positions are deduplicated. It stays lab-only because (i) the survivor cursor is an
   atomic claim (no waiting, but an order race and a memset node), and your rule for captured paths is two-stage partials + a
   separate merge: if the lab shows the select winning, port it to per-tile counts -> scan -> scatter first; (ii) its payoff
   depends on the trained model's top-k stability, which only a serving A/B can measure.

**(d) Oracle.** `check()` = `|out - f64| <= 1e-5 x sum|terms|` on each row's window, -inf/unwritten elsewhere, f64 spec
through the cache views; adversarial CPU cases: ctx 1, 2, 6, 63-65, 127-129, 191-193 at pages 32/64/128 with packed strides,
rows with cl 0, groups straddling requests, `rows % 6 != 0`, contexts past the logits width, all-equal and all-zero logits, one
524288-token request x 6 rows (f64 spot-check of edge columns + 1500 random per row), poison page ids beyond the cache for dead
lanes (the stand-in raises on any unmasked dereference; a mutant without `extra_mask` fails it). GPU: `dsa_indexer_lab`
`oracle` task + the capture gate (f64 on the first replay, ctx up to 524288).

**(e) Lab.** `python -m suffix_hybrid.kernels.dsa_indexer_lab [budget_s]` on one PRO 6000 (plugin mounted, vLLM
39983a5e7): `oracle` -> `gate` -> `tune` (full-effort Helion per case: group, cand, select, decode, prefill; baseline oxide, accuracy
check vs the f64 spec) -> `bench_layer` for every (dense@ctx/2 | source | consumer) x ctx 8k..524288 x 1/4/32 requests (stock =
vendored DeepGEMM -> mask/selection -> persistent_topk as one graph; ours = oxide (reference), per-row, row group at the
**serving config** (longest bucket) and at the config tuned for that context, candsel, cand_fused, warm select at 0/10/50 %
stale) -> `bench_prefill` -> `report`: winner table, **decode-step totals (3 dense + source + 4 consumers) stock vs ours**,
crossovers per request count (first context where each lever / ours-vs-stock wins), the cost of serving one config at every
context (`serving_policy_penalty`), next to the byte model. Resumable (`/tmp/dsa_indexer_lab.json`).

**(f) K5.** Every indexer call logs `layer <name>: decode|prefill path <helion-group|helion|oxide|cand-fused|triton|...>` once
per layer at startup (vLLM's capture warm-up runs every layer eagerly first); `SUFFIX_SM120_DSA_INDEXER_STRICT=1` turns any
Triton fallback into a startup error. Family check: only SM120 gets the shim wrappers; MXFP4 / varlen / non-FP8 shapes route to
the fallback and are refused in strict mode.

## 5. Graph-capture discipline (coordinator rules, enforced in code)

* **No JIT inside a capture.** Helion passes sizes/strides/page parameters as runtime ints that Triton specialises on (== 1,
  % 16, other) plus 16-byte pointer alignment and the int32/int64 index class. `dsa_capture.launch` records the specialisation
  class of every eager launch and **refuses the first launch of an unseen class while a capture is open** (precise
  RuntimeError instead of a mid-capture compile). `_warm` compiles every shipped bucket in every row-count class the serving
  shapes reach (rows 1 | other | % 16; groups 1/2/8/16 at G = 6), keyed on the cache's index class, logits-width class and
  block-table-width class; `warm_all(...)` re-runs it for a given cache; `splits` is always a multiple of 16.
  Same for the Triton slot remap (`warm_remap`), vLLM's `_block_scores_kernel[FULL_WIDTH=False]` (gate warm-up) and the one-time
  patching of vLLM's module (`maybe_fuse` refuses to arm inside a capture).
* **Scratch only from torch's allocator** (`torch.empty` inside the capture = graph pool, stable across replays); the lane keeps
  no first-use GPU state and no persistent buffer of its own. **No host sync** (nothing reads a device value on the host).
* **No cross-program ordering** in any served kernel (see 4b); the only atomic cursor is the lab-only select.
* **Boot gate `dsa_indexer_capture_replay`** (`python -m suffix_hybrid.kernels.dsa_indexer capture_replay [--quick]
  [--default-configs]`, also `compute-sanitizer --tool memcheck --error-exitcode 99 python -m ... capture_replay --quick`):
  for entries `row`, `group`, `cand` (+ persistent top-k + remap when vLLM imports), `candsel` and cases V4.1-like (pages 64 /
  128+128 B pad / 32+128 B pad, 1x6, 4x6, 3x5 (padded group), 8x6, 1x6 at ctx 524288): `warm_all` is the **only** thing that
  runs before the capture (the captured call has never executed eagerly); capture on a side stream; then 4 replays with **every
  input rewritten in place** (K pages incl. garbage pad bytes, q, weights, block tables, candidate blocks, per-row lengths incl.
  0 / 1 / page and tile edges / at and past the logits width) and an eager launch of the same call on the same inputs,
  compared **bit for bit** on every column a consumer reads (top-k as sets); first replay also against the f64 spec. Shipped
  configs are exercised as shipped (a missing one prints `not shipped`, never a silent pass); `--default-configs` forces the
  bounded Helion default on every kernel for the lab. CPU proof of the schedule and the guard: `tests/test_dsa_indexer_gate.py`
  (stand-in graph that forbids unwarmed launches, a stale-replay detector, in-place refill with edge lengths).

## 6. Bounded search space (LFBO full effort)

Rolled row loop (8-11 list entries instead of 42-48), `block_n` 16-256 (decode 16-512), `min_tiles`/`split_div` powers of two
1-16, and `autotune_config_filter=dsa_capture.search_filter` on every KERNELS entry: each candidate is clamped before it is
compiled or timed (no persistent `pid_type`, no warp specialisation, <= 8 warps, <= 4 stages). The serving side refuses a shipped
config with a persistent grid or a warp-specialised loop (`serving_violation`, checked in `_warm`). 33 tune cases (group 9,
cand 8, select 4, decode 8, prefill 4); the lab runs them in that order so the K4 answer for the group body comes first.

## 7. Gate status (K1-K5; K6 = main session)

| kernel | K1 contract | K2 compile | K3 oracle | K4 bench vs replaced | K5 assert |
|---|---|---|---|---|---|
| row-group decode (`dsa_group_logits`) | written (section 4b) | Triton **source** generated for the rolled body, defaults; no-spill / 99 KB / cubin load need silicon | CPU: stand-in + Helion eager + codegen, edge/ties/524288 | needs silicon: lab `tune` + `bench_layer` (baseline oxide group, DeepGEMM) | `note_path` + strict |
| fused candidates + remap | written | codegen only; Triton remap compiles on first (warmed) launch | CPU vs the dense mask path, 3 problems | silicon: `bench_layer` consumer | yes |
| valid-block selection | written | uses vLLM's own kernel | CPU twin vs stock set, anchored signature | silicon | yes |
| survivor select (lab-only) | written (`dsa_select.py`) | codegen incl. atomic cursor + cumsum | CPU reference 20 cases + Helion eager + survivor-set oracle | silicon: `bench_layer` dense (`select_us`) | not wired |
| compressed prefill (existing) | contract anchored to 39983a5e7 | codegen | CPU | silicon | yes |
| capture gate | - | - | CPU plumbing | silicon + compute-sanitizer | - |

## 8. Env knobs and handoffs (nothing set by this lane; presets are not mine)

| name | default | meaning |
|---|---|---|
| `SUFFIX_SM120=1`, `SUFFIX_SM120_DSA_INDEXER=1` | off | **required**: boot5 had the shim inert; without them stock DeepGEMM runs |
| `SUFFIX_HELION_DSA_INDEXER` | `oxide` | set `helion` (V4.1's 32/64/128-token packed pages are Helion-only; every shipped bucket must exist, else startup error) or `auto` while configs are partial |
| `SUFFIX_SM120_DSA_INDEXER_GROUP` | vLLM config: DSpark = 1+k | 6 explicit is safe |
| `SUFFIX_SM120_DSA_INDEXER_CAND=1` | off | fused candidate logits on layers 24-36 (needs `CANDK` configs) |
| `SUFFIX_SM120_DSA_INDEXER_CANDSEL=1` | off | valid-block selection on layer 20 |
| `SUFFIX_SM120_DSA_INDEXER_STRICT=1` | off | K5: any Triton fallback is a startup error (recommended with the above) |
| `SUFFIX_DSA_INDEXER_CTX` | unset = longest bucket | pin the row-group context bucket (positions per request) |

Handoffs: (1) preset `dsv41-sm120` (main session): the 5 keys above + the config pull of `configs/helion/dsa_indexer_{group,cand,decode,prefill}/<platform>.json`
after the lab; (2) run the lab then the gate on a dev pool before any A/B; (3) `--attention-config.indexer_tp_shard_min_rows`
(dossier item 3) is orthogonal and arg-only; (4) the per-row decode body and the select body are not graph-reviewed beyond
this lane; (5) with the gates off nothing changes for GLM or for the oxide backend (a subprocess test pins that the oxide backend
never imports the Helion port; shipped GLM config keys are untouched).

## 9. Risks / what silicon must verify first

1. **Triton specialisation guard false alarm**: the key is deliberately finer than Triton's (shape classes of every tensor).
   In vLLM's flow capture is preceded by an identical eager warm-up so it cannot fire; if the first boot shows it, the gate
   prints the argument that differs (set it right in `spec_key`, do not disable).
2. **L1 reuse of the row group** (the whole 6x claim): compare `groupS_us` with `oxide_us` and DeepGEMM; if L1 does not
   deliver, the fallback design is an explicit shared tile (3-D batched dot) - not built.
3. **Helion int32/int64 index class** at the real cache size (> 2^31 bytes): `_warm` keys on the storage span; the gate runs
   with the pool's real cache only in the pod (the lab cache is small). Run the gate on the pool.
4. `hl.load(extra_mask)` + `min()` on SymInts + rolled `tl.range` nesting compile in Triton source on CPU; ptxas / spills /
   smem are silicon facts. 5. The top-k tie order of persistent_topk differs between runs only at the k-th value (tests accept
   ties, serving A/B decides quality). 6. Dense layers' L2 sharing in stock may already give part of the K/G win: the lab's
   `stock_us` is the truth, the byte model an upper bound.

## 10. Expected effect (estimate, derivation in section 2)

Per 512k request and step: stock indexer 0.85 ms (dossier est.) .. 1.7 ms (byte roofline) -> ours 0.1-0.2 ms => -0.7..-1.5 ms
= 3-7 % of the 20.4 ms c1 step; with eight 512k requests (~45 ms step) -6..-12 ms = 13-27 %; at <= 32k context the indexer is
<= 0.3 ms per step (dossier) and the saving is <= 0.1-0.2 ms (<= 1 %). The candidate-source layer alone is a fixed cost
independent of context (a full-width read + `torch.topk` of 65536 scores per row): at c=32 (192 rows) ~0.4 GB of scores per step
before the top-k - the valid-block selection removes it at every context. Nothing here is a measurement.

## 11. Tests (exact lines, torch 2.14 CPU, Helion 1.5.1; `PYTHONPATH=<worktree> /tmp/lane-c/venv/bin/python -m pytest ... -p no:cacheprovider -q`)

```
tests/test_dsa_indexer.py: 25 passed        tests/test_helion_dsa_indexer.py: 176 passed
tests/test_dsa_candidates.py: 15 passed     tests/test_dsv41_indexer_anchors.py: 8 passed
tests/test_dsa_select.py: 20 passed         tests/test_dsa_indexer_cost.py: 6 passed
tests/test_dsa_capture.py: 11 passed        tests/test_dsa_indexer_gate.py: 7 passed
tests/test_dsa_indexer_lab.py: 4 passed     all nine files together: 272 passed
without Helion installed (import blocked):  244 passed, 28 skipped
```

## 12. Root causes proven in this lane (for `MINED-LESSONS.md`, main session appends)

1. V4.1 decode runs the sparse indexer **inside the FULL CUDA graph**, captured once with `max_seq_len = max_model_len`
   (`model_states/default.py:198`); any host decision from the step's context (bucket, fuse-or-not) is frozen at capture and
   meaningless. Decide from static facts only. vLLM warms every captured size eagerly (`forward_fn(NONE)`) first.
2. Helion `hl.static_range(G)` unrolls the row loop and multiplies the per-memory-op config lists by G (eviction 42 / indexing 48
   for G = 6 vs 8 / 9 rolled): unusable for LFBO. Use a rolled `hl.grid(G)`.
3. Helion hands Triton runtime ints (sizes, strides, page parameters) that Triton specialises on (== 1 / % 16) plus pointer
   alignment: a new batch size or table width can JIT mid-capture. `cap.launch` records classes and refuses unseen ones in a capture.
4. A k-th-largest lower bound from a warm subset is only valid for DISTINCT positions (a repeated member lifts the multiset's
   k-th largest above the truth); and a warm set with one stale member collapses a min-bound to the bulk of the row -> take the
   k-th largest of warm + evenly spaced sample.
5. On SM120 the stock consumer layers pay a rows x max_model_len x 4 B mask rewrite and the source layer a full-width score read +
   `torch.topk(2048)` over 65536 block scores **at every context**, including 8k.

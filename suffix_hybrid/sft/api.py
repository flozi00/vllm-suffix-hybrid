# SPDX-License-Identifier: Apache-2.0
"""SFT HTTP API (vLLM endpoint plugin ``suffix_sft``) + Hub checkpoints.

Routes (API-key protected like every /v1 route when --api-key is set; with
SUFFIX_SFT_TOKEN set, train / load / diag also need the header
``X-SFT-Token: <token>``, so a pool serving without --api-key (prod) does not
let every client retrain it; a server with neither refuses to enable SFT):
  POST /v1/sft/train   {"examples": [{"messages": [...]} | {"prompt": str,
                        "completion": str}], "steps": 1, "batch_size": 1,
                        "learning_rate": 1e-4, "weight_decay": 0.0,
                        "chat_template_kwargs": {}, "seed": 0}
                        -> trains, saves, publishes, then serves the result
                        OR {"dataset": "<org>/<name>", "file": "train.jsonl",
                        "heldout": "heldout.jsonl", "revision": null,
                        "epochs": 1, ...} -> the same, over a private HF
                        dataset repo, encoded per batch (memory O(batch)), so
                        a whole-dataset epoch is ONE request instead of a
                        client-side chunk loop. "steps" wins over "epochs".
                        OR {"resume": "<run id>"} -> continues an unfinished
                        run from its last snapshot (see GET status.resumable)
  POST /v1/sft/load    {"run_id": "<published run>" | "<run>/step-<n>" |
                        "base"} -> serves it and moves latest.json to it
                        (durable rollback; a mid-run snapshot is loadable so
                        a checkpoint can be evaluated before the run ends)
  GET  /v1/sft/status  also reports `resumable`: unfinished runs with a
                        snapshot, i.e. what {"resume": id} can continue
Optional "model" and "prompt" fields are accepted and ignored: the pool
gateway (EPP) routes on them and rejects POST bodies without "prompt".
"wait": false returns 202 at once and the run continues (poll status);
a waiting client of a long run hits the gateway's route timeout, but the
run still finishes and publishes.

Loss: chat examples train on the FINAL assistant message only (prompt =
history rendered with the generation prompt, exactly what serving renders;
earlier assistant turns are context), raw examples on the completion (+EOS).

Checkpoint contract (parakeet-server OnlineTrainer): private model repo
SUFFIX_SFT_REPO_ID; each run commits runs/<id>/{adapter-tp<r>-of-<n>.
safetensors, metadata.json} + latest.json atomically against the parent
commit; only a published run reaches serving; startup resolves the Hub's
latest.json (reusing the local copy when it matches) and loads it before
the server accepts requests. Repo missing / no latest.json -> base model.
A restore that fails (network, fingerprint mismatch) leaves the base
model serving, status ``restore_failed``, and training refused until
/v1/sft/load picks a checkpoint: never silently fork from base.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from suffix_hybrid.sft import Config

MARK = "[suffix sft]"
PARITY_WINDOW = 1024
_S = None   # the plugin's app state (one API server process)


def _log(msg):
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def _http(code, detail):
    """vLLM's HTTPException handler only takes string details."""
    from fastapi import HTTPException
    return HTTPException(code, detail if isinstance(detail, str) else
                         json.dumps(detail, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
PARITY_SE_CAP = 0.35   # nats a noisy short example may add to the parity limit (a broken twin also inflates SE)
OPEN_THINK = "<think>"


def encode(tok, ex, template_kwargs=None, chat_template=None):
    """Example -> (ids, mask): mask 1 on the tokens the loss trains."""
    if not isinstance(ex, dict):
        raise _http(400, "each example must be an object")
    if "messages" in ex:
        msgs = ex["messages"]
        if (not isinstance(msgs, list) or len(msgs) < 2 or not all(isinstance(m, dict) for m in msgs)
                or msgs[-1].get("role") != "assistant"):
            raise _http(400, "messages: >= 2 messages, the last one from the assistant")
        kw = dict(template_kwargs or {}, tokenize=False)
        if chat_template:
            kw["chat_template"] = chat_template
        if ex.get("tools"):
            kw["tools"] = ex["tools"]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, **kw)
        full = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
        tail = None
        if full.startswith(prompt):
            tail = full[len(prompt):]
        elif prompt.endswith(OPEN_THINK) and full.startswith(prompt[:-len(OPEN_THINK)]):
            # LFM2-style: the generation prompt opens <think> for the model, a
            # final assistant message without reasoning renders without it.
            # Served, the model continues from <think>: the example answers
            # with an empty reasoning block.
            tail = "</think>" + full[len(prompt) - len(OPEN_THINK):]
        if tail is None:
            raise _http(400, "the chat template renders the history differently once the final "
                             "assistant message is appended; cannot split prompt / completion")
        p = tok.encode(prompt, add_special_tokens=False)
        c = tok.encode(tail, add_special_tokens=False)
    elif isinstance(ex.get("prompt"), str) and isinstance(ex.get("completion"), str):
        p = tok.encode(ex["prompt"])
        c = tok.encode(ex["completion"], add_special_tokens=False)
        if tok.eos_token_id is not None:
            c = c + [tok.eos_token_id]
    else:
        raise _http(400, "an example is {'messages': [...]} or {'prompt': str, 'completion': str}")
    if not c or not p:
        raise _http(400, "empty prompt or completion")
    return list(p) + list(c), [0] * len(p) + [1] * len(c)


def pack(examples, max_tokens):
    """Greedy, order-preserving packing into micro-batches <= max_tokens."""
    out, cur = [], []
    for ids, mask in examples:
        if cur and sum(len(i) for i, _ in cur) + len(ids) > max_tokens:
            out.append(cur)
            cur = []
        cur.append((ids, mask))
    if cur:
        out.append(cur)
    return [dict(ids=[t for i, _ in mb for t in i], mask=[t for _, m in mb for t in m],
                 seg_lens=[len(i) for i, _ in mb]) for mb in out]


def schedule(n, steps, batch_size, seed):
    """Example indices per optimizer step: shuffled epochs, cycled."""
    rng, order, out = random.Random(seed), [], []
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            if not order:
                order = list(range(n))
                rng.shuffle(order)
            batch.append(order.pop())
        out.append(batch)
    return out


# ---------------------------------------------------------------------------
# dataset (private HF dataset repo, JSONL, one example per line)
# ---------------------------------------------------------------------------
DATASET_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
FILE_NAME = re.compile(r"^[\w.-]+\.jsonl$")


class Dataset:
    """Examples from a private HF dataset repo, read lazily.

    The inline ``examples`` path is capped at SUFFIX_SFT_MAX_EXAMPLES because
    every example is encoded up front and the ids stay in the API server's
    memory. A dataset is read as raw JSONL and encoded per batch, so its cap is
    SUFFIX_SFT_MAX_DATASET and the memory is O(batch) -- which is what makes a
    whole-dataset epoch one request instead of a client-side chunk loop.

    The repo must be private: it is training data. ``heldout.jsonl`` (optional,
    same shapes) is scored at every snapshot so each checkpoint carries its own
    held-out NLL.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _file(self, req, key, default):
        name = req.get(key, default)
        if name in (None, ""):
            return None
        if not isinstance(name, str) or not FILE_NAME.match(name):
            raise _http(400, f"{key}: a plain .jsonl file name inside the dataset repo")
        return name

    def spec(self, req):
        """Validated dataset spec from a train body."""
        repo = req.get("dataset")
        if not isinstance(repo, str) or not DATASET_NAME.match(repo) or ".." in repo:
            raise _http(400, "dataset: '<org>/<name>' of a private HF dataset repo")
        rev = req.get("revision")
        if rev is not None and (not isinstance(rev, str) or not rev.strip()):
            raise _http(400, "revision: a branch, tag or commit sha")
        return dict(dataset=repo, file=self._file(req, "file", "train.jsonl"),
                    heldout=self._file(req, "heldout", "heldout.jsonl"), revision=rev)

    def _download(self, repo, name, revision):
        from huggingface_hub import HfApi, get_token, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        if not get_token():
            raise _http(503, "a dataset needs HF_TOKEN (or a Hugging Face login)")
        try:
            info = HfApi().dataset_info(repo, revision=revision)
        except RepositoryNotFoundError:
            raise _http(404, f"dataset {repo} not found or not accessible") from None
        if not info.private:
            raise _http(409, f"dataset {repo} is public; training data must be private")
        try:
            return Path(hf_hub_download(repo, name, repo_type="dataset", revision=info.sha))
        except EntryNotFoundError:
            raise _http(404, f"{repo} has no {name}") from None

    def read(self, spec):
        """(examples, heldout) as raw dicts -- nothing encoded yet."""
        ex = self._jsonl(self._download(spec["dataset"], spec["file"], spec["revision"]))
        held = []
        if spec.get("heldout"):
            try:
                held = self._jsonl(self._download(spec["dataset"], spec["heldout"],
                                                  spec["revision"]))
            except Exception as exc:  # noqa: BLE001 - held-out is a measurement, not a gate
                _log(f"held-out set unavailable ({getattr(exc, 'detail', exc)}); "
                     "snapshots will carry no heldout_nll")
        return ex, held

    @staticmethod
    def _jsonl(path):
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    raise _http(400, f"{path.name} line {i}: not JSON") from None
                if not isinstance(obj, dict):
                    raise _http(400, f"{path.name} line {i}: an example must be an object")
                out.append(obj)
        return out


# ---------------------------------------------------------------------------
# checkpoints (private HF Hub repo; local-only when no repo is configured)
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self, cfg):
        self.repo_id = cfg.repo_id
        self.directory = Path(cfg.dir) / hashlib.sha256((cfg.repo_id or "local").encode()).hexdigest()[:16]

    def _local_pointer(self):
        p = self.directory / "latest.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def _write_local_pointer(self, meta):
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / "latest.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self.directory / "latest.json")

    def latest(self):
        """The published pointer (dict, {"checkpoint": None} = base) or None."""
        if not self.repo_id:
            return self._local_pointer()
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        try:
            info = HfApi().model_info(self.repo_id)
        except RepositoryNotFoundError:
            _log(f"{self.repo_id} not accessible yet; serving the base model")
            return None
        if not info.private:
            raise RuntimeError("SUFFIX_SFT_REPO_ID must be a private repo")
        try:
            path = hf_hub_download(self.repo_id, "latest.json", revision=info.sha)
        except EntryNotFoundError:
            return None
        meta = json.loads(Path(path).read_text())
        meta["_revision"] = info.sha
        return meta

    def fetch(self, meta):
        """Local directory holding the run's shard files."""
        run_dir = self.directory / meta["checkpoint"]
        if all((run_dir / f).is_file() for f in meta["files"]):
            return run_dir
        if not self.repo_id:
            raise FileNotFoundError(f"{run_dir} incomplete and no SUFFIX_SFT_REPO_ID")
        from huggingface_hub import hf_hub_download
        out = None
        for f in meta["files"]:
            out = Path(hf_hub_download(self.repo_id, f"{meta['checkpoint']}/{f}",
                                       revision=meta.get("_revision")))
        return out.parent

    def run_meta(self, run_id):
        """metadata.json of a published (or local) run."""
        local = self.directory / "runs" / run_id / "metadata.json"
        if local.is_file():
            return json.loads(local.read_text())
        if not self.repo_id:
            raise _http(404, f"no local run {run_id}")
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            return json.loads(Path(hf_hub_download(self.repo_id, f"runs/{run_id}/metadata.json"))
                              .read_text())
        except EntryNotFoundError:
            raise _http(404, f"run {run_id} not published in {self.repo_id}") from None

    def private_repo(self):
        """(api, parent sha) of the destination, created private if needed."""
        if not self.repo_id:
            return None, None
        from huggingface_hub import HfApi, get_token
        if not get_token():
            raise _http(503, "SFT checkpoints need HF_TOKEN (or a Hugging Face login)")
        api = HfApi()
        api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        info = api.model_info(self.repo_id)
        if not info.private:  # create_repo does not change an existing repo's visibility
            raise _http(409, "SUFFIX_SFT_REPO_ID is public; use a private model repo")
        return api, info.sha

    def publish(self, api, parent, meta, run_dir=None, message=""):
        """Commit the run files (if any) + latest.json -> commit or None."""
        payload = json.dumps({k: v for k, v in meta.items() if not k.startswith("_")},
                             indent=2).encode()
        commit = None
        if api is not None:
            from huggingface_hub import CommitOperationAdd
            ops = [CommitOperationAdd(path_in_repo="latest.json", path_or_fileobj=payload)]
            if run_dir is not None:
                files = meta["files"] + ["metadata.json"]
                if (run_dir / "progress.json").is_file():
                    files.append("progress.json")   # the resume marker rides along
                ops = [CommitOperationAdd(path_in_repo=f"{meta['checkpoint']}/{f}",
                                          path_or_fileobj=str(run_dir / f))
                       for f in files] + ops
            commit = api.create_commit(repo_id=self.repo_id, repo_type="model",
                                       parent_commit=parent, commit_message=message, operations=ops)
        self._write_local_pointer(meta)
        return commit


# ---------------------------------------------------------------------------
# app state
# ---------------------------------------------------------------------------
class State:
    def __init__(self, engine, cfg, args):
        self.engine, self.cfg = engine, cfg
        self.hub = Hub(cfg)
        self.dataset = Dataset(cfg)
        self.running = None   # the in-flight run / load task
        self.status = {"state": "starting", "last_run": None, "checkpoint": None}
        self.started = 0     # runs / loads that passed validation (rejection detection)
        self.info = None
        self.chat_template = None
        # serving renders with --default-chat-template-kwargs; so must training
        self.template_defaults = dict(getattr(args, "default_chat_template_kwargs", None) or {})
        if getattr(args, "chat_template", None):
            try:
                from vllm.renderers.chat_utils import load_chat_template
                self.chat_template = load_chat_template(args.chat_template)
            except Exception as exc:  # noqa: BLE001
                _log(f"--chat-template not loadable for SFT ({exc}); using the tokenizer's")
        self._tok = None

    async def rpc(self, op, payload=None, check=True):
        res = await self.engine.collective_rpc("suffix_sft", timeout=self.cfg.rpc_timeout_s,
                                               args=(op, payload or {}))
        bad = [r for r in res if r.get("error")]   # any rank failing fails the op
        if check and bad:
            raise _http(507 if any(r.get("oom") for r in bad) else 500,
                        {"message": f"{op} failed: {bad[0]['error']}", "ranks_failed": len(bad)})
        return bad[0] if bad else res[0]

    async def reset_cache(self):
        """No KV / GDN prefix state computed with the old adapter may be
        reused: running requests are preempted and recomputed."""
        if not await self.engine.reset_prefix_cache(reset_running_requests=True):
            raise RuntimeError("prefix cache reset refused by the engine")

    async def tokenizer(self):
        if self._tok is None:
            tok = self.engine.get_tokenizer()
            self._tok = await tok if asyncio.iscoroutine(tok) else tok
        return self._tok

    def busy(self):
        try:
            return self.engine.output_processor.get_num_unfinished_requests() > 0
        except Exception:  # noqa: BLE001 - unknown engine client: assume load
            return True

    async def yield_to_serving(self, ms):
        await asyncio.sleep(ms / 1e3 * (1.0 / self.cfg.duty - 1.0) if self.busy() else 0)

    async def served_nll(self, ids, mask):
        """Per-token NLL of the masked tokens as SERVING computes them
        (prompt_logprobs); None when the engine cannot produce them."""
        try:
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
            final = None
            async for out in self.engine.generate(TokensPrompt(prompt_token_ids=ids), sp,
                                                  f"suffix-sft-parity-{uuid.uuid4().hex}"):
                final = out
            lp = final.prompt_logprobs
            return [-lp[i][ids[i]].logprob for i in range(1, len(ids)) if mask[i]]
        except Exception as exc:  # noqa: BLE001 - parity is a measurement
            _log(f"served NLL unavailable: {type(exc).__name__}: {exc}")
            return None

    async def parity(self, ids, mask, window=PARITY_WINDOW):
        """Served vs training-forward NLL on one example: mean delta (the
        guard) + per-token |delta| quantiles and where the worst token sits.
        Measured on <= `window` tokens: past ~2k tokens serving's own NLL
        drifts up (NVFP4 KV / long prefill: measured 1.22 vs 1.10 on
        qwen-fable at 3.4k tokens, while <= 1k agrees to ~0.02), which says
        nothing about the training forward."""
        if len(ids) > window:
            ids, mask = (ids[:window], mask[:window]) if any(mask[1:window]) else (ids[-window:], mask[-window:])
        served = await self.served_nll(ids, mask)
        r = await self.rpc("eval", dict(ids=ids, mask=mask, seg_lens=[len(ids)], per_token=True))
        trained = r["loss_sum"] / max(1, r["tokens"])
        out = dict(served_nll=None, trained_nll=round(trained, 5), delta=None, tokens=r["tokens"])
        if served is not None and len(served) == len(r["nll"]):
            d = sorted(abs(a - b) for a, b in zip(served, r["nll"]))
            worst = max(range(len(d)), key=lambda i: abs(served[i] - r["nll"][i]))
            pos = [i for i in range(1, len(ids)) if mask[i]][worst]
            mean = sum(served) / len(served)
            diff = [a - b for a, b in zip(served, r["nll"])]
            mu = sum(diff) / len(diff)
            se = (sum((x - mu) ** 2 for x in diff) / len(diff)) ** 0.5 / len(diff) ** 0.5
            out.update(served_nll=round(mean, 5), delta=round(abs(mean - trained), 5), se=round(se, 5),
                       tok_p50=round(d[len(d) // 2], 4), tok_p99=round(d[int(len(d) * 0.99)], 4),
                       tok_max=round(d[-1], 4), tok_max_pos=pos, len=len(ids),
                       tok_max_served=round(served[worst], 4), tok_max_trained=r["nll"][worst])
        return out

    # -- startup restore ------------------------------------------------------
    async def restore(self):
        for attempt in range(3):
            try:
                meta = await asyncio.to_thread(self.hub.latest)
                if meta and meta.get("checkpoint"):
                    run_dir = await asyncio.to_thread(self.hub.fetch, meta)
                    await self.rpc("load", {"dir": str(run_dir)})
                    _log(f"serving checkpoint {meta['checkpoint']} from "
                         f"{self.hub.repo_id or self.hub.directory}")
                self.status.update(state="idle", checkpoint=_public(meta))
                return
            except Exception as exc:  # noqa: BLE001
                err = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
                _log(f"restore attempt {attempt + 1} failed: {err}")
                if attempt < 2:
                    await asyncio.sleep(10)
        self.status.update(state="restore_failed", error=str(err))
        _log("RESTORE FAILED: serving the BASE model; training refused until POST /v1/sft/load")

    # -- progress marker (resume) ---------------------------------------------
    def _progress_path(self, run_id):
        return self.hub.directory / "runs" / run_id / "progress.json"

    def _write_progress(self, run_id, prog):
        """Local, atomic, after every step: a process restart loses no steps."""
        p = self._progress_path(run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(prog, indent=2))
        tmp.replace(p)

    def _read_progress(self, run_id):
        """The run's marker: local first (freshest), then the Hub (survives a
        pod re-creation, written with each snapshot)."""
        p = self._progress_path(run_id)
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except ValueError:
                pass
        if not self.hub.repo_id:
            return None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            return json.loads(Path(hf_hub_download(
                self.hub.repo_id, f"runs/{run_id}/progress.json")).read_text())
        except (EntryNotFoundError, OSError, ValueError):
            return None

    def resumable(self):
        """Unfinished runs that have a snapshot: what {"resume": id} continues."""
        out = []
        for p in sorted((self.hub.directory / "runs").glob("*/progress.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if d.get("done") or not d.get("checkpoints"):
                continue
            out.append({k: d.get(k) for k in ("run_id", "step", "steps", "dataset",
                                              "loss_first", "loss")})
        return out

    async def _resume_req(self, run_id):
        """Rebuild a train request from a run's marker and load the adapter at
        its last snapshot. Returns (request, resume state).

        The schedule is a pure function of (n, steps, batch_size, seed), so
        step k's batch is reproducible: resume skips the steps already taken.
        The optimizer state is NOT in a snapshot -- AdamW restarts from zero,
        which costs a few steps of biased moments, not correctness.
        """
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise _http(400, "resume: a run id from GET /v1/sft/status")
        prog = self._read_progress(run_id)
        if prog is None:
            raise _http(404, f"no progress marker for run {run_id}")
        if prog.get("done"):
            raise _http(409, f"run {run_id} finished; nothing to resume")
        ck = [x for x in prog.get("checkpoints") or []
              if isinstance(x, dict) and x.get("step", 0) <= prog.get("step", 0)]
        if not ck:
            raise _http(409, f"run {run_id} has no snapshot at or before step "
                             f"{prog.get('step')}; start a new run")
        last = max(ck, key=lambda x: x["step"])
        meta = await asyncio.to_thread(self.hub.run_meta, last["path"][len("runs/"):])
        if meta.get("fingerprint") != (self.info or {}).get("fingerprint"):
            raise _http(409, f"run {run_id} was trained for a different model / TP / "
                             "adapter config")
        run_dir = await asyncio.to_thread(self.hub.fetch, meta)
        req = dict(dataset=prog["dataset"], file=prog.get("file"), heldout=prog.get("heldout"),
                   revision=prog.get("revision"), steps=prog["steps"],
                   batch_size=prog["batch_size"], learning_rate=prog["learning_rate"],
                   weight_decay=prog.get("weight_decay", 0.0), seed=prog.get("seed", 0))
        state: dict[str, Any] = dict(
            run_id=run_id, step=last["step"], best=last["loss"],
            checkpoints=list(prog.get("checkpoints") or []),
            loss_first=prog.get("loss_first"), dir=str(run_dir))
        return req, state

    async def heldout_nll(self, held):
        """Mean per-token NLL of the held-out examples under the CURRENT
        adapter, measured with the training forward (no serving involved)."""
        if not held:
            return None
        tot = n = 0.0
        for ids, mask in held:
            r = await self.rpc("eval", dict(ids=ids, mask=mask, seg_lens=[len(ids)]))
            tot += r["loss_sum"]
            n += r["tokens"]
        return round(tot / max(1.0, n), 5)

    # -- train --------------------------------------------------------------
    async def snapshot(self, api, parent, run_id, step, loss, heldout_nll=None, progress=None):
        """Mid-run checkpoint: this run's adapters at `step`, uploaded under
        runs/<run_id>/step-<n>/ WITHOUT touching latest.json, so a partial run
        can never reach serving. Returns (record, new parent sha).

        `progress` (the run's marker) is updated with this snapshot and
        committed alongside it, so a pod re-creation can still resume.
        """
        d = self.hub.directory / "runs" / run_id / f"step-{step}"
        d.mkdir(parents=True, exist_ok=True)
        r = await self.rpc("save", {"dir": str(d)})
        info = self.info or {}
        files = [f"adapter-tp{i}-of-{info.get('tp', 1)}.safetensors"
                 for i in range(info.get("tp", 1))]
        meta = dict(run_id=run_id, checkpoint=f"runs/{run_id}/step-{step}", files=files,
                    fingerprint=r["fingerprint"], step=step, loss=round(loss, 5),
                    heldout_nll=heldout_nll,
                    created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        rec = dict(step=step, loss=round(loss, 5), path=meta["checkpoint"],
                   heldout_nll=heldout_nll)
        if progress is not None:
            progress.setdefault("checkpoints", []).append(rec)
            progress.update(step=step, best=round(loss, 5), loss=round(loss, 5))
            self._write_progress(run_id, progress)
        if api is not None:
            from huggingface_hub import CommitOperationAdd
            ops = [CommitOperationAdd(path_in_repo=f"{meta['checkpoint']}/{f}",
                                      path_or_fileobj=str(d / f))
                   for f in files + ["metadata.json"]]
            if progress is not None:
                ops.append(CommitOperationAdd(
                    path_in_repo=f"runs/{run_id}/progress.json",
                    path_or_fileobj=str(self._progress_path(run_id))))
            commit = await asyncio.to_thread(
                api.create_commit, repo_id=self.hub.repo_id, repo_type="model",
                parent_commit=parent, operations=ops,
                commit_message=f"SFT {run_id} step {step}: loss {loss:.5f}"
                               + (f", heldout {heldout_nll:.5f}" if heldout_nll else ""))
            parent = getattr(commit, "oid", None) or parent
        _log(f"[suffix sft] checkpoint step {step} loss {loss:.5f}"
             + (f" heldout {heldout_nll:.5f}" if heldout_nll else "")
             + f" -> {meta['checkpoint']}")
        return rec, parent

    async def train(self, req):
        c = self.cfg
        res: dict[str, Any] | None = None
        if req.get("resume") is not None:
            req, res = await self._resume_req(req["resume"])
        steps, bs = int(req.get("steps", 1)), int(req.get("batch_size", 1))
        lr, wd = float(req.get("learning_rate", 1e-4)), float(req.get("weight_decay", 0.0))
        if not math.isfinite(lr) or not 0 < lr <= 1 or not math.isfinite(wd) or wd < 0:
            raise _http(400, "learning_rate must be in (0, 1], weight_decay >= 0")
        if not 1 <= bs <= c.max_examples:
            raise _http(400, f"batch_size must be between 1 and {c.max_examples}")
        tok = await self.tokenizer()
        kw = dict(self.template_defaults, **(req.get("chat_template_kwargs") or {}))
        spec, held_raw = None, []
        if req.get("dataset") is not None:
            spec = self.dataset.spec(req)
            data, held_raw = await asyncio.to_thread(self.dataset.read, spec)
            if not 1 <= len(data) <= c.max_dataset:
                raise _http(400, f"dataset: 1..{c.max_dataset} examples "
                                 f"(SUFFIX_SFT_MAX_DATASET), got {len(data)}")
            if req.get("epochs") is not None and req.get("steps") is None:
                ep = int(req["epochs"])
                if not 1 <= ep <= 100:
                    raise _http(400, "epochs must be between 1 and 100")
                steps = -(-len(data) // bs) * ep
        else:
            examples = req.get("examples")
            if not isinstance(examples, list) or not 1 <= len(examples) <= c.max_examples:
                raise _http(400, f"examples: 1..{c.max_examples} items, or a dataset")
            data = examples
        if not 1 <= steps <= c.max_steps:
            raise _http(400, f"steps must be between 1 and {c.max_steps}"
                             + (f" (one epoch of {len(data)} examples at batch_size {bs} is "
                                f"{-(-len(data) // bs)}; raise SUFFIX_SFT_MAX_STEPS)"
                                if spec else ""))
        if spec is None:
            # inline: encode everything up front, so a bad example fails before
            # the run starts (the ids are bounded by max_examples)
            data_enc = [encode(tok, ex, kw, self.chat_template) for ex in data]
            for i, (ids, _) in enumerate(data_enc):
                if len(ids) > c.max_tokens:
                    raise _http(400, f"example {i} has {len(ids)} tokens > SUFFIX_SFT_MAX_TOKENS="
                                     f"{c.max_tokens}")
            enc = data_enc.__getitem__
        else:
            # dataset: encode per batch (memory O(batch)); preflight the first
            # few so a systematically too-long dataset still fails up front
            cache = {}

            def enc(i):
                hit = cache.get(i)
                if hit is None:
                    hit = encode(tok, data[i], kw, self.chat_template)
                    if len(hit[0]) > c.max_tokens:
                        raise _http(400, f"example {i} has {len(hit[0])} tokens > "
                                         f"SUFFIX_SFT_MAX_TOKENS={c.max_tokens}")
                    if len(cache) >= 64:
                        cache.pop(next(iter(cache)))
                    cache[i] = hit
                return hit

            for i in range(min(len(data), 8)):
                enc(i)
        held = []
        for ex in held_raw[:c.heldout_n]:
            try:
                ids, mask = encode(tok, ex, kw, self.chat_template)
            except Exception:  # noqa: BLE001 - a bad held-out row must not fail the run
                continue
            if len(ids) <= c.max_tokens:
                held.append((ids, mask))
        if self.status["state"] == "restore_failed":
            raise _http(409, "checkpoint restore failed at startup; POST /v1/sft/load first")

        run_id = res["run_id"] if res else uuid.uuid4().hex
        stage, commit, t0 = "initialization", None, time.monotonic()
        run_dir = self.hub.directory / "runs" / run_id
        self.status.pop("rejected", None)
        self.started += 1
        start = res["step"] if res else 0
        self.status.update(state="training", run_id=run_id, step=start, steps=steps, loss=None,
                           resumed_from=start if res else None)
        # the resume marker: written locally after every step, committed with
        # every snapshot (so it survives a pod re-creation, not just a restart)
        prog = dict(run_id=run_id, dataset=spec["dataset"] if spec else None,
                    file=spec.get("file") if spec else None,
                    heldout=spec.get("heldout") if spec else None,
                    revision=spec.get("revision") if spec else None,
                    steps=steps, batch_size=bs, learning_rate=lr, weight_decay=wd,
                    seed=int(req.get("seed", 0)), examples=len(data),
                    epochs=req.get("epochs"), step=start, done=False,
                    checkpoints=list(res["checkpoints"]) if res else [],
                    loss_first=res.get("loss_first") if res else None)
        try:
            api, parent = await asyncio.to_thread(self.hub.private_repo)
            if res:
                # the adapter at the last snapshot, then continue from there
                await self.rpc("load", {"dir": res["dir"]})
                _log(f"resuming run {run_id} at step {start}/{steps} from {res['dir']}")
            pre = await self.parity(*enc(0))
            # noise-aware: a numerically sensitive model (LFM2 FP8: per-token NLL moves by
            # ~1 nat between two correct bf16 paths) has a large standard error on a short
            # example; a real forward bug moves the mean by far more than 3 SE
            if pre["delta"] is not None and pre["delta"] > c.parity_nats + min(
                    3 * pre.get("se", 0.0), PARITY_SE_CAP):
                raise _http(409, {"message": "training forward diverges from serving; refusing "
                                             "(SUFFIX_SFT_PARITY_NATS + up to 3 standard errors)", "parity": pre})
            stage = "training"
            mem = dict(train_peak_gib=0.0, fwd_peak_gib=0.0, min_free_gib=None, host_pinned_gib=None,
                       max_micro_ms=0.0, fwd_ms=0.0, bwd_ms=0.0, tokens=0)
            kv = await self.rpc("kv_free")
            if kv.get("ok"):
                mem["kv_freed_gib"] = round(kv.get("free_gib", 0.0), 2)
            await self.rpc("begin", dict(lr=lr, weight_decay=wd))
            losses, checked = [], False
            first = res.get("loss_first") if res else None
            best = res["best"] if res else None
            ckpts = list(res["checkpoints"]) if res else []
            for step, idx in enumerate(schedule(len(data), steps, bs, int(req.get("seed", 0)))):
                if step < start:      # already taken before the interruption
                    continue
                batch = [enc(i) for i in idx]
                norm = sum(sum(m[1:]) for _, m in batch)
                loss_sum = 0.0
                for mb in pack(batch, c.max_tokens):
                    r = await self.rpc("micro", dict(mb, norm=norm, check=not checked))
                    checked = True
                    loss_sum += r["loss_sum"]
                    mem["max_micro_ms"] = max(mem["max_micro_ms"], r["ms"])
                    for k in ("fwd_ms", "bwd_ms", "tokens"):
                        mem[k] += r.get(k, 0)
                    if "peak_gib" in r:
                        mem["train_peak_gib"] = max(mem["train_peak_gib"], r["peak_gib"])
                        mem["fwd_peak_gib"] = max(float(mem["fwd_peak_gib"] or 0.0),
                                                  float(r.get("fwd_peak_gib") or 0.0))
                        mem["min_free_gib"] = min(mem["min_free_gib"] or 1e9, r["free_gib"])
                        mem["host_pinned_gib"] = r.get("host_pinned_gib")
                    await self.yield_to_serving(r["ms"])
                loss = loss_sum / norm
                if not math.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                r = await self.rpc("step")
                first = loss if first is None else first
                losses.append(loss)
                self.status.update(step=step + 1, loss=round(loss, 5), grad_norm=r["grad_norm"],
                                   memory=mem)
                prog.update(step=step + 1, loss=round(loss, 5), best=best, loss_first=first)
                self._write_progress(run_id, prog)
                # A checkpoint every time the loss beats the best so far by
                # ckpt_drop (relative, default 10%). Mid-run checkpoints go to
                # runs/<run_id>/step-<n>/ and never touch latest.json, so a
                # partial run can never reach serving. Each one carries the
                # held-out NLL of the adapter it saved.
                if c.ckpt_drop > 0 and (best is None or loss <= best * (1.0 - c.ckpt_drop)):
                    best = loss
                    hn = await self.heldout_nll(held)
                    rec, parent = await self.snapshot(api, parent, run_id, step + 1, loss,
                                                      heldout_nll=hn, progress=prog)
                    ckpts.append(rec)
                    self.status.update(checkpoints=ckpts, heldout_nll=hn)
                await self.yield_to_serving(r["ms"])
            stage = "checkpoint"
            run_dir.mkdir(parents=True, exist_ok=True)   # a mid-run snapshot may have made it
            r = await self.rpc("save", {"dir": str(run_dir)})
            info = self.info or {}
            files = [f"adapter-tp{i}-of-{info.get('tp', 1)}.safetensors" for i in range(info.get("tp", 1))]
            final_loss = losses[-1] if losses else prog.get("loss")
            hn = await self.heldout_nll(held)
            meta = dict(run_id=run_id, checkpoint=f"runs/{run_id}", files=files,
                        fingerprint=r["fingerprint"], model=info.get("model"), rank=info.get("rank"),
                        alpha=info.get("alpha"), targets=info.get("targets"), tp=info.get("tp"),
                        parent=(self.status.get("checkpoint") or {}).get("run_id"),
                        examples=len(data), steps=steps, batch_size=bs, learning_rate=lr,
                        weight_decay=wd,
                        loss_first=round(first, 5) if first is not None else None,
                        loss=round(final_loss, 5) if final_loss is not None else None,
                        heldout_nll=hn, dataset=spec, epochs=req.get("epochs"),
                        resumed_from=start if res else None,
                        checkpoints=ckpts,
                        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            prog.update(done=True, step=steps, loss=meta["loss"], heldout_nll=hn)
            self._write_progress(run_id, prog)
            stage = "upload"
            commit = await asyncio.to_thread(
                self.hub.publish, api, parent, meta, run_dir,
                f"SFT {run_id}: {steps} steps, {len(data)} examples")
            stage = "serve"
            await self.rpc("kv_restore")
            await self.rpc("commit")
            await self.reset_cache()
            post = await self.parity(*enc(0))
            result = dict(meta, parity_before=pre, parity_after=post, served=True, memory=mem,
                          commit_sha=getattr(commit, "oid", None),
                          commit_url=getattr(commit, "commit_url", None),
                          training_time=round(time.monotonic() - t0, 3))
            self.status.update(state="idle", last_run=result, checkpoint=_public(meta))
            self.status.pop("error", None)
            return result
        except BaseException as exc:
            await self.rpc("kv_restore", check=False)
            await self.rpc("abort", check=False)
            if stage == "serve":   # partial commit: keep every TP rank on the same weights
                await self.rpc("reset", check=False)
            saved = (run_dir / "metadata.json").is_file()
            self.status.update(state="failed" if stage != "serve" else "serve_failed_base", stage=stage,
                               error=str(getattr(exc, "detail", exc)))
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
            code = getattr(exc, "status_code", 502 if stage == "upload" else 500)
            raise _http(code, {
                "message": f"SFT run failed during {stage}", "run_id": run_id, "stage": stage,
                "error": getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}",
                "checkpoint_saved": saved, "local_checkpoint": str(run_dir) if saved else None,
                "checkpoint_published": commit is not None,
                "commit_sha": getattr(commit, "oid", None)}) from exc

    # -- load ---------------------------------------------------------------
    async def load(self, run_id):
        if not isinstance(run_id, str) or not run_id or ".." in run_id:
            raise _http(400, "run_id: a published run id, '<run>/step-<n>', or 'base'")
        if "/" in run_id and not re.fullmatch(r"[0-9a-f]{32}/step-\d+", run_id):
            raise _http(400, "run_id: a published run id, '<run>/step-<n>', or 'base'")
        keep = {k: self.status[k] for k in ("state", "run_id", "error") if k in self.status}
        self.status.pop("rejected", None)
        self.started += 1
        self.status.update(state="loading", run_id=run_id)
        stage = "prepare"     # nothing served has been touched yet
        try:
            api, parent = await asyncio.to_thread(self.hub.private_repo)
            if run_id == "base":
                meta = {"checkpoint": None}
            else:
                meta = await asyncio.to_thread(self.hub.run_meta, run_id)
                if meta.get("fingerprint") != (self.info or {}).get("fingerprint"):
                    raise _http(409, f"run {run_id} was trained for a different model / TP / "
                                     "adapter config")
                run_dir = await asyncio.to_thread(self.hub.fetch, meta)
            stage = "apply"
            if run_id == "base":
                await self.rpc("reset")
            else:
                await self.rpc("load", {"dir": str(run_dir)})
            commit = await asyncio.to_thread(self.hub.publish, api, parent, meta, None,
                                             f"SFT serve {run_id}")
            await self.reset_cache()
            self.status.update(state="idle", checkpoint=_public(meta))
            self.status.pop("error", None)
            return dict(serving=run_id, commit_sha=getattr(commit, "oid", None))
        except BaseException as exc:
            if stage == "prepare":   # unknown / foreign / unfetchable run: keep serving what we serve
                for k in ("state", "run_id", "error"):    # exactly the status from before the attempt
                    if k in keep:
                        self.status[k] = keep[k]
                    else:
                        self.status.pop(k, None)
                self.status["rejected"] = dict(code=getattr(exc, "status_code", 500),
                                               error=str(getattr(exc, "detail", exc)))
                raise
            await self.rpc("reset", check=False)   # never leave TP ranks on different adapters
            self.status.update(state="load_failed", checkpoint={"checkpoint": None},
                               error=str(getattr(exc, "detail", exc)))
            raise


def _public(meta):
    return None if not meta else {k: v for k, v in meta.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# endpoint plugin
# ---------------------------------------------------------------------------
TOKEN_HEADER = "x-sft-token"


def _authorized(request):
    """Constant-time check of X-SFT-Token against SUFFIX_SFT_TOKEN (unset: the
    --api-key middleware is the only gate; init_state refused a pod with neither)."""
    import hmac
    want = os.environ.get("SUFFIX_SFT_TOKEN", "")
    if not want:
        return
    got = request.headers.get(TOKEN_HEADER, "")
    if not hmac.compare_digest(got.encode(), want.encode()):
        raise _http(401, "missing or wrong X-SFT-Token")


async def _exclusive(request, run):
    """One run / load at a time per server; an accepted one finishes (and
    publishes) even if the client disconnects."""
    if _S is None:
        raise _http(503, "SFT is not available on this server (see [suffix sft] log lines)")
    _authorized(request)
    try:
        body = await request.json()
    except ValueError:
        raise _http(400, "body must be JSON") from None
    if not isinstance(body, dict):
        raise _http(400, "body must be a JSON object")
    if _S.running is not None and not _S.running.done():
        raise _http(409, "an SFT run or load is already in progress")
    async def guarded():
        n0 = _S.started
        try:
            return await run(body)
        except BaseException as exc:
            # a request rejected before it started a run / load (validation,
            # tokenization): with wait=false nobody reads this exception
            if _S.started == n0:
                _S.status["rejected"] = dict(code=getattr(exc, "status_code", 500),
                                             error=str(getattr(exc, "detail", exc)))
            raise
    _S.running = asyncio.ensure_future(guarded())
    _S.running.add_done_callback(lambda t: t.cancelled() or t.exception())
    if body.get("wait", True) is False:   # long runs vs the gateway's route timeout
        from fastapi.responses import JSONResponse
        return JSONResponse({"accepted": True, "poll": "GET /v1/sft/status"}, status_code=202)
    return await asyncio.shield(_S.running)


async def train_route(request):
    return await _exclusive(request, lambda body: _S.train(body))


async def load_route(request):
    return await _exclusive(request, lambda body: _S.load(body.get("run_id")))


async def status_route():
    if _S is None:
        raise _http(503, "SFT is not available on this server")
    return dict(_S.status, sft=_S.info, repo=_S.hub.repo_id or None,
                resumable=_S.resumable())


async def diag_route(request):
    """Dev only (SUFFIX_SFT_DIAG=1): served vs training-forward per-token NLL of
    one text under forward variants."""
    if _S is None:
        raise _http(503, "SFT is not available on this server")
    _authorized(request)
    if _S.running is not None and not _S.running.done():
        raise _http(409, "an SFT run or load is in progress")
    body = await request.json()
    tok = await _S.tokenizer()
    ids = tok.encode(body["text"], add_special_tokens=False)[:min(int(body.get("max_tokens", 512)), _S.cfg.max_tokens)]
    mask = [0] + [1] * (len(ids) - 1)
    served = await _S.served_nll(ids, mask)
    r = await _S.rpc("diag", dict(ids=ids, mask=mask, seg_lens=[len(ids)], variants=body["variants"]))
    return dict(served=served, **{k: v for k, v in r.items() if k != "ms"})


class SftEndpointPlugin:
    name = "suffix_sft"
    required_tasks = None

    def attach_router(self, app) -> None:
        from fastapi import Request
        if os.environ.get("SUFFIX_SFT", "").strip() != "1":
            _log("routes skipped (SUFFIX_SFT != 1)")
            return

        async def train(request):
            return await train_route(request)

        async def load(request):
            return await load_route(request)

        # real classes, not strings: this module uses postponed annotations
        # and FastAPI would otherwise read `request` as a query parameter
        train.__annotations__ = load.__annotations__ = {"request": Request}

        app.add_api_route("/v1/sft/train", train, methods=["POST"], name="suffix_sft_train")
        app.add_api_route("/v1/sft/load", load, methods=["POST"], name="suffix_sft_load")
        app.add_api_route("/v1/sft/status", status_route, methods=["GET"], name="suffix_sft_status")
        if os.environ.get("SUFFIX_SFT_DIAG", "").strip() == "1":
            async def diag(request):
                return await diag_route(request)
            diag.__annotations__ = {"request": Request}
            app.add_api_route("/v1/sft/diag", diag, methods=["POST"], name="suffix_sft_diag")
            _log("DEV route: POST /v1/sft/diag")
        _log("routes: POST /v1/sft/train, POST /v1/sft/load, GET /v1/sft/status")

    async def init_state(self, engine_client, state, args) -> None:
        global _S
        if os.environ.get("SUFFIX_SFT", "").strip() != "1" or engine_client is None:
            return
        if not getattr(args, "api_key", None) and not os.environ.get("SUFFIX_SFT_TOKEN", ""):
            _log("disabled: /v1/sft would be open to every client; set --api-key or SUFFIX_SFT_TOKEN")
            return
        if int(getattr(args, "api_server_count", 1) or 1) > 1 or int(
                getattr(args, "data_parallel_size", 1) or 1) > 1:
            _log("disabled: needs a single API server and data_parallel_size 1")
            return
        s = State(engine_client, Config.from_env(), args)
        r = await s.rpc("info", check=False)
        if r.get("error"):
            _log(f"disabled: workers report {r['error']}")
            return
        s.info = {k: v for k, v in r.items() if k != "ms"}
        _S = s
        await s.restore()


def register():
    return SftEndpointPlugin()

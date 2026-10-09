# SPDX-License-Identifier: Apache-2.0
"""Promotion gate: candidate adapter vs live adapter on held-out windows.

Metric = replayed mean accepted draft length per anchor (``#leading
correct depths`` of the greedy chain, depths 1..k, anchors with labels for
all k depths) - the quantity serving turns into tokens/step (1 + it).
Both adapters are replayed on the SAME windows (paired). Promote when
candidate - live >= margin and anchors >= min_anchors. The swap copies
the candidate into the live tensors in place (CUDA-graph safe) on the
serving stream, between engine steps. Per-language buckets are reported
when a tag is available (``lang_tag``); they never gate.
"""
from __future__ import annotations

import re

import torch

from suffix_hybrid.mtp_tune.train import chain_replay

_DE_WORDS = frozenset(
    "der die das und ist nicht ich sie es ein eine mit auf für von zu den dem "
    "des sich auch wir ihr wie aber oder wenn dass noch nur bei nach werden".split())
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def lang_tag(text: str) -> str:
    """'de' | 'other' | '?' from decoded text: umlaut/ß density or German
    function-word share. Cheap and deliberately rough."""
    words = _WORD.findall(text.lower())
    if len(words) < 8:
        return "?"
    umlaut = sum(ch in "äöüß" for ch in text.lower()) / max(len(text), 1)
    de = sum(w in _DE_WORDS for w in words) / len(words)
    return "de" if umlaut > 0.004 or de > 0.12 else "other"


class Stats:
    """Accumulates accepted-length sums per bucket for one adapter."""

    def __init__(self, k):
        self.k = k
        self.buckets = {}   # tag -> [anchors, accepted_sum, depth-1 correct, parity hits, parity n]

    def add(self, rep, tags, served_d1=None):
        corr, full = rep["correct"], rep["full"]
        run = torch.cumprod(corr.long(), 0).sum(0)          # [B,T] leading correct
        for i, tag in enumerate(tags):
            f = full[i]
            b = self.buckets.setdefault(tag, [0, 0, 0, 0, 0])
            b[0] += int(f.sum())
            b[1] += int(run[i][f].sum())
            b[2] += int(corr[0, i][f].sum())
            if served_d1 is not None:
                s = served_d1[i]
                m = f & (s >= 0)
                b[3] += int((rep["pred1"][i][m] == s[m]).sum())
                b[4] += int(m.sum())

    def total(self):
        n = sum(b[0] for b in self.buckets.values())
        acc = sum(b[1] for b in self.buckets.values())
        return n, acc / max(n, 1)

    def summary(self):
        out = {}
        for tag, (n, acc, d1, ph, pn) in sorted(self.buckets.items()):
            out[tag] = dict(n=n, acc_len=acc / max(n, 1), d1=d1 / max(n, 1),
                            parity=(ph / pn) if pn else None)
        return out


class Gate:
    def __init__(self, k, margin=0.05, min_anchors=2000):
        self.k, self.margin, self.min_anchors = k, margin, min_anchors
        self.promotions, self.last = 0, None
        self.reset()

    def reset(self):
        self.live, self.cand = Stats(self.k), Stats(self.k)

    def feed(self, fam, batch, tags, live_params, cand_params, served_d1=None):
        self.live.add(chain_replay(fam, batch, live_params, self.k), tags, served_d1)
        self.cand.add(chain_replay(fam, batch, cand_params, self.k), tags)

    def decide(self):
        n, live = self.live.total()
        _, cand = self.cand.total()
        ok = n >= self.min_anchors and cand - live >= self.margin
        self.last = dict(n=n, live=live, cand=cand, promote=ok,
                         live_buckets=self.live.summary(), cand_buckets=self.cand.summary())
        return ok

    @torch.no_grad()
    def promote(self, loras, cand_params):
        for name, lora in loras.items():
            lora.load_(*cand_params[name])
        self.promotions += 1

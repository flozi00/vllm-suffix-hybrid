#!/usr/bin/env bash
# Bake harvested ficache seeds into a new S3 runtime-bundle prefix.
# Usage: scripts/bake_seed.sh <harvested-seeds-dir>
# Precondition: logs already harvested via scripts/ficache_harvest.py.
# Steps: drop seeds into sm120/ficache/seeds/ (replace), commit, CI,
# download artifact, push orphan runtime-<sha> branch. Prints the exact
# download_weights args for the console (prefix plugins/suffix-hybrid-sm120-<sha>).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS_SRC="${1:?usage: bake_seed.sh <harvested-seeds-dir>}"
BRANCH="${BRANCH:-sm120-ficache}"
cd "$REPO"
git -c advice.detachedHead=false fetch origin "$BRANCH"
WT="$(mktemp -d /tmp/bake-seed-XXXX)"
trap 'git worktree remove --force "$WT" 2>/dev/null || true; rmdir "$WT" 2>/dev/null || true' EXIT
git worktree add "$WT" "origin/$BRANCH" --detach
rm -rf "$WT/sm120/ficache/seeds"
mkdir -p "$WT/sm120/ficache/seeds"
cp "$SEEDS_SRC"/*.json.gz "$WT/sm120/ficache/seeds/"
SHA_SHORT="$(git rev-parse --short=8 HEAD@{0} 2>/dev/null || echo seed)"
( cd "$WT" && git add -A && git commit -q -m "ficache: bake tuned MoE tactic seeds ($(ls "$WT/sm120/ficache/seeds" | wc -l | tr -d ' ') files)" )
NEW_SHA="$(cd "$WT" && git rev-parse --short=8 HEAD)"
git push -q origin "HEAD:refs/heads/seed-$NEW_SHA"
git push -q origin "origin/$BRANCH:refs/heads/$BRANCH" 2>/dev/null || true  # seeds ride on their own branch
gh workflow run native.yml --ref "seed-$NEW_SHA"
echo "CI triggered on seed-$NEW_SHA. When green:"
echo "  gh run download <run-id> -D /tmp/sh-seed"
echo "  # orphan runtime branch = bundle-only, then:"
echo "  download_weights(source=github, repo=flozi00/vllm-suffix-hybrid, ref=runtime-$NEW_SHA, prefix=plugins/suffix-hybrid-sm120-$NEW_SHA)"

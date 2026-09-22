#!/usr/bin/env bash
# devsync — push a fresh plugin bundle into the running next-spec-dev pod and
# restart vLLM in place (same pod: /plugins emptyDir + page cache survive).
#
# Requires the scoped kubeconfig from ops/next-spec-dev-devmode.yaml:
#   export KUBECONFIG=~/kubeconfigs/next-spec-dev-dev.conf
#
# Usage:
#   scripts/devsync.sh /tmp/plugin-art3            # sync bundle dir, restart
#   scripts/devsync.sh /tmp/plugin-art3 --no-kill  # sync only
#   scripts/devsync.sh --args '{"--moe-backend":"flashinfer_b12x"}'
#       # write dev_args.json (replaces existing), then restart
#
# The pod's PID 1 is the vLLM entrypoint; killing it restarts the container
# IN PLACE (restartCount+1, no reschedule, no weight re-stream from S3 when
# weights_staging is on). sitecustomize re-applies dev_args.json on every
# process start, so the dev delta survives restarts until you delete it.
set -euo pipefail

NS=${NS:-maas-inference}
POOL=${POOL:-next-spec-dev}
EP=${EP:-https://next-spec-dev.pl-ai.net}

BUNDLE=""
ARGS_JSON=""
if [[ "${1:-}" == "--args" ]]; then
  [[ -n "${2:-}" ]] || { echo "usage: devsync.sh --args '<json>'" >&2; exit 2; }
  ARGS_JSON=$2
else
  BUNDLE=${1:-}
  MODE=${2:-}
  [[ -n "$BUNDLE" ]] || { echo "usage: devsync.sh <bundle-dir> [--no-kill] | --args '<json>'" >&2; exit 2; }
fi

POD=$(kubectl -n "$NS" get pods -l "app=$POOL" \
  -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' \
  | awk '{print $1}')
[[ -n "$POD" ]] || { echo "no Running pod for $POOL in $NS" >&2; exit 1; }
echo "pod: $POD"

if [[ -n "$ARGS_JSON" ]]; then
  # The splice reads /plugins/dev_args.json on every process start, so the
  # dev delta survives kill-PID1 restarts until you delete the file.
  printf '%s' "$ARGS_JSON" > /tmp/dev_args.json
  kubectl -n "$NS" cp /tmp/dev_args.json "$POD:/plugins/dev_args.json"
  echo "dev_args.json -> $POD:/plugins/dev_args.json"
elif [[ -d "$BUNDLE" ]]; then
  kubectl -n "$NS" cp "$BUNDLE/sitecustomize.py" "$POD:/plugins/sitecustomize.py"
  kubectl -n "$NS" cp "$BUNDLE/BUILD.json"      "$POD:/plugins/BUILD.json"
  kubectl -n "$NS" cp "$BUNDLE/suffix_hybrid"   "$POD:/plugins/suffix_hybrid"
  echo "bundle synced: $(tr -d '\n ' < "$BUNDLE/BUILD.json" | head -c 120)"
fi

if [[ "$MODE" == "--no-kill" ]]; then echo "done (no restart)"; exit 0; fi

echo "restarting vLLM (kill PID 1, in-place container restart)..."
kubectl -n "$NS" exec "$POD" -- sh -c 'kill 1' || true

echo "waiting for $EP/health ..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$EP/health" 2>/dev/null || true)
  [[ "$code" == "200" ]] && { echo "READY after $((i*10))s"; exit 0; }
  sleep 10
done
echo "NOT READY after 600s — check: kubectl -n $NS logs $POD --tail=100" >&2
exit 1

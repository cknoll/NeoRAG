#!/usr/bin/env bash
# Run / debug GitHub Actions jobs locally via `act`.
#
# Usage:
#   scripts/ci-local.sh                 # list jobs
#   scripts/ci-local.sh <job_id>        # run one job
#   scripts/ci-local.sh <job_id> shell  # run job, then drop into a shell
#                                       # in the (still-running) container

# When you're done with the container:
# docker ps --filter "name=act-" -q | xargs -r docker rm -f

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

JOB="${1:-}"
MODE="${2:-run}"

if [[ -z "$JOB" ]]; then
    act -l
    exit 0
fi

if [[ "$MODE" == "shell" ]]; then
    # --reuse keeps the container alive after the job finishes (or fails),
    # so we can exec into it.
    act -j "$JOB" --reuse || true
    CONTAINER=$(docker ps --filter "name=act-" --format '{{.Names}}' | head -n1)
    if [[ -z "$CONTAINER" ]]; then
        echo "No act container found. The job may have exited cleanly."
        exit 1
    fi
    echo ">>> Entering $CONTAINER"
    # Try the workspace first, fall back to / if it does not exist yet.
    if docker exec "$CONTAINER" test -d /github/workspace; then
        docker exec -it -w /github/workspace "$CONTAINER" bash
    else
        echo "Note: /github/workspace does not exist in the container."
        echo "      Dropping you into / instead."
        docker exec -it "$CONTAINER" bash
    fi
else
    act -j "$JOB"
fi

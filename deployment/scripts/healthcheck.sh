#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="${1:-http://127.0.0.1}"
printf 'showcase: '
curl --fail --silent --show-error --max-time 10 "$BASE_URL/" >/dev/null
printf 'ok\nstreamlit: '
curl --fail --silent --show-error --max-time 10 "$BASE_URL/app/_stcore/health"
printf 'health: '
curl --fail --silent --show-error --max-time 10 "$BASE_URL/health"

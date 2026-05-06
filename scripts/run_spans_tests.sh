#!/usr/bin/env bash
# Run all spans / Legolink tests (unit + e2e) in one shot.
# Forwards extra arguments to pytest, so flags like -v, -s, -k, --pdb work.
#
# Usage:
#   ./scripts/run_spans_tests.sh
#   ./scripts/run_spans_tests.sh -v
#   ./scripts/run_spans_tests.sh -k legolink_replay -v
set -euo pipefail
cd "$(dirname "$0")/.."
exec pytest -m spans "$@"

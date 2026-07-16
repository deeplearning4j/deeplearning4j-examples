#!/usr/bin/env bash
# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
#
# run_server.sh — Start the SDX SDK runner serving REST (port 8080) and gRPC (port 50051).
#
# The runner serves both protocols from a single process.  Models are loaded
# lazily via the REST or gRPC LoadModel call after the server starts.
#
# Usage:
#   ./run_server.sh [--rest-port PORT] [--grpc-port PORT] [--disable-grpc] [--disable-rest]
#   ./run_server.sh --library /custom/path/libnd4jcpu.so
#
# Environment variables (all optional):
#   SDX_RUNTIME_LIBRARY_DIR  — colon-separated directories to scan for the native lib
#   SDX_RUNTIME_HOME         — root of an unpacked SDX SDK
#   PYTHON                   — Python executable to use (default: /usr/bin/python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Locate the SDX Python wrappers ──────────────────────────────────────────
# Priority: $SDX_RUNTIME_HOME/wrappers/python, then sibling deeplearning4j tree.
if [[ -n "${SDX_RUNTIME_HOME:-}" && -f "${SDX_RUNTIME_HOME}/wrappers/python/sdx_sdk_runner.py" ]]; then
    WRAPPER_DIR="${SDX_RUNTIME_HOME}/wrappers/python"
else
    # Look for a sibling deeplearning4j checkout (standard monorepo layout).
    FALLBACK_WRAPPER="${REPO_ROOT}/../deeplearning4j/libnd4j/include/dsp/runtime/bindings/python"
    if [[ -f "${FALLBACK_WRAPPER}/sdx_sdk_runner.py" ]]; then
        WRAPPER_DIR="${FALLBACK_WRAPPER}"
    else
        echo "ERROR: Cannot locate sdx_sdk_runner.py." >&2
        echo "  Set SDX_RUNTIME_HOME to a packaged SDK root, or place a sibling" >&2
        echo "  deeplearning4j checkout at ${REPO_ROOT}/../deeplearning4j/" >&2
        exit 1
    fi
fi

echo "Using wrappers from: ${WRAPPER_DIR}"

# ── Auto-detect library if SDX_RUNTIME_LIBRARY_DIR is not set ───────────────
if [[ -z "${SDX_RUNTIME_LIBRARY_DIR:-}" ]]; then
    CANDIDATE="${REPO_ROOT}/../deeplearning4j/libnd4j/blasbuild/cpu/sdx-runtime-sdk/lib"
    if [[ -d "${CANDIDATE}" ]]; then
        export SDX_RUNTIME_LIBRARY_DIR="${CANDIDATE}"
        echo "Auto-detected SDX_RUNTIME_LIBRARY_DIR=${SDX_RUNTIME_LIBRARY_DIR}"
    fi
fi

# ── Python ───────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-/usr/bin/python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
    echo "ERROR: Python executable not found: ${PYTHON}" >&2
    echo "  Set PYTHON=/path/to/python3 and ensure fastapi, uvicorn, grpcio are installed." >&2
    exit 1
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
REST_PORT="${REST_PORT:-8080}"
GRPC_PORT="${GRPC_PORT:-50051}"

echo "Starting SDX SDK runner:"
echo "  REST  -> http://0.0.0.0:${REST_PORT}"
echo "  gRPC  -> 0.0.0.0:${GRPC_PORT}"
echo "  Press Ctrl-C to stop."
echo ""

# Add wrapper dir to PYTHONPATH so sdx_sdk_runner.py can find sdx_runtime etc.
export PYTHONPATH="${WRAPPER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" "${WRAPPER_DIR}/sdx_sdk_runner.py" \
    --rest-port  "${REST_PORT}" \
    --grpc-port  "${GRPC_PORT}" \
    --log-level  INFO \
    "$@"

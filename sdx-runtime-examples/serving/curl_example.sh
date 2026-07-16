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
# curl_example.sh — Pure-curl REST demo against a running SDX SDK runner.
#
# Demonstrates:
#   1. Health check (GET /healthz)
#   2. Model load (POST /v1/models:load)
#   3. JSON inference (POST /v1/models/{id}:run) with the canonical MLP input
#   4. Model unload (POST /v1/models/{id}:unload)
#
# Run (with the server already started via run_server.sh):
#   ./curl_example.sh [http://127.0.0.1:8080] [/path/to/model.sdz]
#
# Input tensors (float32, dtype code 5) are pre-encoded as base64.
# The canonical input x = linspace(0.1, 0.8, 8).reshape(2,4) should produce:
#   probs ≈ [[0.4448, 0.3220, 0.2331], [0.4567, 0.3196, 0.2237]]

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${2:-${SCRIPT_DIR}/../models/mlp.sdz}"
MODEL_PATH="$(realpath "${MODEL_PATH}")"

echo "=== SDX REST curl demo ==="
echo "Server: ${BASE_URL}"
echo "Model:  ${MODEL_PATH}"
echo ""

# ── 1. Health check ───────────────────────────────────────────────────────────
echo "--- 1. Health check ---"
curl -sf "${BASE_URL}/healthz" | python3 -m json.tool
echo ""

# ── 2. Load model ─────────────────────────────────────────────────────────────
echo "--- 2. Load model ---"
LOAD_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/models:load" \
  -H "Content-Type: application/json" \
  -d "{\"model_path\": \"${MODEL_PATH}\", \"requested_outputs\": [\"probs\"]}")
echo "${LOAD_RESPONSE}" | python3 -m json.tool

MODEL_ID=$(echo "${LOAD_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_id'])")
echo "Extracted model_id: ${MODEL_ID}"
echo ""

# ── 3. JSON inference ─────────────────────────────────────────────────────────
# mlp.sdz plan binding order (from models/README.md): b2[3], w2[8,3], w1[4,8], b1[8], x[2,4]
# All inputs are float32 (dtype code 5), raw bytes base64-encoded.
#
# Weight values (same linspace values embedded in mlp.sdz):
#   b2  = linspace(-0.1, 0.1, 3)
#   w2  = linspace(1.0, -1.0, 24).reshape(8,3)
#   w1  = linspace(-1.0, 1.0, 32).reshape(4,8)
#   b1  = linspace(0.0, 0.7, 8)
#   x   = linspace(0.1, 0.8, 8).reshape(2,4)  <-- canonical verification input

echo "--- 3. JSON inference ---"
curl -sf -X POST "${BASE_URL}/v1/models/${MODEL_ID}:run" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "b2",
        "dtype": 5,
        "shape": [3],
        "data_base64": "zczMvQAAAADNzMw9"
      },
      {
        "name": "w2",
        "dtype": 5,
        "shape": [8, 3],
        "data_base64": "AACAPzi9aT9velM/pzc9P9/0Jj8WshA/nN70PgtZyD5605s+05tePrKQBT5DFjI9QxYyvbKQBb7Tm16+etObvgtZyL6c3vS+FrIQv9/0Jr+nNz2/b3pTvzi9ab8AAIC/"
      },
      {
        "name": "w1",
        "dtype": 5,
        "shape": [4, 8],
        "data_base64": "AACAv997b7++916/nXNOv3zvPb9bay2/OuccvxljDL/vvfe+rbXWvmuttb4ppZS+zjlnvkopJb6MMca9CCEEvQghBD2MMcY9SiklPs45Zz4ppZQ+a621Pq211j7vvfc+GWMMPzrnHD9bay0/fO89P51zTj++914/33tvPwAAgD8="
      },
      {
        "name": "b1",
        "dtype": 5,
        "shape": [8],
        "data_base64": "AAAAAM3MzD3NzEw+mpmZPs3MzD4AAAA/mpkZPzMzMz8="
      },
      {
        "name": "x",
        "dtype": 5,
        "shape": [2, 4],
        "data_base64": "zczMPc3MTD6amZk+zczMPgAAAD+amRk/MzMzP83MTD8="
      }
    ],
    "outputs": [
      {"name": "probs", "dtype": 5, "shape": [2, 3]}
    ]
  }' | python3 -m json.tool
echo ""

# ── 4. Unload model ───────────────────────────────────────────────────────────
echo "--- 4. Unload model ---"
curl -sf -X POST "${BASE_URL}/v1/models/${MODEL_ID}:unload" | python3 -m json.tool

echo ""
echo "Done.  Expected probs ≈ [[0.4448, 0.3220, 0.2331], [0.4567, 0.3196, 0.2237]]"

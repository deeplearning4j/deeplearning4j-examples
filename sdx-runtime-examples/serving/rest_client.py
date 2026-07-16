#!/usr/bin/env python3
# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""SDX Runtime REST client example.

Demonstrates the full lifecycle against a running sdx_sdk_runner.py server:
  1. Health check (GET /healthz)
  2. Model load (POST /v1/models:load)
  3. JSON inference (POST /v1/models/{id}:run)
  4. NPZ binary inference (POST /v1/models/{id}:run-npz)
  5. Model unload (POST /v1/models/{id}:unload)

No third-party HTTP library is required — uses only urllib from the standard
library and numpy (which is already required by the runner).

Run (with the server already started via run_server.sh):

    /usr/bin/python3 rest_client.py [--host 127.0.0.1] [--port 8080] [model.sdz]

The canonical MLP model (../models/mlp.sdz) is used by default.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Canonical verification vector (matches mlp.sdz baked weights) ────────────
# Model signature from models/README.md:
#   inputs (plan binding order): b2[3], w2[8,3], w1[4,8], b1[8], x[batch,4]
#   output: probs[batch,3]  (float32, dtype code 5)

CANONICAL_X: np.ndarray = np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape(2, 4)
EXPECTED_PROBS: np.ndarray = np.array(
    [
        [0.44481823, 0.32203630, 0.23314552],
        [0.45671480, 0.31961477, 0.22367041],
    ],
    dtype=np.float32,
)
CANONICAL_TOLERANCE: float = 1e-4

MODEL_WEIGHTS: Dict[str, np.ndarray] = {
    "w1": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(4, 8),
    "b1": np.linspace(0.0, 0.7, 8, dtype=np.float32),
    "w2": np.linspace(1.0, -1.0, 24, dtype=np.float32).reshape(8, 3),
    "b2": np.linspace(-0.1, 0.1, 3, dtype=np.float32),
}

# SDX dtype code for float32 (matches nd4j DataType::FLOAT = 5)
DTYPE_FLOAT32 = 5


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────

def _post_json(url: str, payload: Any) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def _post_npz(url: str, body: bytes, output_specs: List[Dict], input_order: List[str]) -> Tuple[bytes, Dict]:
    """POST an NPZ payload and return (npz_response_bytes, execution_report_dict)."""
    headers = {
        "Content-Type": "application/x-sdx-npz",
        "X-SDX-Output-Specs": json.dumps(output_specs),
        "X-SDX-Input-Order": json.dumps(input_order),
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        report_raw = resp.getheader("X-SDX-Execution-Report", "{}")
        return resp.read(), json.loads(report_raw)


# ── Tensor codec helpers ──────────────────────────────────────────────────────

def _tensor_to_json_entry(name: str, arr: np.ndarray) -> Dict:
    arr_c = np.ascontiguousarray(arr)
    return {
        "name": name,
        "dtype": DTYPE_FLOAT32,
        "shape": list(arr_c.shape),
        "data_base64": base64.b64encode(arr_c.tobytes()).decode("ascii"),
    }


def _decode_npz(blob: bytes) -> np.ndarray:
    with np.load(io.BytesIO(blob), allow_pickle=False) as f:
        keys = list(f.files)
        return np.array(f[keys[0]])


def _encode_npz(named_arrays: List[Tuple[str, np.ndarray]]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{name: np.ascontiguousarray(arr) for name, arr in named_arrays})
    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SDX REST client example")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server REST port")
    parser.add_argument("model", nargs="?", default=None, help="Path to .sdz model (default: ../models/mlp.sdz)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    model_path = args.model
    if model_path is None:
        model_path = str(pathlib.Path(__file__).resolve().parent.parent / "models" / "mlp.sdz")
    model_path = str(pathlib.Path(model_path).resolve())

    # ── 1. Health check ──────────────────────────────────────────────────────
    print("=== 1. Health check ===")
    try:
        health = _get_json(f"{base_url}/healthz")
    except urllib.error.URLError as exc:
        print(f"FAIL: Cannot reach server at {base_url}: {exc}")
        print("  Is the server running?  Start it with:  ./run_server.sh")
        return 1
    print(f"Health: {health}")
    assert health["status"] == "ok", f"Unexpected health status: {health}"

    # ── 2. Load model ────────────────────────────────────────────────────────
    # requested_outputs pins which output variable names the context will expose.
    # Without it sdxGetNumOutputs returns 0 and sdxRun will report a mismatch.
    print(f"\n=== 2. Load model: {model_path} ===")
    load_resp = _post_json(
        f"{base_url}/v1/models:load",
        {"model_path": model_path, "requested_outputs": ["probs"]},
    )
    model_id = load_resp["model_id"]
    print(f"Loaded model_id: {model_id}  (abi_version={load_resp['abi_version']})")

    try:
        # ── 3. JSON inference ────────────────────────────────────────────────
        print("\n=== 3. JSON inference (POST /v1/models/{id}:run) ===")
        # Plan binding order from models/README.md: b2, w2, w1, b1, x
        inputs_json = [
            _tensor_to_json_entry("b2", MODEL_WEIGHTS["b2"]),
            _tensor_to_json_entry("w2", MODEL_WEIGHTS["w2"]),
            _tensor_to_json_entry("w1", MODEL_WEIGHTS["w1"]),
            _tensor_to_json_entry("b1", MODEL_WEIGHTS["b1"]),
            _tensor_to_json_entry("x", CANONICAL_X),
        ]
        output_specs = [{"name": "probs", "dtype": DTYPE_FLOAT32, "shape": [2, 3]}]

        run_resp = _post_json(
            f"{base_url}/v1/models/{model_id}:run",
            {"inputs": inputs_json, "outputs": output_specs},
        )
        # Decode the base64 output tensor from the JSON response
        out_entry = run_resp["outputs"][0]
        probs_bytes = base64.b64decode(out_entry["data_base64"])
        probs = np.frombuffer(probs_bytes, dtype=np.float32).reshape(out_entry["shape"])
        report = run_resp["report"]

        print("REST JSON inference OK")
        print(f"  probs:\n{probs}")
        print(f"  report: status_code={report['status_code']}  plan_phase={report['plan_phase']}  "
              f"execution_time_ns={report['execution_time_ns']}")

        if not np.allclose(probs, EXPECTED_PROBS, atol=CANONICAL_TOLERANCE):
            print(f"FAIL: canonical mismatch.\n  got={probs}\n  expected={EXPECTED_PROBS}")
            return 1
        print("  Canonical verification: PASS (within 1e-4)")

        # ── 4. NPZ binary inference ──────────────────────────────────────────
        print("\n=== 4. NPZ binary inference (POST /v1/models/{id}:run-npz) ===")
        named_arrays = [
            ("b2", MODEL_WEIGHTS["b2"]),
            ("w2", MODEL_WEIGHTS["w2"]),
            ("w1", MODEL_WEIGHTS["w1"]),
            ("b1", MODEL_WEIGHTS["b1"]),
            ("x", CANONICAL_X),
        ]
        npz_payload = _encode_npz(named_arrays)
        input_order = [name for name, _ in named_arrays]

        npz_response, npz_report = _post_npz(
            f"{base_url}/v1/models/{model_id}:run-npz",
            npz_payload,
            [{"name": "probs", "dtype": DTYPE_FLOAT32, "shape": [2, 3]}],
            input_order,
        )
        probs_npz = _decode_npz(npz_response)
        print("REST NPZ inference OK")
        print(f"  probs:\n{probs_npz}")
        print(f"  report: status_code={npz_report['status_code']}  plan_phase={npz_report['plan_phase']}  "
              f"execution_time_ns={npz_report['execution_time_ns']}")

        if not np.allclose(probs_npz, EXPECTED_PROBS, atol=CANONICAL_TOLERANCE):
            print(f"FAIL: NPZ canonical mismatch.\n  got={probs_npz}\n  expected={EXPECTED_PROBS}")
            return 1
        print("  Canonical verification: PASS (within 1e-4)")

    finally:
        # ── 5. Unload model ──────────────────────────────────────────────────
        print(f"\n=== 5. Unload model {model_id} ===")
        unload_resp = _post_json(f"{base_url}/v1/models/{model_id}:unload", {})
        print(f"Unload status: {unload_resp['status']}")

    print("\nSUCCESS: SDX REST serving round-trip verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

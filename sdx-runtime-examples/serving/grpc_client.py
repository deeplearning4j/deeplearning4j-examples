#!/usr/bin/env python3
# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""SDX Runtime gRPC client example.

Demonstrates the full gRPC lifecycle against a running sdx_sdk_runner.py server:
  1. Health check (SdxRuntimeService.Health)
  2. Model load (SdxRuntimeService.LoadModel)
  3. Inference (SdxRuntimeService.Run)
  4. Model unload (SdxRuntimeService.UnloadModel)

Prerequisites — generate gRPC stubs (once):

    /usr/bin/python3 -m grpc_tools.protoc \\
        -I. --python_out=. --grpc_python_out=. sdx_serving.proto
    # Produces: sdx_serving_pb2.py  sdx_serving_pb2_grpc.py

Run (with the server already started via run_server.sh):

    /usr/bin/python3 grpc_client.py [--host 127.0.0.1] [--port 50051] [model.sdz]

The max gRPC message size is raised to 64 MiB to handle large ndarray payloads
(per ADR 0074 — the gRPC default of 4 MiB is too small for real workloads).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict

import grpc  # type: ignore  # pip install grpcio
import numpy as np

# gRPC stubs — generated from sdx_serving.proto via grpcio-tools.
# If missing, run: python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sdx_serving.proto
try:
    import sdx_serving_pb2      # type: ignore
    import sdx_serving_pb2_grpc  # type: ignore
except ImportError:
    print(
        "ERROR: gRPC stubs not found.  Generate them first:\n"
        "  /usr/bin/python3 -m grpc_tools.protoc \\\n"
        "      -I. --python_out=. --grpc_python_out=. sdx_serving.proto",
        file=sys.stderr,
    )
    sys.exit(1)

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

# SDX dtype code for float32 (nd4j DataType::FLOAT = 5)
DTYPE_FLOAT32 = 5

# gRPC message size: 64 MiB (ADR 0074 mandates raising the 4 MiB default)
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _numpy_to_tensor(name: str, arr: np.ndarray) -> "sdx_serving_pb2.Tensor":
    """Pack a numpy array into a protobuf Tensor message."""
    arr_c = np.ascontiguousarray(arr, dtype=np.float32)
    t = sdx_serving_pb2.Tensor()
    t.name = name
    t.dtype = DTYPE_FLOAT32
    t.shape.extend(arr_c.shape)
    t.data = arr_c.tobytes()
    return t


def _tensor_to_numpy(tensor: "sdx_serving_pb2.Tensor") -> np.ndarray:
    """Unpack a protobuf Tensor message into a numpy array."""
    shape = list(tensor.shape)
    return np.frombuffer(bytes(tensor.data), dtype=np.float32).reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(description="SDX gRPC client example")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("model", nargs="?", default=None, help="Path to .sdz model (default: ../models/mlp.sdz)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = str(pathlib.Path(__file__).resolve().parent.parent / "models" / "mlp.sdz")
    model_path = str(pathlib.Path(model_path).resolve())

    address = f"{args.host}:{args.port}"
    channel_options = [
        ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
    ]

    with grpc.insecure_channel(address, options=channel_options) as channel:
        stub = sdx_serving_pb2_grpc.SdxRuntimeServiceStub(channel)

        # ── 1. Health check ──────────────────────────────────────────────────
        print("=== 1. Health check ===")
        try:
            health_resp = stub.Health(sdx_serving_pb2.HealthRequest())
        except grpc.RpcError as exc:
            print(f"FAIL: Cannot reach gRPC server at {address}: {exc.details()}")
            print("  Is the server running?  Start it with:  ./run_server.sh")
            return 1
        print(f"Health: status={health_resp.status}  abi_version={health_resp.abi_version}")
        assert health_resp.status == "ok", f"Unexpected health status: {health_resp.status}"

        # ── 2. Load model ────────────────────────────────────────────────────
        print(f"\n=== 2. Load model: {model_path} ===")
        load_resp = stub.LoadModel(
            sdx_serving_pb2.LoadModelRequest(
                model_path=model_path,
                requested_outputs=["probs"],
            )
        )
        model_id = load_resp.model_id
        print(f"Loaded model_id: {model_id}  (abi_version={load_resp.abi_version})")

        try:
            # ── 3. Inference ─────────────────────────────────────────────────
            print("\n=== 3. gRPC inference (SdxRuntimeService.Run) ===")
            # Plan binding order from models/README.md: b2, w2, w1, b1, x
            run_req = sdx_serving_pb2.RunRequest(
                model_id=model_id,
                inputs=[
                    _numpy_to_tensor("b2", MODEL_WEIGHTS["b2"]),
                    _numpy_to_tensor("w2", MODEL_WEIGHTS["w2"]),
                    _numpy_to_tensor("w1", MODEL_WEIGHTS["w1"]),
                    _numpy_to_tensor("b1", MODEL_WEIGHTS["b1"]),
                    _numpy_to_tensor("x", CANONICAL_X),
                ],
                outputs=[
                    sdx_serving_pb2.TensorSpec(name="probs", dtype=DTYPE_FLOAT32, shape=[2, 3]),
                ],
            )
            run_resp = stub.Run(run_req)

            if not run_resp.outputs:
                print("FAIL: no outputs returned by gRPC Run")
                return 1

            probs = _tensor_to_numpy(run_resp.outputs[0])
            report = run_resp.report

            print("gRPC inference OK")
            print(f"  probs[0] = {probs[0]}")
            print(f"  probs[1] = {probs[1]}")
            print(
                f"  plan_phase={report.plan_phase}  "
                f"execution_count={report.execution_count}  "
                f"execution_time_ns={report.execution_time_ns}"
            )

            if not np.allclose(probs, EXPECTED_PROBS, atol=CANONICAL_TOLERANCE):
                print(f"FAIL: canonical mismatch.\n  got={probs}\n  expected={EXPECTED_PROBS}")
                return 1
            print("  Canonical verification: PASS (within 1e-4)")

        finally:
            # ── 4. Unload model ──────────────────────────────────────────────
            print(f"\n=== 4. Unload model {model_id} ===")
            stub.UnloadModel(sdx_serving_pb2.UnloadModelRequest(model_id=model_id))
            print("Unloaded.")

    print("\nSUCCESS: SDX gRPC serving round-trip verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

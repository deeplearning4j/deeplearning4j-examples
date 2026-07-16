# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
"""End-to-end walkthrough of the SDX runtime from Python — no JVM involved.

Loads ``models/mlp.sdz`` (a softmax MLP exported by the Java
``GenerateExampleModel`` tool) through the SDX C ABI via the ``sdx_runtime``
ctypes wrapper and demonstrates the full SDK lifecycle using a numpy-first,
onnxruntime-style API:

1. Runtime → model → context creation with context managers.
2. Input-contract discovery via ``ctx.get_inputs()`` — returns
   ``InputMetadata`` objects (name + index), mirroring
   ``onnxruntime.InferenceSession.get_inputs()``.
3. Placeholder marking, warmup runs, ``freeze_shapes()`` and the DSP
   REPLAYING fast path.
4. Execution via ``ctx.run_named(feed_dict, [output_buffer])`` — dict-by-name
   inputs like ``session.run(output_names, {"x": arr})``.
5. Structured telemetry via ``ExecutionSummary`` dataclass (plan phase,
   backend, timing, fallback).
6. Canonical output verification (probs[2, 3] ≤ 1e-4 of expected values).
7. Error-path demonstration (``last_error()``).

Running the example
-------------------
Use the **system** Python (not conda) to avoid libstdc++ version conflicts::

    SDX_RUNTIME_LIBRARY_DIR=/path/to/sdk/lib /usr/bin/python3 end_to_end.py

The wrapper is resolved from ``SDX_RUNTIME_HOME/wrappers/python`` when set,
otherwise from a sibling ``deeplearning4j`` checkout.  Requires numpy.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Dict, Sequence

import numpy as np

# ── Locate the sdx_runtime wrapper ──────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent

_WRAPPER_CANDIDATES = []
if os.environ.get("SDX_RUNTIME_HOME"):
    _WRAPPER_CANDIDATES.append(
        pathlib.Path(os.environ["SDX_RUNTIME_HOME"]) / "wrappers" / "python"
    )
_WRAPPER_CANDIDATES.append(
    _THIS_DIR.parents[2]
    / "deeplearning4j"
    / "libnd4j"
    / "include"
    / "dsp"
    / "runtime"
    / "bindings"
    / "python"
)
for _cand in _WRAPPER_CANDIDATES:
    if (_cand / "sdx_runtime.py").exists():
        sys.path.insert(0, str(_cand))
        break

from sdx_runtime import (  # noqa: E402
    ExecutionSummary,
    InputMetadata,
    ModelOptions,
    SdxRuntime,
    SDX_BACKEND_AUTO,
)

# ── Model fixture ────────────────────────────────────────────────────────────
# MLP: probs = softmax(relu(x @ W1 + b1) @ W2 + b2)
# External plan inputs (constants + variables + placeholders, all positional):
#   w1[4, 8], b1[8], w2[8, 3], b2[3]  — fixed weights
#   x[batch, 4]                         — variable input (placeholder)

# Canonical verification vector from GenerateExampleModel:
CANONICAL_X: np.ndarray = np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape(2, 4)
EXPECTED_PROBS: np.ndarray = np.array(
    [
        [0.44481823, 0.3220363, 0.23314552],
        [0.4567148, 0.31961477, 0.22367041],
    ],
    dtype=np.float32,
)
CANONICAL_TOLERANCE: float = 1e-4

# Weights shipped with the example (same values the Java tool embedded in mlp.sdz).
MODEL_WEIGHTS: Dict[str, np.ndarray] = {
    "w1": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(4, 8),
    "b1": np.linspace(0.0, 0.7, 8, dtype=np.float32),
    "w2": np.linspace(1.0, -1.0, 24, dtype=np.float32).reshape(8, 3),
    "b2": np.linspace(-0.1, 0.1, 3, dtype=np.float32),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_feed(
    meta: Sequence[InputMetadata],
    x_value: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build a name→array feed dict from model metadata and the current x."""
    return {
        m.name: MODEL_WEIGHTS[m.name] if m.name in MODEL_WEIGHTS else np.ascontiguousarray(x_value)
        for m in meta
    }


def _run_and_verify(
    ctx,
    meta: Sequence[InputMetadata],
    probs: np.ndarray,
    step: int,
) -> bool:
    """Execute one inference step; return True if all checks pass."""
    x_value = CANONICAL_X * step
    feed = _build_feed(meta, x_value)

    # onnxruntime-style: run_named reorders feed_dict to plan binding order.
    report: ExecutionSummary = ctx.run_named(feed, [probs])

    row_sums_ok = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))
    canonical_ok = True
    checks = [f"row-sums≈1: {row_sums_ok}"]
    if step == 1:
        canonical_ok = bool(np.allclose(probs, EXPECTED_PROBS, atol=CANONICAL_TOLERANCE))
        checks.append(f"matches canonical: {canonical_ok}")

    print(
        f"  run {step}: phase={report.plan_phase_name:<20} "
        f"count={report.execution_count}  "
        f"time={report.execution_time_ms:.3f}ms  "
        + "  ".join(checks)
    )
    return row_sums_ok and canonical_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    model_path = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else _THIS_DIR.parent / "models" / "mlp.sdz"
    )
    if not model_path.exists():
        print(
            f"Model not found: {model_path}\n"
            "Generate it with the java-end-to-end GenerateExampleModel tool."
        )
        return 2

    # ── Step 1: create the runtime and load the model ────────────────────────
    print(f"== Step 1: create the runtime and load {model_path.name} ==")
    with SdxRuntime() as runtime:
        print(f"SDX runtime ABI version: {runtime.abi_version()}")

        with runtime.load_model(str(model_path), ModelOptions(backend=SDX_BACKEND_AUTO)) as model:
            with model.create_context(["probs"]) as ctx:

                # ── Step 2: discover the plan's input contract ───────────────
                print("\n== Step 2: discover the plan's input contract ==")
                # get_inputs() mirrors onnxruntime's session.get_inputs(): returns
                # a list of InputMetadata objects (name + positional index).
                meta = ctx.get_inputs()
                print(
                    f"Plan expects {ctx.num_inputs()} external inputs, "
                    f"{ctx.num_outputs()} output(s):"
                )
                for m in meta:
                    kind = "weight" if m.name in MODEL_WEIGHTS else "placeholder"
                    print(f"  input[{m.index}] = {m.name!r}  ({kind})")

                # Mark the input tensor "x" as a PLACEHOLDER (shape fixed, values change).
                placeholder_indices = [m.index for m in meta if m.name not in MODEL_WEIGHTS]
                for idx in placeholder_indices:
                    ctx.mark_input_placeholder(idx)

                # Allocate the output buffer once; run_named fills it in-place.
                probs = np.zeros((2, 3), dtype=np.float32)

                # ── Step 3: warmup runs (SLOT_BY_SLOT → SHAPES_FROZEN) ───────
                print("\n== Step 3: warmup runs ==")
                for step in (1, 2, 3):
                    if not _run_and_verify(ctx, meta, probs, step):
                        return 1

                # ── Step 4: freeze → DSP REPLAYING fast path ─────────────────
                print("\n== Step 4: freeze_shapes() → DSP REPLAYING fast path ==")
                ctx.freeze_shapes()
                print(f"Plan phase after freeze: {ctx.plan_phase()}")
                for step in (4, 5, 6):
                    if not _run_and_verify(ctx, meta, probs, step):
                        return 1

                # ── Step 5: structured execution-report telemetry ────────────
                print("\n== Step 5: execution report (ExecutionSummary dataclass) ==")
                # run_named always returns an ExecutionSummary.  Here we run
                # one final canonical inference and inspect the report.
                feed = _build_feed(meta, CANONICAL_X)
                report: ExecutionSummary = ctx.run_named(feed, [probs])

                print(f"  status_code         = {report.status_code}")
                print(f"  requested_backend   = {report.requested_backend}")
                print(f"  applied_backend     = {report.applied_backend_name}")
                print(f"  used_fallback       = {report.used_fallback}")
                print(f"  plan_phase          = {report.plan_phase_name}")
                print(f"  execution_count     = {report.execution_count}")
                print(f"  execution_time_ms   = {report.execution_time_ms:.3f}")

                # Final canonical check on the structured report run.
                if not np.allclose(probs, EXPECTED_PROBS, atol=CANONICAL_TOLERANCE):
                    print(f"FAIL: canonical mismatch.\n  got={probs}\n  expected={EXPECTED_PROBS}")
                    return 1

        # ── Step 6: the error path is part of the ABI too ───────────────────
        print("\n== Step 6: error handling ==")
        try:
            runtime.load_model("/definitely/not/a/model.sdz", None)
            print("unexpected: bogus load succeeded")
            return 1
        except RuntimeError as exc:
            print(f"Loading a bogus path raised: {exc}")

    print("\nSUCCESS: SDX C ABI outputs verified from pure Python (no JVM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

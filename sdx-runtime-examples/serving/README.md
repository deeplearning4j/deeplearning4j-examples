# SDX Runtime Serving Example — REST + gRPC

The SDX serving layer (specified in ADR 0074 — *SDX Runtime Serving Protocol*) exposes
any `.sdz` / `.sdnb` model bundle over two complementary transports: a **REST** API
(FastAPI/Uvicorn) for debugging and compatibility, and a **gRPC** API for typed
binary-first workloads.  Both transports share a single execution core
(`sdx_sdk_runner.py` + `sdx_tensor_transport.py`) so their behaviour is identical —
the JSON/base64 REST path and the NPZ binary REST path both call the same
`sdxRun(...)` C ABI underneath.  This example demonstrates the full lifecycle:
load a model, query health, run inference over REST (JSON and NPZ), and run
inference over gRPC.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8 + | Use the **system** python (`/usr/bin/python3`), not a conda env, to avoid libstdc++ conflicts |
| `numpy` | `pip install numpy` |
| `fastapi` + `uvicorn[standard]` | REST server: `pip install fastapi uvicorn[standard]` |
| `grpcio` + `grpcio-tools` | gRPC client and stub generation: `pip install grpcio grpcio-tools` |
| SDX runtime library | `libsdx_cpu.so` / `libnd4jcpu.so` (CPU) or their CUDA equivalents |
| `mlp.sdz` model | Bundled at `../models/mlp.sdz`; regenerate with the java-end-to-end tool |
| SDK Python wrappers | `sdx_runtime.py`, `sdx_sdk_runner.py`, `sdx_tensor_transport.py` from the framework |

### Locating the SDX runtime library

The loader searches in priority order:

1. **`--library /path/to/libnd4jcpu.so`** — explicit path passed to the runner.
2. **`SDX_RUNTIME_LIBRARY_DIR`** env var — colon-separated list of directories to
   scan for `libsdx_cpu.so` / `libnd4jcpu.so` (and their CUDA/AMD equivalents).
3. **`SDX_RUNTIME_HOME`** env var — root of an unpacked SDX SDK; the loader
   checks `$SDX_RUNTIME_HOME/lib/`, `$SDX_RUNTIME_HOME/wrappers/python/`, and
   `$SDX_RUNTIME_HOME/bindings/<platform>/<backend>/lib/`.
4. System library path (`LD_LIBRARY_PATH`, `ldconfig` cache).

For a local build from this repository the library lives at:

```
deeplearning4j/libnd4j/blasbuild/cpu/sdx-runtime-sdk/lib/libnd4jcpu.so
```

Set this path before starting the server:

```bash
export SDX_RUNTIME_LIBRARY_DIR=/path/to/deeplearning4j/libnd4j/blasbuild/cpu/sdx-runtime-sdk/lib
```

### Locating the SDK Python wrappers

The runner (`sdx_sdk_runner.py`) imports `sdx_runtime` and `sdx_tensor_transport`
from the same directory as itself.  `run_server.sh` prepends the framework
wrapper directory (`libnd4j/include/dsp/runtime/bindings/python/`) to `PYTHONPATH`
automatically, or you can point `SDX_RUNTIME_HOME` to a packaged SDK whose
`wrappers/python/` subdirectory carries those modules.

## Starting the server

```bash
cd sdx-runtime-examples/serving

# One command — starts REST on :8080 AND gRPC on :50051 simultaneously
./run_server.sh
```

Customise ports or pass `--disable-grpc` / `--disable-rest` to the underlying
runner:

```bash
SDX_RUNTIME_LIBRARY_DIR=/custom/lib \
  /usr/bin/python3 -m sdx_sdk_runner \
      --rest-port 9090 --grpc-port 50052 \
      --log-level DEBUG
```

## REST endpoints

Once the server is running, a loaded model is needed before inference calls can
be made.  Loading is a one-time operation per server lifetime; the returned
`model_id` is used for all subsequent calls.

### Load the model

```bash
MODEL_ID=$(curl -s -X POST http://localhost:8080/v1/models:load \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "../models/mlp.sdz"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['model_id'])")
echo "model_id: $MODEL_ID"
```

### Health check

```bash
curl -s http://localhost:8080/healthz | python3 -m json.tool
# {"status": "ok", "abi_version": 1}
```

### JSON inference (small/debug workloads)

The `POST /v1/models/{model_id}:run` endpoint accepts tensors as JSON with
base64-encoded data.  The MLP model (`mlp.sdz`) expects five inputs in plan
binding order: `b2 [3]`, `w2 [8,3]`, `w1 [4,8]`, `b1 [8]`, `x [batch,4]`
(all float32, dtype code 5).  The output is `probs [batch,3]` float32.

See `curl_example.sh` for a complete ready-to-run curl command that embeds the
canonical verification vector.

### NPZ binary inference (efficient large payloads)

```bash
curl -s -X POST "http://localhost:8080/v1/models/${MODEL_ID}:run-npz" \
  -H 'Content-Type: application/x-sdx-npz' \
  -H 'X-SDX-Output-Specs: [{"name":"probs","dtype":5,"shape":[2,3]}]' \
  --data-binary @payload.npz \
  -o output.npz
```

The response body is a compressed NPZ archive; the `X-SDX-Execution-Report`
response header carries JSON telemetry (plan phase, backend, timing).

## gRPC

### Generate stubs (one-time)

Copy or reference `sdx_serving.proto` (located here in `serving/` — source:
`deeplearning4j/libnd4j/include/dsp/runtime/bindings/python/sdx_serving.proto`)
and generate Python stubs:

```bash
/usr/bin/python3 -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    sdx_serving.proto
# Produces: sdx_serving_pb2.py  sdx_serving_pb2_grpc.py
```

> The runner auto-generates these stubs on first start via `grpcio-tools` if
> they are not already present — manual generation is only needed for
> `grpc_client.py`.

### Run the gRPC client

```bash
# Generate stubs first (see above), then:
/usr/bin/python3 grpc_client.py
```

Expected output:

```
Health: status=ok  abi_version=1
Loaded model_id: <uuid>
gRPC inference OK
  probs[0] = [0.4448 0.3220 0.2331]
  probs[1] = [0.4567 0.3196 0.2237]
  plan_phase=1  execution_time_ns=<N>
```

## REST Python client

```bash
/usr/bin/python3 rest_client.py
```

Expected output:

```
Health: {'status': 'ok', 'abi_version': 1}
Loaded model_id: <uuid>
REST inference OK
  probs = [[0.4448 0.3220 0.2331]
            [0.4567 0.3196 0.2237]]
  report: status_code=0  plan_phase=1  execution_time_ns=<N>
```

## Expected output (verification)

The canonical verification input `x = linspace(0.1, 0.8, 8).reshape(2,4)` always
produces:

```
probs[2,3] = [[0.44481823, 0.32203630, 0.23314552],
              [0.45671480, 0.31961477, 0.22367041]]
```

Row sums are ≈ 1.0 (softmax output).  Both clients assert this within 1e-4.

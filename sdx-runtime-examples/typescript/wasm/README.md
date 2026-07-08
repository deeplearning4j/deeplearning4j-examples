# SDX Runtime — WebAssembly end-to-end example

The SDX C ABI (`dsp_runtime_c.h`) on WebAssembly: an
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)-style TypeScript
wrapper over an Emscripten `MODULARIZE` module, plus the same canonical
walkthrough as every other language example (input-contract discovery,
placeholder marking, warmup, `freezeShapes` → replay, execution-report
telemetry, canonical output verification, error path).

## Status — read this first

The **wrapper and its heap marshaling are verified**; the **wasm build of the
runtime itself is not yet available** (libnd4j has no Emscripten port today).
The two halves are decoupled:

- `npm run start:mock` runs the full walkthrough against
  `src/mock-module.ts` — a pure-JS **reference implementation of the C ABI**
  over a simulated linear memory. It decodes the `sdx_tensor_view_t` structs
  the wrapper writes at the exact wasm32 offsets, computes the real MLP
  forward pass **from those marshaled bytes**, and writes results back
  through the output views. The canonical-output check therefore proves every
  struct offset, pointer, and byte length the wrapper produces. Verified
  passing (maxDiff ≈ 9e-8).
- `npm start` expects the real `sdx_runtime_wasm.{js,wasm}` next to this
  README, produced by `./build-wasm.sh` — the Emscripten build recipe that
  encodes the target configuration (exported `sdx*` symbols, MEMFS, memory
  growth). Porting libnd4j to Emscripten is the remaining work; the script
  documents the known requirements (CPU-only, generic BLAS, single-threaded
  or COOP/COEP + `-pthread`).

## API at a glance

```typescript
import { SdxRuntime } from './sdx-wasm';

const module  = await createSdxModule();          // Emscripten factory
const runtime = SdxRuntime.create(module);
const model   = runtime.loadBundle('mlp.sdz', sdzBytes);  // bytes -> MEMFS
const session = model.createContext(['probs']);

const names = session.inputNames();               // discover the contract
session.markInputPlaceholder('x');

const [probs] = session.run(
  { ...weights, x: { data: xData, dims: [2, 4] } },  // named feeds
  [[2, 3]],                                          // output specs
);
// probs.data: Float32Array

session.freezeShapes();                            // -> DSP replay fast path
const report = session.executionReport();          // typed telemetry

session.dispose(); model.dispose(); runtime.dispose();
// or on Node 22+/TS 5.2+: using session = model.createContext(['probs']);
```

Conventions: `Tensor { data: Float32Array, dims }` and named feeds
(onnxruntime-web), `Symbol.dispose` on all handles (TC39 explicit resource
management), growth-safe heap views (re-derived per access —
`ALLOW_MEMORY_GROWTH` detaches cached TypedArrays).

## wasm32 struct layouts

Pointers and `size_t` are 4 bytes on wasm32; the wrapper marshals at these
offsets (verified byte-exactly by the reference-ABI run):

```
sdx_tensor_view_t (28B):  data@0  shape@4  rank@8  dtype@12  bytes@16  device_type@20  device_id@24
sdx_execution_report_t (48B): struct_size@0 … used_fallback@16  [pad]  execution_time_ns:u64@24 … plan_phase@40  execution_count@44
```

## Run

```bash
npm install

# Marshaling verification against the JS reference ABI (works everywhere):
npm run start:mock

# Against the real runtime, once built:
./build-wasm.sh /path/to/deeplearning4j     # requires emsdk + libnd4j wasm port
npm start
```

Browser demo: after `npm run build` (and a real wasm build), serve this
directory and open `index.html` — it fetches `../../models/mlp.sdz`, writes it
into MEMFS, and renders the same walkthrough.

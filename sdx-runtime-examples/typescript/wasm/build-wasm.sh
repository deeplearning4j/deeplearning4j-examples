#!/usr/bin/env bash
set -euo pipefail

# Emscripten build recipe for the SDX standalone runtime -> sdx_runtime_wasm.{js,wasm}.
#
# STATUS: libnd4j does not yet ship an Emscripten port — this script is the
# entry point for that work, encoding the target configuration the wrapper in
# src/sdx-wasm.ts programs against. Known porting requirements:
#   * CPU-only, all GPU/JIT backends off (no CUDA, Triton, oneDNN, MLIR,
#     OpenVINO, ARM Compute in wasm).
#   * BLAS: OpenBLAS's assembly kernels do not build under wasm — use the
#     generic C fallback (SD_FALLBACK_BLAS) or a wasm-ported BLAS.
#   * Threads: either -pthread + a COOP/COEP-served page, or single-threaded
#     (disable OpenMP). Start single-threaded.
#   * Filesystem: model bundles load via MEMFS (-sFORCE_FILESYSTEM=1); the
#     wrapper writes .sdz bytes with Module.FS.writeFile before sdxLoadBundle.
#
# Usage:
#   ./build-wasm.sh [/path/to/deeplearning4j]

DL4J_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../deeplearning4j" && pwd)}"
LIBND4J_DIR="${DL4J_DIR}/libnd4j"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LIBND4J_DIR}/blasbuild/wasm"

command -v emcmake >/dev/null || {
  echo "emcmake not found — install and activate the Emscripten SDK (emsdk) first." >&2
  exit 1
}

# The 18 exported sdx* symbols of dsp_runtime_c.h, plus the Emscripten
# allocator. Runtime methods cover string + filesystem marshaling used by
# src/sdx-wasm.ts.
SDX_EXPORTS='_malloc,_free,_sdxGetRuntimeAbiVersion,_sdxCreateRuntime,_sdxDestroyRuntime,_sdxLoadBundle,_sdxUnloadModel,_sdxCreateContext,_sdxDestroyContext,_sdxRun,_sdxGetLastError,_sdxGetExecutionReport,_sdxMarkInputVariable,_sdxMarkInputPlaceholder,_sdxFreezeShapes,_sdxGetPlanPhase,_sdxGetExecutionCount,_sdxGetNumInputs,_sdxGetNumOutputs,_sdxGetInputName'
RUNTIME_METHODS='UTF8ToString,stringToUTF8,lengthBytesUTF8,FS'

emcmake cmake -S "${LIBND4J_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_CPU=ON \
  -DSD_CUDA=OFF \
  -DSD_BUILD_SDX_STANDALONE=ON \
  -DSDX_INCLUDE_TRITON=OFF \
  -DSDX_INCLUDE_ONEDNN=OFF \
  -DSDX_INCLUDE_MLIR=OFF \
  -DSDX_INCLUDE_OPENVINO=OFF \
  -DSD_SHARED_LIB=OFF

cmake --build "${BUILD_DIR}" --target sdx_cpu --parallel "$(nproc)"

# Link the wasm module with the JS glue the wrapper expects.
emcc "${BUILD_DIR}/blas/libsdx_cpu.a" \
  -O3 \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createSdxModule \
  -sEXPORTED_FUNCTIONS="${SDX_EXPORTS}" \
  -sEXPORTED_RUNTIME_METHODS="${RUNTIME_METHODS}" \
  -sALLOW_MEMORY_GROWTH=1 \
  -sFORCE_FILESYSTEM=1 \
  -sENVIRONMENT=web,node \
  -o "${OUT_DIR}/sdx_runtime_wasm.js"

echo "Wrote ${OUT_DIR}/sdx_runtime_wasm.js + .wasm"
echo "Run the example against it with: npm start"

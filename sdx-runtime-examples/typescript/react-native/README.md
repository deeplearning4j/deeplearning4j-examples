# SDX Runtime — TypeScript / React Native end-to-end example

A complete **native-module bridge** for the SDX C ABI with a modern
React Native library layout: TurboModule spec, imperative class, and React hook.

The demo walks the same lifecycle as the other language examples:
input-contract discovery, placeholder marking, DSP warmup, `freezeShapes`,
execution-report telemetry, canonical output verification, and the error path.

```
src/
  NativeSdxRuntime.ts   TurboModule spec (codegen-shaped, getEnforcing)
  SdxSession.ts         Imperative class — open/run/freeze/close lifecycle
  useSdxModel.ts        React hook (react-native-executorch pattern)
  index.ts              Public barrel export
  App.tsx               Demo screen using useSdxModel + session.*

android/
  src/main/java/…/SdxRuntimeModule.kt   Kotlin @ReactMethod bridge
  src/main/cpp/sdx_jni.cpp              JNI/C++ → sdx* C ABI
  src/main/cpp/CMakeLists.txt
  build.gradle

ios/SdxRuntime.mm       ObjC++ RCT_EXPORT_METHOD bridge → sdx* C ABI
```

## Library conventions used

This example follows the conventions established by `react-native-builder-bob`
and the New Architecture (TurboModules):

| Convention | Where |
|---|---|
| `Native<ModuleName>.ts` spec with `TurboModuleRegistry.getEnforcing` | `src/NativeSdxRuntime.ts` |
| `codegenConfig` in `package.json` pointing at `src/` | `package.json` |
| Public barrel re-exporting class + hook (not the raw spec) | `src/index.ts` |
| Imperative class wrapping the spec (onnxruntime-react-native pattern) | `SdxSession` |
| `isReady`/`isLoading`/`error`/`run` hook (react-native-executorch pattern) | `useSdxModel` |

Tensors cross the RN bridge as plain number arrays + shapes — correct for
example-scale payloads. Production apps can replace `SdxTensor.data` with an
ArrayBuffer via a JSI custom binding for zero-copy transfer (see RFC #947).

## API surface

### Hook (recommended for React components)

```ts
import { useSdxModel } from 'sdx-runtime-react-native';

const { isReady, isLoading, error, run, session } = useSdxModel(
  'mlp.sdz',          // bundled asset name
  ['probs'],          // output variable names
  new Set(['x']),     // placeholder inputs (shape fixed after freeze)
);

if (!isReady) return <LoadingView />;

const [probs] = await run(inputs, [[batchSize, 3]]);
```

### Imperative class (for non-component code or tests)

```ts
import { SdxSession } from 'sdx-runtime-react-native';

const session = await SdxSession.open('mlp.sdz', ['probs'], new Set(['x']));
const [probs] = await session.run(inputs, [[2, 3]]);
await session.freezeShapes();      // transition to DSP replay mode
const report = await session.report();
await session.close();             // always close to free native handles
```

### Constants

```ts
import { PLAN_PHASE_NAMES, BACKEND_NAMES, phaseName, backendName } from 'sdx-runtime-react-native';
// 0=SLOT_BY_SLOT,1=SHAPES_FROZEN,2=REPLAYING,3=REPLAY_BLOCKED
// 0=AUTO,1=SLOT_BY_SLOT,2=CUDA_GRAPHS,...,8=NNAPI
```

## Android integration

1. Get the SDK's Android package (`sdx-runtime-android-arm64-cpu.zip` /
   `.aar` from the release artifacts, or build with
   `libnd4j/tools/sdx-generate-bindings.sh --platform android-arm64`).
2. Copy the runtime library into `android/src/main/jniLibs/arm64-v8a/`
   (`libsdx_cpu.so` preferred; the monolithic `libnd4jcpu.so` also works).
3. The JNI shim compiles against `dsp_runtime_c.h`; point CMake at the SDK
   headers if you are not using a sibling `deeplearning4j` checkout:
   `-DSDX_INCLUDE_DIR=/path/to/sdk/include` (see `android/build.gradle`).
4. Add this directory as a module of your app and register
   `SdxRuntimePackage()` in `getPackages()`.
5. Bundle the model: copy `../../models/mlp.sdz` into your app's
   `src/main/assets/`.

## iOS integration

1. Link the SDK's `ND4JDSPRuntime-*.xcframework` (or the runtime dylib) into
   the app target.
2. Add the SDK `include/` directory to `HEADER_SEARCH_PATHS` so
   `<dsp/runtime/dsp_runtime_c.h>` resolves, and add `ios/SdxRuntime.mm` to
   the target sources.
3. Add `mlp.sdz` to the app bundle resources.

## Demo

Render `src/App.tsx` as your app root; it auto-runs the walkthrough on mount
using `useSdxModel` and prints each step, ending with:

```
SUCCESS: SDX C ABI outputs verified from React Native.
```

## Typecheck

```bash
npm install --no-save react@18 react-native@0.74 @types/react@18 typescript@5.5
npx tsc --noEmit   # must print nothing (zero errors)
rm -rf node_modules package-lock.json
```

## Notes on JNI name mangling

The Kotlin `private external fun native*` declarations in `SdxRuntimeModule.kt`
are resolved by JNI symbol name mangling to
`Java_org_nd4j_sdx_rn_SdxRuntimeModule_native*`. These names are stable as long
as the Kotlin class name, package (`org.nd4j.sdx.rn`), and `fun` names are
unchanged — which they are. Any rename in the TS spec has no effect on JNI
mangling because the bridge method names (`@ReactMethod fun getAbiVersion`, etc.)
map to the TurboModule spec independently.

/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * TurboModule spec for the SDX runtime native bridge.
 *
 * Naming convention (required by React Native codegen): the file MUST be named
 * `Native<ModuleName>.ts` and the module string passed to `getEnforcing` MUST
 * match `getName()` on the native side ("SdxRuntime" in SdxRuntimeModule.kt /
 * RCT_EXPORT_MODULE in SdxRuntime.mm).
 *
 * This spec is intentionally codegen-compatible (only types supported by the
 * flow/TS codegen are used). Bulk tensor data crosses the bridge as plain
 * number arrays — ArrayBuffer would be faster for large models but is not yet
 * in the codegen type system (RFC #947). The comment in the public API
 * (`SdxSession`) calls this out so integrators know where to optimise.
 *
 * @see https://reactnative.dev/docs/turbo-native-modules-introduction
 */

import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

/**
 * Execution telemetry returned by `getExecutionReport`.
 *
 * Integer fields are `number` (JS has no int64; doubles are safe below 2^53,
 * which covers all realistic handle addresses on 48-bit Android and iOS).
 */
export interface SdxExecutionReport {
  /** 0 = SDX_STATUS_OK */
  statusCode: number;
  /** Requested backend enum (0=AUTO … 14=HEXAGON — see BACKEND_NAMES). */
  requestedBackend: number;
  /** Actually selected backend enum. */
  appliedBackend: number;
  /** -1=unknown, 0=no, 1=yes */
  usedFallback: number;
  /** Wall-clock ns for the last sdxRun call. */
  executionTimeNs: number;
  /** 0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED */
  planPhase: number;
  /** Total successful sdxRun calls on this context. */
  executionCount: number;
}

/**
 * A single float32 tensor: flat row-major data + shape dimensions.
 * Production apps should replace this with a JSI ArrayBuffer to avoid
 * JSON serialisation overhead over the bridge.
 */
export interface SdxTensor {
  /** Row-major float32 values. */
  data: number[];
  shape: number[];
}

/**
 * Raw TurboModule interface — mirrors the Kotlin `@ReactMethod` surface and
 * the ObjC++ `RCT_EXPORT_METHOD` surface exactly. Consumers should use
 * `SdxSession` or `useSdxModel` instead of calling this directly.
 */
export interface Spec extends TurboModule {
  /** SDX_RUNTIME_ABI_VERSION of the loaded runtime library. */
  getAbiVersion(): Promise<number>;

  /**
   * Resolve a model asset bundled with the app to a readable filesystem path.
   * Android copies the asset into the files dir; iOS returns the bundle
   * resource path.
   */
  resolveModelAsset(assetName: string): Promise<string>;

  /** sdxLoadBundle — returns an opaque model handle (encoded as a double). */
  loadModel(path: string): Promise<number>;
  unloadModel(modelId: number): Promise<void>;

  /** sdxCreateContext — returns an opaque context handle (encoded as a double). */
  createContext(modelId: number, requestedOutputs: string[]): Promise<number>;
  destroyContext(contextId: number): Promise<void>;

  /** sdxGetNumInputs / sdxGetInputName — the plan's positional input contract. */
  getInputNames(contextId: number): Promise<string[]>;
  getNumOutputs(contextId: number): Promise<number>;

  /** Mark a plan input as a trainable variable (shape may change each run). */
  markInputVariable(contextId: number, inputIndex: number): Promise<void>;
  /** Mark a plan input as a placeholder (shape fixed after freeze). */
  markInputPlaceholder(contextId: number, inputIndex: number): Promise<void>;

  /**
   * sdxRun — inputs positionally match getInputNames(); outputShapes carry the
   * caller-declared output shapes. Returns one flat float32 array per output.
   */
  run(
    contextId: number,
    inputs: SdxTensor[],
    outputShapes: number[][],
  ): Promise<number[][]>;

  /** sdxFreezeShapes — transitions the DSP plan from warmup to replay mode. */
  freezeShapes(contextId: number): Promise<void>;

  getPlanPhase(contextId: number): Promise<number>;
  getExecutionCount(contextId: number): Promise<number>;
  getExecutionReport(contextId: number): Promise<SdxExecutionReport>;
}

/**
 * Enforcing accessor — throws immediately if the native module is not linked.
 * Import this from `src/index.ts`; do not reference it from app code.
 */
export default TurboModuleRegistry.getEnforcing<Spec>('SdxRuntime');

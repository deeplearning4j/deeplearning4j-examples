/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * Imperative session class wrapping the raw SDX TurboModule spec.
 *
 * `SdxSession` owns the native model + context handles and exposes a cleaner
 * lifecycle API:
 *  - `SdxSession.open(assetName, requestedOutputs)` — async factory
 *  - `session.run(inputs, outputShapes)` — single inference call
 *  - `session.freezeShapes()` — transition to DSP replay mode
 *  - `session.report()` — execution telemetry
 *  - `session.close()` — deterministic teardown
 *
 * Input marking (variable vs placeholder) is resolved automatically from
 * `getInputNames()` against a caller-supplied `placeholderNames` set: names in
 * the set are marked as placeholders (shape fixed after freeze); all others are
 * treated as variables. Pass `undefined` to skip marking (all inputs variable).
 *
 * Pattern credit: onnxruntime-react-native `InferenceSession` class.
 */

import NativeSdxRuntime from './NativeSdxRuntime';
import type { SdxExecutionReport, SdxTensor } from './NativeSdxRuntime';

export type { SdxExecutionReport, SdxTensor };

export class SdxSession {
  private readonly _modelId: number;
  private readonly _contextId: number;
  private readonly _inputNames: string[];
  private _closed = false;

  private constructor(
    modelId: number,
    contextId: number,
    inputNames: string[],
  ) {
    this._modelId = modelId;
    this._contextId = contextId;
    this._inputNames = inputNames;
  }

  /**
   * Open a session from an app-bundled model asset.
   *
   * @param assetName         Asset filename (e.g. `"mlp.sdz"`).
   * @param requestedOutputs  Output variable names to compute (e.g. `["probs"]`).
   * @param placeholderNames  Names of inputs whose shape is fixed after the
   *                          first freeze (typically the data tensor, not
   *                          weights). Pass `undefined` to skip marking.
   */
  static async open(
    assetName: string,
    requestedOutputs: string[],
    placeholderNames?: ReadonlySet<string>,
  ): Promise<SdxSession> {
    const path = await NativeSdxRuntime.resolveModelAsset(assetName);
    const modelId = await NativeSdxRuntime.loadModel(path);
    const contextId = await NativeSdxRuntime.createContext(modelId, requestedOutputs);
    const inputNames = await NativeSdxRuntime.getInputNames(contextId);

    if (placeholderNames !== undefined) {
      await Promise.all(
        inputNames.map((name, i) =>
          name !== '' && placeholderNames.has(name)
            ? NativeSdxRuntime.markInputPlaceholder(contextId, i)
            : NativeSdxRuntime.markInputVariable(contextId, i),
        ),
      );
    }

    return new SdxSession(modelId, contextId, inputNames);
  }

  /** Positional input names as discovered from the plan. */
  get inputNames(): readonly string[] {
    return this._inputNames;
  }

  /**
   * Run inference. Inputs must be provided in the same positional order as
   * `inputNames`; `outputShapes` must match the number and shapes of the
   * outputs requested at `open()`.
   *
   * Returns one flat float32 array per requested output.
   *
   * Note: tensors cross the bridge as plain number arrays. For large models
   * where this becomes a bottleneck, replace `SdxTensor.data` with an
   * ArrayBuffer transferred via a JSI custom binding (see RFC #947).
   */
  async run(
    inputs: SdxTensor[],
    outputShapes: number[][],
  ): Promise<number[][]> {
    this._assertOpen();
    return NativeSdxRuntime.run(this._contextId, inputs, outputShapes);
  }

  /**
   * Freeze shapes on the DSP plan, transitioning from SLOT_BY_SLOT warmup to
   * the SHAPES_FROZEN → REPLAYING fast path.
   */
  async freezeShapes(): Promise<void> {
    this._assertOpen();
    return NativeSdxRuntime.freezeShapes(this._contextId);
  }

  /** Current DSP plan phase: 0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED. */
  async planPhase(): Promise<number> {
    this._assertOpen();
    return NativeSdxRuntime.getPlanPhase(this._contextId);
  }

  /** Number of successful sdxRun calls on this context since it was created. */
  async executionCount(): Promise<number> {
    this._assertOpen();
    return NativeSdxRuntime.getExecutionCount(this._contextId);
  }

  /** Full execution telemetry for the last run. */
  async report(): Promise<SdxExecutionReport> {
    this._assertOpen();
    return NativeSdxRuntime.getExecutionReport(this._contextId);
  }

  /**
   * Destroy the native context and model handles. Safe to call multiple times
   * (subsequent calls are no-ops). Always call this when done to avoid leaking
   * native memory.
   */
  async close(): Promise<void> {
    if (this._closed) return;
    this._closed = true;
    await NativeSdxRuntime.destroyContext(this._contextId).catch(() => {});
    await NativeSdxRuntime.unloadModel(this._modelId).catch(() => {});
  }

  private _assertOpen(): void {
    if (this._closed) {
      throw new Error('SdxSession has been closed');
    }
  }
}

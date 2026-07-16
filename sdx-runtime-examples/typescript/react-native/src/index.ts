/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * Public API surface for the SDX Runtime React Native library.
 *
 * Library layout follows react-native-builder-bob conventions:
 *   src/NativeSdxRuntime.ts  — TurboModule spec (codegen-shaped, getEnforcing)
 *   src/SdxSession.ts        — Imperative class API (onnxruntime-react-native style)
 *   src/useSdxModel.ts       — React hook (react-native-executorch style)
 *   src/index.ts             — This barrel (the only import consumers need)
 *
 * Consumers should import from this file, not from the sub-modules:
 *   import { useSdxModel, SdxSession, PLAN_PHASE_NAMES } from 'sdx-runtime-react-native';
 *
 * The TurboModule spec (NativeSdxRuntime.ts) is intentionally NOT re-exported
 * here — it is internal plumbing between codegen and the native implementation.
 */

// ── Imperative class ─────────────────────────────────────────────────────────
export { SdxSession } from './SdxSession';
export type { SdxExecutionReport, SdxTensor } from './SdxSession';

// ── React hook ───────────────────────────────────────────────────────────────
export { useSdxModel } from './useSdxModel';
export type { UseSdxModelResult } from './useSdxModel';

// ── Constants / helpers ──────────────────────────────────────────────────────

/**
 * Human-readable names for the DSP plan phase enum.
 * Index matches the `planPhase` field in `SdxExecutionReport`.
 */
export const PLAN_PHASE_NAMES: readonly string[] = [
  'SLOT_BY_SLOT (warmup)',
  'SHAPES_FROZEN',
  'REPLAYING',
  'REPLAY_BLOCKED',
] as const;

/**
 * Human-readable names for the backend enum.
 * Index matches `requestedBackend` / `appliedBackend` in `SdxExecutionReport`.
 */
export const BACKEND_NAMES: readonly string[] = [
  'AUTO',
  'SLOT_BY_SLOT',
  'CUDA_GRAPHS',
  'NVRTC',
  'PTX',
  'TRITON',
  'MLX',
  'ARM_HYBRID',
  'NNAPI',
  'HIP_GRAPHS',
  'LEVEL_ZERO',
  'VULKAN',
  'METAL',
  'TPU',
  'HEXAGON',
] as const;

/** Stringify a plan phase number, falling back to `"? (n)"` for unknowns. */
export const phaseName = (p: number): string =>
  PLAN_PHASE_NAMES[p] ?? `? (${p})`;

/** Stringify a backend number, falling back to `"? (n)"` for unknowns. */
export const backendName = (b: number): string =>
  BACKEND_NAMES[b] ?? `? (${b})`;

/**
 * ABI version of the runtime library linked into this process.
 * Call after the native module is confirmed to be loaded.
 *
 * @deprecated Prefer `SdxSession` or `useSdxModel` for all interactions.
 *             This helper exists for diagnostic / version-check use cases.
 */
export async function getSdxAbiVersion(): Promise<number> {
  const { default: native } = await import('./NativeSdxRuntime');
  return native.getAbiVersion();
}

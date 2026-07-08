/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * React hook for SDX model inference.
 *
 * Follows the pattern established by react-native-executorch hooks
 * (`useObjectDetection`, `useImageEmbeddings`, etc.): the hook owns the
 * session lifecycle, exposes `isReady`/`isLoading`/`error` state, and
 * provides a `run()` function for callers.
 *
 *   const model = useSdxModel('mlp.sdz', ['probs'], new Set(['x']));
 *   if (!model.isReady) return <Loading />;
 *   const [probs] = await model.run(inputs, [[2, 3]]);
 *
 * The session is opened on mount (after the first render) and closed on
 * unmount. Re-rendering with the same `assetName` and `requestedOutputs`
 * does NOT reopen the session.
 *
 * @see https://docs.swmansion.com/react-native-executorch/
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { SdxSession } from './SdxSession';
import type { SdxExecutionReport, SdxTensor } from './SdxSession';

export type { SdxExecutionReport, SdxTensor };

export interface UseSdxModelResult {
  /** True once the session is open and ready for inference. */
  isReady: boolean;
  /** True while the session is being opened. */
  isLoading: boolean;
  /** Non-null if the session failed to open or a run threw. */
  error: Error | null;
  /**
   * Run a single inference. Returns one flat float32 array per requested
   * output. Throws (and sets `error`) if the session is not ready or the
   * native run fails.
   */
  run(inputs: SdxTensor[], outputShapes: number[][]): Promise<number[][]>;
  /**
   * Access the underlying session for advanced use (freeze, report, etc.).
   * Null until `isReady` is true.
   */
  session: SdxSession | null;
}

/**
 * @param assetName         App-bundled model filename (e.g. `"mlp.sdz"`).
 * @param requestedOutputs  Output variable names to compute.
 * @param placeholderNames  Names of inputs to mark as placeholders
 *                          (shape fixed after freeze). Pass `undefined` to
 *                          skip marking.
 */
export function useSdxModel(
  assetName: string,
  requestedOutputs: string[],
  placeholderNames?: ReadonlySet<string>,
): UseSdxModelResult {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const sessionRef = useRef<SdxSession | null>(null);

  // Stable refs for the open parameters so the effect only fires when they
  // semantically change (the caller may create new Set instances each render).
  const assetNameRef = useRef(assetName);
  const outputsKey = requestedOutputs.join(',');
  const placeholdersKey = placeholderNames
    ? [...placeholderNames].sort().join(',')
    : '';

  useEffect(() => {
    let cancelled = false;

    setIsReady(false);
    setIsLoading(true);
    setError(null);

    SdxSession.open(assetNameRef.current, requestedOutputs, placeholderNames)
      .then((session) => {
        if (cancelled) {
          session.close().catch(() => {});
          return;
        }
        sessionRef.current = session;
        setIsReady(true);
        setIsLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
      if (sessionRef.current !== null) {
        sessionRef.current.close().catch(() => {});
        sessionRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetNameRef.current, outputsKey, placeholdersKey]);

  const run = useCallback(
    async (inputs: SdxTensor[], outputShapes: number[][]): Promise<number[][]> => {
      const session = sessionRef.current;
      if (session === null) {
        throw new Error('useSdxModel: session is not ready');
      }
      try {
        return await session.run(inputs, outputShapes);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        throw error;
      }
    },
    [],
  );

  return {
    isReady,
    isLoading,
    error,
    run,
    session: sessionRef.current,
  };
}

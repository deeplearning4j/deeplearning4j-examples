/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * Demo screen: the same end-to-end SDX walkthrough as the other language
 * examples, rendered as a scrollable log on-device.
 *
 * Uses the `useSdxModel` hook (react-native-executorch style) for session
 * lifecycle and calls `session.*` directly for advanced DSP controls
 * (freezeShapes, report). Bundle `models/mlp.sdz` as an app asset named
 * `mlp.sdz` (see README) before running.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text } from 'react-native';
import {
  backendName,
  phaseName,
  useSdxModel,
} from './index';
import type { SdxTensor } from './index';

// ── Canonical verification vector ─────────────────────────────────────────────
// These values are printed by the Java GenerateExampleModel tool that produced
// models/mlp.sdz. x[2,4] = [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]].
const CANONICAL_X: number[] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const EXPECTED_PROBS: number[] = [
  0.44481823, 0.3220363, 0.23314552,
  0.4567148,  0.31961477, 0.22367041,
];

function linspace(start: number, end: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
}

/**
 * The model's weights travel INSIDE the .sdz — a generic client obtains their
 * values from the model provider. This example ships deterministic linspace
 * initializers next to the fixture for simplicity.
 */
const WEIGHTS: Record<string, SdxTensor> = {
  w1: { data: linspace(-1.0, 1.0, 32), shape: [4, 8] },
  b1: { data: linspace(0.0, 0.7, 8),   shape: [8] },
  w2: { data: linspace(1.0, -1.0, 24), shape: [8, 3] },
  b2: { data: linspace(-0.1, 0.1, 3),  shape: [3] },
};

// ── Placeholder input names for this model ────────────────────────────────────
// "x" is the data tensor (shape changes per batch); weight names are variables.
const PLACEHOLDER_NAMES = new Set(['x']);

export default function App(): React.JSX.Element {
  const [lines, setLines] = useState<string[]>([]);
  const log = useCallback((line: string) => {
    setLines((prev) => [...prev, line]);
  }, []);

  // ── useSdxModel hook ────────────────────────────────────────────────────────
  // The hook opens the session on mount, marks "x" as a placeholder, and
  // cleans up on unmount. `session` is non-null once `isReady` is true.
  const { isReady, isLoading, error, run, session } = useSdxModel(
    'mlp.sdz',
    ['probs'],
    PLACEHOLDER_NAMES,
  );

  useEffect(() => {
    if (isLoading) {
      log('Opening SDX session for mlp.sdz…');
      return;
    }
    if (error !== null) {
      log(`FAILURE (session open): ${error.message}`);
      return;
    }
    if (!isReady || session === null) return;

    (async () => {
      try {
        // ── Step 1: report ABI version and input contract ──────────────────
        log('== Step 1: session open — ABI version + input contract ==');
        log(`Input names: [${session.inputNames.join(', ')}]`);
        log(`Placeholders: [${[...PLACEHOLDER_NAMES].join(', ')}]`);

        // ── Step 2: helper ────────────────────────────────────────────────
        const runOnce = async (step: number): Promise<boolean> => {
          // Scale the canonical input per step so values change while the
          // shape stays fixed; step 1 uses the exact canonical vector.
          const x: SdxTensor = {
            data: CANONICAL_X.map((v) => v * step),
            shape: [2, 4],
          };
          const inputs = session.inputNames.map((name) =>
            name === 'x' ? x : WEIGHTS[name] ?? x,
          );
          const [probs] = await run(inputs, [[2, 3]]);

          const rowSumsOk =
            Math.abs(probs[0] + probs[1] + probs[2] - 1) <= 1e-5 &&
            Math.abs(probs[3] + probs[4] + probs[5] - 1) <= 1e-5;
          let ok = rowSumsOk;
          let checks = `rows sum to 1: ${rowSumsOk}`;
          if (step === 1) {
            const maxDiff = Math.max(
              ...EXPECTED_PROBS.map((e, i) => Math.abs(probs[i]! - e)),
            );
            const matches = maxDiff <= 1e-4;
            ok = ok && matches;
            checks += `; canonical match (≤1e-4): ${matches}`;
          }
          const phase = phaseName(await session.planPhase());
          log(
            `  run ${step}: phase=${phase} ` +
              `execCount=${await session.executionCount()}  ${checks}`,
          );
          return ok;
        };

        // ── Step 3: warmup runs ───────────────────────────────────────────
        log('== Step 2: warmup runs (SLOT_BY_SLOT) ==');
        for (let step = 1; step <= 3; step++) {
          if (!(await runOnce(step))) {
            throw new Error(`run ${step} verification failed`);
          }
        }

        // ── Step 4: freeze → DSP replay fast path ─────────────────────────
        log('== Step 3: freezeShapes → DSP replay fast path ==');
        await session.freezeShapes();
        log(`Plan phase after freeze: ${phaseName(await session.planPhase())}`);
        for (let step = 4; step <= 6; step++) {
          if (!(await runOnce(step))) {
            throw new Error(`run ${step} verification failed`);
          }
        }

        // ── Step 5: execution report ──────────────────────────────────────
        log('== Step 4: execution report ==');
        const report = await session.report();
        const fallback =
          report.usedFallback < 0
            ? 'unknown'
            : report.usedFallback === 1
              ? 'yes'
              : 'no';
        log(`  status_code       = ${report.statusCode}`);
        log(`  requested_backend = ${backendName(report.requestedBackend)}`);
        log(`  applied_backend   = ${backendName(report.appliedBackend)}`);
        log(`  used_fallback     = ${fallback}`);
        log(`  plan_phase        = ${phaseName(report.planPhase)}`);
        log(`  execution_count   = ${report.executionCount}`);
        log(`  execution_time    = ${(report.executionTimeNs / 1e6).toFixed(3)} ms`);

        // ── Step 6: error handling ────────────────────────────────────────
        log('== Step 5: error handling ==');
        try {
          // SdxSession.open() on a bad path should reject via E_LOAD.
          const { SdxSession } = await import('./SdxSession');
          await SdxSession.open('/definitely/not/a/model.sdz', ['probs']);
          log('unexpected: bogus load succeeded');
        } catch (e) {
          log(`Loading a bogus path rejected: ${(e as Error).message}`);
        }

        log('');
        log('SUCCESS: SDX C ABI outputs verified from React Native.');
      } catch (e) {
        log(`FAILURE: ${(e as Error).message}`);
      }
    })();
  }, [isReady, isLoading, error, session, run, log]);

  return (
    <SafeAreaView style={styles.root}>
      <Text style={styles.title}>SDX Runtime — end-to-end</Text>
      <ScrollView contentContainerStyle={styles.scroll}>
        {lines.map((line, i) => (
          <Text key={i} style={styles.line}>
            {line}
          </Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#101418' },
  title: { color: '#7fd4ff', fontSize: 18, fontWeight: '600', padding: 12 },
  scroll: { paddingHorizontal: 12, paddingBottom: 24 },
  line: { color: '#d7e1ea', fontFamily: 'monospace', fontSize: 12, lineHeight: 18 },
});

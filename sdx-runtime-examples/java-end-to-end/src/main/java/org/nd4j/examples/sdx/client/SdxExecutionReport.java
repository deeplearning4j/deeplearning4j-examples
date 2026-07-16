/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.examples.sdx.client;

/**
 * Execution telemetry returned by {@link SdxSession#getExecutionReport()}.
 *
 * <p>This is a plain Java object (no JNA dependency), analogous to the
 * {@code OrtModelMetadata} / profiling result types in the ONNX Runtime Java API.
 *
 * <p>Backend and phase codes are translated to human-readable strings by
 * {@link #appliedBackendName()}, {@link #requestedBackendName()}, and
 * {@link #planPhaseName()}.
 */
public final class SdxExecutionReport {

    /** Symbolic names for the {@code SdxBackend} enum (indices 0–14). */
    public static final String[] BACKEND_NAMES = {
            "AUTO", "SLOT_BY_SLOT", "CUDA_GRAPHS", "NVRTC", "PTX", "TRITON",
            "MLX", "ARM_HYBRID", "NNAPI", "HIP_GRAPHS", "LEVEL_ZERO",
            "VULKAN", "METAL", "TPU", "HEXAGON"
    };

    /** Symbolic names for the DSP plan phase (indices 0–3). */
    public static final String[] PLAN_PHASE_NAMES = {
            "SLOT_BY_SLOT (warmup)", "SHAPES_FROZEN", "REPLAYING", "REPLAY_BLOCKED"
    };

    private final int statusCode;
    private final int requestedBackend;
    private final int appliedBackend;
    private final boolean usedFallback;
    private final boolean fallbackKnown;
    private final long executionTimeNs;
    private final int requestedGpuTarget;
    private final int appliedGpuTarget;
    private final int planPhase;
    private final int executionCount;

    SdxExecutionReport(int statusCode, int requestedBackend, int appliedBackend,
                       int usedFallbackRaw, long executionTimeNs,
                       int requestedGpuTarget, int appliedGpuTarget,
                       int planPhase, int executionCount) {
        this.statusCode       = statusCode;
        this.requestedBackend = requestedBackend;
        this.appliedBackend   = appliedBackend;
        this.fallbackKnown    = usedFallbackRaw >= 0;
        this.usedFallback     = usedFallbackRaw == 1;
        this.executionTimeNs  = executionTimeNs;
        this.requestedGpuTarget = requestedGpuTarget;
        this.appliedGpuTarget   = appliedGpuTarget;
        this.planPhase        = planPhase;
        this.executionCount   = executionCount;
    }

    /** The C ABI status code from the last {@code sdxRun}; 0 = OK. */
    public int statusCode() { return statusCode; }

    /** The {@code SdxBackend} ordinal that was requested. */
    public int requestedBackend() { return requestedBackend; }

    /** The {@code SdxBackend} ordinal that was actually used. */
    public int appliedBackend() { return appliedBackend; }

    /**
     * Returns {@code true} if the runtime fell back to a different backend.
     * Only meaningful when {@link #isFallbackKnown()} is {@code true}.
     */
    public boolean usedFallback() { return usedFallback; }

    /** {@code true} if the runtime reported a definite fallback / no-fallback value. */
    public boolean isFallbackKnown() { return fallbackKnown; }

    /** Wall time of the last {@code sdxRun} in nanoseconds. */
    public long executionTimeNs() { return executionTimeNs; }

    /** Wall time of the last {@code sdxRun} in milliseconds. */
    public double executionTimeMs() { return executionTimeNs / 1.0e6; }

    /** Requested GPU device ordinal (-1 = auto). */
    public int requestedGpuTarget() { return requestedGpuTarget; }

    /** Applied GPU device ordinal. */
    public int appliedGpuTarget() { return appliedGpuTarget; }

    /**
     * DSP plan phase at the time the report was captured.
     * 0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED.
     */
    public int planPhase() { return planPhase; }

    /** Total number of successful {@code sdxRun} calls on this context. */
    public int executionCount() { return executionCount; }

    /** Human-readable name for {@link #appliedBackend()}. */
    public String appliedBackendName() { return backendName(appliedBackend); }

    /** Human-readable name for {@link #requestedBackend()}. */
    public String requestedBackendName() { return backendName(requestedBackend); }

    /** Human-readable name for {@link #planPhase()}. */
    public String planPhaseName() { return phaseName(planPhase); }

    /** Human-readable fallback description. */
    public String fallbackDescription() {
        return fallbackKnown ? (usedFallback ? "yes" : "no") : "unknown";
    }

    private static String backendName(int code) {
        return code >= 0 && code < BACKEND_NAMES.length ? BACKEND_NAMES[code] : "? (" + code + ")";
    }

    private static String phaseName(int phase) {
        return phase >= 0 && phase < PLAN_PHASE_NAMES.length ? PLAN_PHASE_NAMES[phase] : "? (" + phase + ")";
    }

    @Override
    public String toString() {
        return "SdxExecutionReport{"
                + "statusCode=" + statusCode
                + ", requestedBackend=" + requestedBackendName()
                + ", appliedBackend=" + appliedBackendName()
                + ", usedFallback=" + fallbackDescription()
                + ", planPhase=" + planPhaseName()
                + ", executionCount=" + executionCount
                + ", executionTimeMs=" + String.format("%.3f", executionTimeMs())
                + '}';
    }
}

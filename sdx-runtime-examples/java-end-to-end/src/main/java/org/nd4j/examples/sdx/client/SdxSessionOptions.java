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
 * Configuration options for an {@link SdxSession}.
 *
 * <p>Analogous to {@code SessionOptions} in the ONNX Runtime Java API.
 * Uses the builder pattern for Java 11 compatibility (no records).
 *
 * <p>Default configuration requests AUTO backend selection with no GPU target
 * override, which is the correct starting point for most applications.
 *
 * <pre>{@code
 * SdxSessionOptions opts = new SdxSessionOptions()
 *     .withBackend(SdxBackend.SLOT_BY_SLOT)
 *     .withGpuTarget(0);
 * try (SdxSession session = env.openSession("model.sdz", outputs, opts)) { ... }
 * }</pre>
 */
public final class SdxSessionOptions {

    /**
     * Backend selector ordinals, matching {@code SdxBackend} in {@code dsp_runtime_c.h}.
     */
    public static final class SdxBackend {
        public static final int AUTO        = 0;
        public static final int SLOT_BY_SLOT = 1;
        public static final int CUDA_GRAPHS  = 2;
        public static final int NVRTC        = 3;
        public static final int PTX          = 4;
        public static final int TRITON       = 5;
        public static final int MLX          = 6;
        public static final int ARM_HYBRID   = 7;
        public static final int NNAPI        = 8;
        public static final int HIP_GRAPHS   = 9;
        public static final int LEVEL_ZERO   = 10;
        public static final int VULKAN       = 11;
        public static final int METAL        = 12;
        public static final int TPU          = 13;
        public static final int HEXAGON      = 14;
        private SdxBackend() {}
    }

    private int backend        = SdxBackend.AUTO;
    private boolean strictBackend   = false;
    private boolean allowRuntimeJit = true;
    private int gpuTarget      = 0;    // 0 = default/auto per C ABI convention

    /** Returns a new options object with default settings. */
    public SdxSessionOptions() {}

    /**
     * Selects the execution backend.  Use one of the {@link SdxBackend} constants.
     *
     * @param backend backend ordinal
     * @return {@code this} for chaining
     */
    public SdxSessionOptions withBackend(int backend) {
        this.backend = backend;
        return this;
    }

    /**
     * If {@code true}, the runtime will fail rather than fall back when the
     * requested backend is unavailable.
     *
     * @param strict whether to disallow fallback
     * @return {@code this} for chaining
     */
    public SdxSessionOptions withStrictBackend(boolean strict) {
        this.strictBackend = strict;
        return this;
    }

    /**
     * Controls whether the runtime may compile kernels at session open time.
     * Defaults to {@code true}.
     *
     * @param allow whether to allow JIT compilation
     * @return {@code this} for chaining
     */
    public SdxSessionOptions withAllowRuntimeJit(boolean allow) {
        this.allowRuntimeJit = allow;
        return this;
    }

    /**
     * Pins the session to a specific GPU device ordinal.
     * Use {@code 0} (the default) to let the runtime select automatically.
     *
     * @param gpuTarget device ordinal (0 = auto per C ABI convention)
     * @return {@code this} for chaining
     */
    public SdxSessionOptions withGpuTarget(int gpuTarget) {
        this.gpuTarget = gpuTarget;
        return this;
    }

    int backend()        { return backend; }
    boolean strictBackend()   { return strictBackend; }
    boolean allowRuntimeJit() { return allowRuntimeJit; }
    int gpuTarget()      { return gpuTarget; }
}

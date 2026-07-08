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

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Low-level JNA binding for the SDX C ABI ({@code dsp_runtime_c.h}).
 *
 * <p>This interface is <em>package-private</em>.  Application code uses the
 * higher-level {@link SdxEnvironment} / {@link SdxSession} / {@link SdxTensor}
 * API instead.  The raw ABI is exposed here only so that the example can
 * demonstrate that any language capable of loading a shared library can bind
 * these same symbols directly.
 *
 * <p>In production Java code, prefer the SDK wrapper shipped under
 * {@code wrappers/java} ({@code org.nd4j.dsp.runtime.SdxRuntime}).
 */
interface SdxAbi extends Library {

    int SDX_STATUS_OK  = 0;
    int SDX_DEVICE_HOST = 0;

    // ── Runtime lifecycle ─────────────────────────────────────────────────────

    int sdxGetRuntimeAbiVersion();

    int sdxCreateRuntime(RuntimeOptions options, PointerByReference outRuntime);
    void sdxDestroyRuntime(Pointer runtime);

    // ── Model lifecycle ───────────────────────────────────────────────────────

    int sdxLoadBundle(Pointer runtime, String bundlePath,
                      ModelOptions options, PointerByReference outModel);
    void sdxUnloadModel(Pointer model);

    // ── Context lifecycle ─────────────────────────────────────────────────────

    int sdxCreateContext(Pointer model, StringArray requestedOutputs,
                         int numRequestedOutputs, PointerByReference outContext);
    void sdxDestroyContext(Pointer context);

    // ── Inference ─────────────────────────────────────────────────────────────

    int sdxRun(Pointer context, TensorView[] inputs, int numInputs,
               TensorView[] outputs, int numOutputs, RunOptions options);

    // ── Introspection ─────────────────────────────────────────────────────────

    int sdxGetNumInputs(Pointer context);
    int sdxGetNumOutputs(Pointer context);
    Pointer sdxGetInputName(Pointer context, int inputIndex);

    // ── Lifecycle hints ───────────────────────────────────────────────────────

    int sdxMarkInputVariable(Pointer context, int inputIndex);
    int sdxMarkInputPlaceholder(Pointer context, int inputIndex);
    int sdxFreezeShapes(Pointer context);
    int sdxGetPlanPhase(Pointer context);
    int sdxGetExecutionCount(Pointer context);

    // ── Telemetry and error ───────────────────────────────────────────────────

    Pointer sdxGetLastError(Pointer runtime);
    int sdxGetExecutionReport(Pointer context, ExecutionReport outReport);

    // ── JNA Structure types ───────────────────────────────────────────────────

    class RuntimeOptions extends Structure {
        public int struct_size;

        public RuntimeOptions() { struct_size = size(); }

        @Override
        protected List<String> getFieldOrder() {
            return Collections.singletonList("struct_size");
        }
    }

    class ModelOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_backend;
        public int allow_runtime_jit;
        public int gpu_target;

        public ModelOptions() { struct_size = size(); }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_backend",
                    "allow_runtime_jit", "gpu_target");
        }
    }

    class RunOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_signature;
        public int gpu_target;

        public RunOptions() {
            struct_size = size();
            strict_signature = 1;
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_signature", "gpu_target");
        }
    }

    /**
     * Descriptor for a single tensor passed to or from {@code sdxRun}.
     * Mirrors {@code SdxTensorView} in {@code dsp_runtime_c.h}.
     */
    class TensorView extends Structure {
        public Pointer data;
        public Pointer shape;
        public int rank;
        public int dtype;
        public long bytes;
        public int device_type;
        public int device_id;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("data", "shape", "rank", "dtype",
                    "bytes", "device_type", "device_id");
        }
    }

    /**
     * Execution telemetry returned by {@code sdxGetExecutionReport}.
     * Mirrors {@code SdxExecutionReport} in {@code dsp_runtime_c.h}.
     */
    class ExecutionReport extends Structure {
        public int struct_size;
        public int requested_backend;
        public int applied_backend;
        public int status_code;
        public int used_fallback;
        public long execution_time_ns;
        public int requested_gpu_target;
        public int applied_gpu_target;
        public int plan_phase;
        public int execution_count;

        public ExecutionReport() { struct_size = size(); }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "requested_backend", "applied_backend",
                    "status_code", "used_fallback", "execution_time_ns",
                    "requested_gpu_target", "applied_gpu_target",
                    "plan_phase", "execution_count");
        }
    }
}

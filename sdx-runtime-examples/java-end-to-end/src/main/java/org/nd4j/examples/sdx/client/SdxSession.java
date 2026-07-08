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

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Inference session for a single SDX model, analogous to
 * {@code OrtSession} in the ONNX Runtime Java API.
 *
 * <p>Obtain an instance from {@link SdxEnvironment#openSession}.  Use
 * try-with-resources to ensure native resources are released:
 * <pre>{@code
 * try (SdxSession session = env.openSession("model.sdz", new String[]{"probs"})) {
 *
 *     // Mark per-request inputs so the DSP can pick the right staging strategy.
 *     session.markInputPlaceholder("x");
 *
 *     // Named input map — the session reorders to positional slots automatically.
 *     Map<String, SdxTensor> inputs = new LinkedHashMap<>();
 *     inputs.put("x",  SdxTensor.fromArray(data, new long[]{2, 4}));
 *     inputs.put("w1", SdxTensor.fromArray(w1,   new long[]{4, 8}));
 *     // ... remaining weights ...
 *
 *     Map<String, long[]> outputShapes = Collections.singletonMap("probs", new long[]{2, 3});
 *     Map<String, SdxTensor> outputs = session.run(inputs, outputShapes);
 *     float[] probs = outputs.get("probs").toFloatArray();
 * }
 * }</pre>
 *
 * <h3>External input contract</h3>
 * <p>The SDX plan's external inputs cover the model's <em>constants</em>,
 * <em>variables</em> (weights), and <em>placeholders</em>, discovered at
 * session-open time via {@code sdxGetInputName}.  The caller must supply a
 * value for every input; the weights typically come from the model provider
 * (e.g. extracted from the {@code .sdz} at training time) while the
 * placeholder ({@code "x"}) is the per-request batch input.
 *
 * <p>{@link #run(Map, Map)} accepts a {@code Map<String, SdxTensor>} keyed by
 * input name and reorders the values to match the plan's positional contract
 * automatically — exactly as ONNX Runtime's named-input API works.
 *
 * <h3>Shape freezing</h3>
 * <p>After warmup, call {@link #freezeShapes()} to enter the DSP replay fast
 * path.  This corresponds to the transition from {@code SLOT_BY_SLOT} to
 * {@code REPLAYING} in the execution plan lifecycle.
 */
public final class SdxSession implements AutoCloseable {

    private final SdxEnvironment env;
    private final Pointer model;
    private final Pointer context;

    /** Input names in positional order, discovered once at open time. */
    private final String[] inputNames;
    /** Number of output slots the context provides. */
    private final int numOutputs;

    private volatile boolean closed = false;

    SdxSession(SdxEnvironment env, Pointer model, Pointer context) {
        this.env     = env;
        this.model   = model;
        this.context = context;

        // Discover the plan's external input contract once, at open time.
        int n = env.api.sdxGetNumInputs(context);
        this.inputNames = new String[n];
        for (int i = 0; i < n; i++) {
            Pointer p = env.api.sdxGetInputName(context, i);
            this.inputNames[i] = p.getString(0);
        }
        this.numOutputs = env.api.sdxGetNumOutputs(context);
    }

    /**
     * Returns the input names the plan expects, in positional order.
     * Analogous to {@code OrtSession.getInputNames()} in ONNX Runtime.
     *
     * @return unmodifiable list of input names
     */
    public List<String> inputNames() {
        List<String> names = new ArrayList<>(inputNames.length);
        Collections.addAll(names, inputNames);
        return Collections.unmodifiableList(names);
    }

    /**
     * Returns the number of output slots in the plan.
     *
     * @return output count
     */
    public int numOutputs() {
        return numOutputs;
    }

    /**
     * Marks an input as a <em>placeholder</em> (batch input whose shape may
     * change between calls).  Pass the <em>name</em> rather than an index;
     * the session resolves the position internally.
     *
     * @param name the input name (e.g. {@code "x"})
     * @throws IllegalArgumentException if the name is not found
     */
    public void markInputPlaceholder(String name) {
        int idx = indexOfInput(name);
        checkStatus(env.api.sdxMarkInputPlaceholder(context, idx),
                "sdxMarkInputPlaceholder(" + name + ")");
    }

    /**
     * Marks an input as a <em>variable</em> (fixed-shape weight that may
     * update its values between calls but keeps the same shape).
     *
     * @param name the input name (e.g. {@code "w1"})
     * @throws IllegalArgumentException if the name is not found
     */
    public void markInputVariable(String name) {
        int idx = indexOfInput(name);
        checkStatus(env.api.sdxMarkInputVariable(context, idx),
                "sdxMarkInputVariable(" + name + ")");
    }

    /**
     * Signals that all input shapes are stable and the DSP runtime should
     * transition to the replay fast path.
     *
     * <p>Call this after the warmup runs (typically 2–3 executions) once you
     * are confident the input shapes will not change.  After freezing, the
     * plan phase transitions from {@code SLOT_BY_SLOT} through
     * {@code SHAPES_FROZEN} to {@code REPLAYING}.
     */
    public void freezeShapes() {
        checkStatus(env.api.sdxFreezeShapes(context), "sdxFreezeShapes");
    }

    /**
     * Returns the current DSP plan phase ordinal.
     * 0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING, 3=REPLAY_BLOCKED.
     *
     * @return the plan phase ordinal
     */
    public int planPhase() {
        return env.api.sdxGetPlanPhase(context);
    }

    /**
     * Returns the human-readable name of the current DSP plan phase.
     *
     * @return phase name string
     */
    public String planPhaseName() {
        int phase = planPhase();
        return phase >= 0 && phase < SdxExecutionReport.PLAN_PHASE_NAMES.length
                ? SdxExecutionReport.PLAN_PHASE_NAMES[phase] : "? (" + phase + ")";
    }

    /** Returns the total number of successful {@code sdxRun} calls made so far. */
    public int executionCount() {
        return env.api.sdxGetExecutionCount(context);
    }

    /**
     * Returns the execution telemetry from the most recent {@link #run} call,
     * analogous to the profiling result types in the ONNX Runtime API.
     *
     * @return an {@link SdxExecutionReport} POJO with backend, phase, and timing info
     */
    public SdxExecutionReport getExecutionReport() {
        SdxAbi.ExecutionReport raw = new SdxAbi.ExecutionReport();
        raw.write();
        checkStatus(env.api.sdxGetExecutionReport(context, raw), "sdxGetExecutionReport");
        raw.read();
        return new SdxExecutionReport(
                raw.status_code, raw.requested_backend, raw.applied_backend,
                raw.used_fallback, raw.execution_time_ns,
                raw.requested_gpu_target, raw.applied_gpu_target,
                raw.plan_phase, raw.execution_count);
    }

    /**
     * Runs inference with named inputs and pre-allocated output tensors,
     * analogous to the pinned-output overload of
     * {@code OrtSession.run(inputs, requestedOutputs, pinnedOutputs)} in
     * the ONNX Runtime Java API.
     *
     * <p>The {@code inputs} map keys are input names; the session maps them to
     * positional slots automatically by querying the plan's discovered input
     * contract.  Every name returned by {@link #inputNames()} must have a
     * corresponding entry in {@code inputs}.
     *
     * <p>The {@code outputSpecs} map provides pre-allocated caller-owned
     * {@link SdxTensor} objects; the runtime writes results into their backing
     * buffers.  The map is returned as-is after the call for chaining.
     *
     * @param inputs      named input tensors (all plan inputs must be present)
     * @param outputSpecs pre-allocated output tensors keyed by output name
     * @return the same {@code outputSpecs} map, now populated with results
     * @throws IllegalArgumentException if a required input name is missing
     * @throws IllegalStateException    if the runtime returns a non-OK status
     */
    public Map<String, SdxTensor> run(Map<String, SdxTensor> inputs,
                                      Map<String, SdxTensor> outputSpecs) {
        checkNotClosed();

        // ── Build the positional input array ──────────────────────────────────
        // JNA requires a contiguous Structure array.
        SdxAbi.TensorView[] inputViews =
                (SdxAbi.TensorView[]) new SdxAbi.TensorView().toArray(inputNames.length);

        // Native Memory objects must stay alive until sdxRun returns.
        Memory[] inputKeepAlive = new Memory[inputNames.length * 2];

        for (int i = 0; i < inputNames.length; i++) {
            String name = inputNames[i];
            SdxTensor tensor = inputs.get(name);
            if (tensor == null) {
                throw new IllegalArgumentException(
                        "Missing required input '" + name + "' (plan slot " + i + ")");
            }
            fillInputView(inputViews[i], tensor, inputKeepAlive, i * 2);
        }

        // ── Build the output array ────────────────────────────────────────────
        // Each output tensor provides a caller-allocated Memory buffer;
        // the runtime writes directly into it.
        List<Map.Entry<String, SdxTensor>> outputList =
                new ArrayList<>(outputSpecs.entrySet());
        SdxAbi.TensorView[] outputViews =
                (SdxAbi.TensorView[]) new SdxAbi.TensorView().toArray(outputList.size());
        // Native Memory for output data buffers (results written here by native side).
        Memory[] outputDataMem  = new Memory[outputList.size()];
        Memory[] outputShapeMem = new Memory[outputList.size()];

        for (int i = 0; i < outputList.size(); i++) {
            SdxTensor tensor = outputList.get(i).getValue();
            long[] shape = tensor.shape();

            Memory dataMem  = new Memory(tensor.byteCount());
            Memory shapeMem = new Memory((long) shape.length * Long.BYTES);
            shapeMem.write(0, shape, 0, shape.length);

            outputViews[i].data        = dataMem;
            outputViews[i].shape       = shapeMem;
            outputViews[i].rank        = tensor.rank();
            outputViews[i].dtype       = SdxTensor.SDX_DTYPE_FLOAT;
            outputViews[i].bytes       = tensor.byteCount();
            outputViews[i].device_type = SdxAbi.SDX_DEVICE_HOST;
            outputViews[i].device_id   = -1;
            outputViews[i].write();

            outputDataMem[i]  = dataMem;
            outputShapeMem[i] = shapeMem;
        }

        // ── Execute ───────────────────────────────────────────────────────────
        SdxAbi.RunOptions runOpts = new SdxAbi.RunOptions();
        runOpts.write();
        int status = env.api.sdxRun(context,
                inputViews, inputNames.length,
                outputViews, outputList.size(),
                runOpts);
        if (status != SdxAbi.SDX_STATUS_OK) {
            throw new IllegalStateException("sdxRun failed: status=" + status
                    + ", error=" + env.lastError());
        }

        // ── Copy native output buffers back into the caller's SdxTensors ─────
        for (int i = 0; i < outputList.size(); i++) {
            SdxTensor tensor = outputList.get(i).getValue();
            int n = tensor.numElements();
            float[] result = new float[n];
            outputDataMem[i].read(0, result, 0, n);
            tensor.update(result);
        }

        return outputSpecs;
    }

    /**
     * Convenience factory: allocates fresh output {@link SdxTensor}s from the
     * given shapes, runs inference, and returns the populated results.
     *
     * <p>Analogous to the basic {@code OrtSession.run(inputs)} overload in
     * ONNX Runtime that auto-allocates output tensors.  Named separately from
     * {@link #run(Map, Map)} to avoid the Java type-erasure clash between
     * {@code Map<String, SdxTensor>} and {@code Map<String, long[]>}.
     *
     * <p>Use {@link #run(Map, Map)} when you want to pre-allocate output
     * buffers for reuse across calls.
     *
     * @param inputs       named input tensors
     * @param outputShapes map from output name to its expected shape
     * @return a new map from output name to populated result tensor
     */
    public Map<String, SdxTensor> runWithShapes(Map<String, SdxTensor> inputs,
                                                 Map<String, long[]> outputShapes) {
        Map<String, SdxTensor> outputSpecs = new LinkedHashMap<>();
        for (Map.Entry<String, long[]> e : outputShapes.entrySet()) {
            long[] shape = e.getValue();
            int n = 1;
            for (long d : shape) {
                n = Math.toIntExact(Math.multiplyExact(n, d));
            }
            outputSpecs.put(e.getKey(), SdxTensor.fromArray(new float[n], shape));
        }
        return run(inputs, outputSpecs);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            env.api.sdxDestroyContext(context);
            env.api.sdxUnloadModel(model);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * Fills a TensorView JNA structure from an SdxTensor for use as an input.
     * Copies the float data into a fresh native Memory allocation to guarantee
     * contiguous C-order layout.
     */
    private static void fillInputView(SdxAbi.TensorView view, SdxTensor tensor,
                                      Memory[] keepAlive, int baseIdx) {
        float[] data = tensor.toFloatArray();
        Memory dataMem = new Memory(tensor.byteCount());
        dataMem.write(0, data, 0, data.length);

        long[] shape = tensor.shape();
        Memory shapeMem = new Memory((long) shape.length * Long.BYTES);
        shapeMem.write(0, shape, 0, shape.length);

        keepAlive[baseIdx]     = dataMem;
        keepAlive[baseIdx + 1] = shapeMem;

        view.data        = dataMem;
        view.shape       = shapeMem;
        view.rank        = shape.length;
        view.dtype       = SdxTensor.SDX_DTYPE_FLOAT;
        view.bytes       = tensor.byteCount();
        view.device_type = SdxAbi.SDX_DEVICE_HOST;
        view.device_id   = -1;
        view.write();
    }

    private int indexOfInput(String name) {
        for (int i = 0; i < inputNames.length; i++) {
            if (inputNames[i].equals(name)) return i;
        }
        throw new IllegalArgumentException("Input '" + name + "' not found in plan");
    }

    private void checkStatus(int status, String op) {
        if (status != SdxAbi.SDX_STATUS_OK) {
            throw new IllegalStateException(op + " failed: status=" + status
                    + ", error=" + env.lastError());
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("SdxSession has been closed");
        }
    }
}

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

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.ptr.PointerByReference;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

/**
 * Global environment for the SDX runtime — analogous to
 * {@code OrtEnvironment} in the ONNX Runtime Java API.
 *
 * <p>There is typically one {@code SdxEnvironment} per process.  It owns the
 * native {@code SdxRuntime*} pointer, manages library loading, and is the
 * factory for {@link SdxSession} objects.
 *
 * <p>Usage (try-with-resources mirrors the ONNX Runtime pattern exactly):
 * <pre>{@code
 * try (SdxEnvironment env = SdxEnvironment.create()) {
 *     try (SdxSession session = env.openSession("model.sdz", new String[]{"probs"})) {
 *         Map<String, SdxTensor> inputs = ...;
 *         Map<String, SdxTensor> outputs = session.run(inputs);
 *     }
 * }
 * }</pre>
 *
 * <p><b>Library resolution order</b> (first match wins):
 * <ol>
 *   <li>System property {@code sdx.library} or env-var {@code SDX_RUNTIME_LIBRARY}
 *       — explicit path to the shared library file.</li>
 *   <li>{@code SDX_RUNTIME_HOME/lib} from an unpacked SDK ZIP, preferring the
 *       JVM-free standalone runtime ({@code libsdx_cpu.so} / {@code libsdx_cuda.so}).</li>
 *   <li>The backend library JavaCPP already extracted for this JVM process —
 *       dlopen re-uses the already-loaded image so the ABI is shared.</li>
 * </ol>
 */
public final class SdxEnvironment implements AutoCloseable {

    // Visible inside the package for SdxSession
    final SdxAbi api;
    final Pointer runtime;

    private volatile boolean closed = false;

    private SdxEnvironment(SdxAbi api, Pointer runtime) {
        this.api     = api;
        this.runtime = runtime;
    }

    /**
     * Creates the environment by auto-detecting the runtime library.
     * Equivalent to {@code OrtEnvironment.getEnvironment()} in ONNX Runtime.
     *
     * @return a new environment; close it when the process is done with inference
     * @throws IllegalStateException if the library cannot be found or the runtime cannot be created
     */
    public static SdxEnvironment create() {
        return create(resolveRuntimeLibrary());
    }

    /**
     * Creates the environment using an explicit library path.
     *
     * @param libraryPath absolute path to the SDX shared library
     * @return a new environment
     */
    public static SdxEnvironment create(String libraryPath) {
        SdxAbi api = Native.load(libraryPath, SdxAbi.class);

        SdxAbi.RuntimeOptions opts = new SdxAbi.RuntimeOptions();
        opts.write();

        PointerByReference outRuntime = new PointerByReference();
        int status = api.sdxCreateRuntime(opts, outRuntime);
        if (status != SdxAbi.SDX_STATUS_OK) {
            throw new IllegalStateException("sdxCreateRuntime failed: status=" + status);
        }
        return new SdxEnvironment(api, outRuntime.getValue());
    }

    /**
     * Returns the SDX C ABI version reported by the loaded library.
     * Useful for compatibility checks in production code.
     *
     * @return the ABI version integer
     */
    public int abiVersion() {
        return api.sdxGetRuntimeAbiVersion();
    }

    /**
     * Opens an inference session for the given model bundle.
     * Analogous to {@code new OrtSession(env, modelPath, options)} in ONNX Runtime.
     *
     * <p>The session must be closed after use (try-with-resources recommended).
     * The environment must remain open for the lifetime of all sessions it creates.
     *
     * @param bundlePath      path to the {@code .sdz} model bundle file
     * @param requestedOutputs the output names to request from the model (e.g. {@code "probs"})
     * @return a new session ready for inference
     * @throws IllegalStateException if loading or context creation fails
     */
    public SdxSession openSession(String bundlePath, String[] requestedOutputs) {
        return openSession(bundlePath, requestedOutputs, new SdxSessionOptions());
    }

    /**
     * Opens an inference session with explicit backend options.
     *
     * @param bundlePath       path to the {@code .sdz} bundle
     * @param requestedOutputs the output names to request
     * @param options          session configuration (backend, GPU target, etc.)
     * @return a new session
     */
    public SdxSession openSession(String bundlePath, String[] requestedOutputs,
                                  SdxSessionOptions options) {
        checkNotClosed();

        SdxAbi.ModelOptions modelOpts = new SdxAbi.ModelOptions();
        modelOpts.backend          = options.backend();
        modelOpts.strict_backend   = options.strictBackend() ? 1 : 0;
        modelOpts.allow_runtime_jit = options.allowRuntimeJit() ? 1 : 0;
        modelOpts.gpu_target       = options.gpuTarget();
        modelOpts.write();

        PointerByReference outModel = new PointerByReference();
        int status = api.sdxLoadBundle(runtime, bundlePath, modelOpts, outModel);
        if (status != SdxAbi.SDX_STATUS_OK) {
            throw new IllegalStateException("sdxLoadBundle failed: status=" + status
                    + ", error=" + lastError());
        }
        Pointer model = outModel.getValue();

        PointerByReference outContext = new PointerByReference();
        status = api.sdxCreateContext(model, new StringArray(requestedOutputs),
                requestedOutputs.length, outContext);
        if (status != SdxAbi.SDX_STATUS_OK) {
            api.sdxUnloadModel(model);
            throw new IllegalStateException("sdxCreateContext failed: status=" + status
                    + ", error=" + lastError());
        }

        return new SdxSession(this, model, outContext.getValue());
    }

    /**
     * Returns the last error message from the native runtime, or an empty string
     * if none is available.
     */
    public String lastError() {
        Pointer p = api.sdxGetLastError(runtime);
        return p == null ? "" : p.getString(0);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            api.sdxDestroyRuntime(runtime);
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("SdxEnvironment has been closed");
        }
    }

    // ── Library resolution ────────────────────────────────────────────────────

    private static String resolveRuntimeLibrary() {
        // 1) Explicit override via system property or environment variable.
        String explicit = System.getProperty("sdx.library",
                System.getenv("SDX_RUNTIME_LIBRARY"));
        if (explicit != null && !explicit.isEmpty() && new File(explicit).exists()) {
            return explicit;
        }

        // 2) Prefer the JVM-free standalone runtime from an unpacked SDK ZIP.
        boolean isCuda;
        try {
            isCuda = org.nd4j.linalg.factory.Nd4j.getBackend()
                    .getClass().getName().toLowerCase().contains("cuda");
        } catch (Throwable ignored) {
            isCuda = false;
        }
        String[] preferred = isCuda
                ? new String[]{"libsdx_cuda.so", "libnd4jcuda.so"}
                : new String[]{"libsdx_cpu.so", "libsdx_cpu.dylib",
                               "libnd4jcpu.so", "libnd4jcpu.dylib"};

        String sdkHome = System.getenv("SDX_RUNTIME_HOME");
        if (sdkHome != null && !sdkHome.isEmpty()) {
            for (String name : preferred) {
                File candidate = new File(sdkHome, "lib/" + name);
                if (candidate.exists()) {
                    return candidate.getAbsolutePath();
                }
            }
        }

        // 3) Fall back to the JavaCPP-extracted backend library already
        //    loaded in this process (monolithic build exports the same ABI).
        try {
            File cacheDir = org.bytedeco.javacpp.Loader.getCacheDir();
            if (cacheDir != null && cacheDir.isDirectory()) {
                for (String name : preferred) {
                    try (Stream<Path> walk = Files.walk(cacheDir.toPath())) {
                        Path hit = walk.filter(p -> p.getFileName().toString().equals(name))
                                .findFirst().orElse(null);
                        if (hit != null) {
                            return hit.toAbsolutePath().toString();
                        }
                    }
                }
            }
        } catch (Exception ignored) {
            // JavaCPP not on classpath or cache unavailable; fall through.
        }

        throw new IllegalStateException(
                "SDX runtime library not found. "
                + "Set SDX_RUNTIME_LIBRARY=/path/to/libsdx_cpu.so "
                + "or SDX_RUNTIME_HOME=/path/to/unpacked-sdk");
    }
}

/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.examples.samediff.advanced.execution;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanDiskCache;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.TritonCacheManager;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * DSP Disk Cache and Triton Compilation Cache Example.
 *
 * The Dynamic Shape Plan (DSP) subsystem compiles SameDiff graphs into
 * optimized execution plans. These plans can be cached to disk to avoid
 * recompilation on subsequent runs. This example demonstrates:
 *
 * <h3>1. DSP Disk Cache ({@link DspPlanDiskCache})</h3>
 * Compiled plans are serialized and stored on disk, keyed by a FNV-1a hash
 * of the plan structure. On subsequent runs, the plan is loaded from disk
 * instead of recompiling — saving seconds of startup time for large models.
 *
 * <pre>
 * Default cache directory: ~/.kompile/cache/dsp/dsp_plan_cache/
 * Override directory:      ~/.kompile/cache/dsp/dsp_plan_override/
 * File format: dsp_&lt;16hex&gt;.bin + dsp_&lt;16hex&gt;.meta
 * Index files: dsp_model_&lt;16hex&gt;.idx
 * </pre>
 *
 * <h3>2. Triton Cache ({@link TritonCacheManager})</h3>
 * On GPU, Triton-compiled kernels are cached independently. The cache bundle
 * can be exported/imported across machines with the same GPU architecture.
 *
 * <h3>3. TritonCacheTool CLI</h3>
 * <pre>
 * java -cp ... org.nd4j.tools.TritonCacheTool export output.tkcache
 * java -cp ... org.nd4j.tools.TritonCacheTool import bundle.tkcache
 * java -cp ... org.nd4j.tools.TritonCacheTool import bundle.tkcache --skip-arch-check
 * java -cp ... org.nd4j.tools.TritonCacheTool inspect bundle.tkcache
 * </pre>
 *
 * <h3>System Properties:</h3>
 * <pre>
 * # Disk cache control
 * -Dnd4j.dsp.planCache.diskEnabled=true          # Enable/disable disk caching (default: true)
 * -Dnd4j.dsp.planCache.diskDir=/path/to/cache     # Custom cache directory
 * -Dnd4j.dsp.planCache.overrideDir=/path/override  # Override cache (checked first)
 * -Dnd4j.dsp.planCache.forceRecompile=true         # Force recompilation, ignore cache
 *
 * # Environment variable alternative
 * ND4J_DSP_PLAN_CACHE_DIR=/path/to/cache
 * </pre>
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.advanced.execution.DSPDiskCacheAndTritonExample"
 */
public class DSPDiskCacheAndTritonExample {
    private static final Logger log = LoggerFactory.getLogger(DSPDiskCacheAndTritonExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build and execute a SameDiff graph (triggers DSP compilation)
        // =====================================================================
        log.info("=== 1. Building and executing SameDiff graph ===");

        SameDiff sd = SameDiff.create();

        // Simple two-layer network
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w1 = sd.var("w1", Nd4j.randn(64, 32).muli(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(32));
        SDVariable w2 = sd.var("w2", Nd4j.randn(32, 10).muli(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(10));

        SDVariable z1 = input.mmul(w1).add(b1);
        SDVariable a1 = sd.nn.relu(z1, 0);
        SDVariable z2 = a1.mmul(w2).add(b2);
        SDVariable output = sd.nn.softmax("output", z2, -1);

        // Execute — this triggers DSP plan compilation and caching
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(8, 64));
        INDArray result = sd.outputSingle(ph, "output");
        log.info("Output shape: {}", Arrays.toString(result.shape()));

        // =====================================================================
        // 2. Inspect DSP Disk Cache State
        // =====================================================================
        log.info("\n=== 2. DSP Disk Cache State ===");

        // Check if disk caching is enabled
        boolean cacheEnabled = DspPlanDiskCache.isEnabled();
        log.info("Disk cache enabled: {}", cacheEnabled);

        // Check if force recompile is on (ignores cache)
        boolean forceRecompile = DspPlanDiskCache.isForceRecompile();
        log.info("Force recompile: {}", forceRecompile);

        // Get cache directories
        String cacheDir = DspPlanDiskCache.getCacheDir();
        String overrideDir = DspPlanDiskCache.getOverrideDir();
        log.info("Cache directory: {}", cacheDir);
        log.info("Override directory: {}", overrideDir);
        log.info("  (Override dir is checked first; use it to deploy pre-compiled plans)");

        // =====================================================================
        // 3. List and Inspect Cached Plans
        // =====================================================================
        log.info("\n=== 3. Cached Plans ===");

        // List all cached plan hashes
        List<String> cachedHashes = DspPlanDiskCache.listCachedHashes();
        log.info("Number of cached plans: {}", cachedHashes.size());
        for (int i = 0; i < Math.min(5, cachedHashes.size()); i++) {
            log.info("  Cached plan hash: {}", cachedHashes.get(i));
        }
        if (cachedHashes.size() > 5) {
            log.info("  ... and {} more", cachedHashes.size() - 5);
        }

        // =====================================================================
        // 4. Cache Operations — Load, Store, Invalidate
        // =====================================================================
        log.info("\n=== 4. Cache Operations ===");

        // Compute a model identity hash — identifies THIS model's structure
        // regardless of weight values. Uses requested outputs + external input
        // keys + slot count as the identity.
        Set<String> requestedOutputs = new HashSet<>(Arrays.asList("output"));
        String[] externalInputKeys = new String[]{"input", "w1", "b1", "w2", "b2"};
        int numSlots = 10;  // example slot count

        long modelIdentityHash = DspPlanDiskCache.computeModelIdentityHash(
                requestedOutputs, externalInputKeys, numSlots);
        log.info("Model identity hash: 0x{}", Long.toHexString(modelIdentityHash));

        // Try loading by model identity (returns null if not cached)
        byte[] cachedPlan = DspPlanDiskCache.tryLoadByModelIdentity(
                requestedOutputs, externalInputKeys, numSlots);
        log.info("Cached plan found by model identity: {}", cachedPlan != null);

        // Check if a specific hash exists in cache
        long exampleHash = 0x1234567890ABCDEFL;
        boolean exists = DspPlanDiskCache.exists(exampleHash);
        log.info("Example hash exists: {}", exists);

        // Try loading by exact structure hash
        byte[] planBytes = DspPlanDiskCache.tryLoadByHash(exampleHash);
        log.info("Plan bytes loaded: {}", planBytes != null ? planBytes.length + " bytes" : "null");

        // Store a plan to cache (normally done automatically by DSP)
        // DspPlanDiskCache.store(structureHash, planBytes, numSlots, numExtInputs, numOutputs, outputSet);
        log.info("  store(hash, bytes, numSlots, numExtInputs, numOutputs, outputSet)");
        log.info("  — stores compiled plan binary and metadata to disk");

        // Store a model identity index (maps model identity to structure hash)
        // DspPlanDiskCache.storeModelIdentityIndex(requestedOutputs, externalInputKeys, numSlots, structureHash);
        log.info("  storeModelIdentityIndex(outputs, inputKeys, numSlots, hash)");
        log.info("  — creates index for fast lookup by model structure");

        // Invalidate a cached plan
        // DspPlanDiskCache.invalidate(structureHash);
        log.info("  invalidate(hash) — removes a specific cached plan");

        // Ensure a cache directory exists
        boolean dirReady = DspPlanDiskCache.ensureCacheDir(cacheDir);
        log.info("Cache directory ready: {}", dirReady);

        // =====================================================================
        // 5. Triton Availability and Cache Management
        // =====================================================================
        log.info("\n=== 5. Triton Cache Management ===");

        // Check if Triton JIT compilation is available on this machine
        boolean tritonAvailable = TritonCacheManager.isTritonAvailable();
        log.info("Triton available: {}", tritonAvailable);

        if (tritonAvailable) {
            // Export all cached Triton kernels to a portable bundle.
            // This bundle contains compiled PTX/AMDGCN and can be imported
            // on another machine with the same GPU architecture.
            // NOTE: exportCache throws IllegalStateException on CPU (no GPU kernel cache).
            // Wrap in try-catch so the example runs on CPU without crashing.
            Path exportPath = Paths.get(System.getProperty("java.io.tmpdir"), "triton-cache.tkcache");
            try {
                int numExported = TritonCacheManager.exportCache(exportPath);
                log.info("Exported {} Triton kernel entries to {}", numExported, exportPath);

                // Inspect a bundle without importing
                String manifest = TritonCacheManager.inspectBundle(exportPath);
                log.info("Bundle manifest (JSON): {}", manifest);

                // Import a bundle (validates GPU architecture compatibility)
                int numImported = TritonCacheManager.importCache(exportPath);
                log.info("Imported {} Triton kernel entries", numImported);

                // Import with architecture validation skipped (cross-platform deployment)
                // Returns -2 if architecture is incompatible (when validateArch=true)
                // int imported = TritonCacheManager.importCache(exportPath, false);
                log.info("  importCache(path, false) — skip architecture validation");
            } catch (IllegalStateException | IllegalArgumentException e) {
                // On CPU (or when no Triton kernels have been compiled yet),
                // exportCache returns error code -1 and throws. This is expected.
                log.info("Triton cache export not available on this platform: {}", e.getMessage());
                log.info("  (Triton support is compiled in, but kernel cache requires GPU execution first)");
                log.info("  On CUDA with compiled Triton kernels, this would export a portable bundle.");
                log.info("  TritonCacheManager.exportCache(Path) → int (entries exported)");
                log.info("  TritonCacheManager.importCache(Path) → int (entries imported)");
                log.info("  TritonCacheManager.importCache(Path, validateArch) → int (-2 = arch mismatch)");
                log.info("  TritonCacheManager.inspectBundle(Path) → String (JSON manifest)");
            }
        } else {
            log.info("Triton not available — showing API reference only");
            log.info("  TritonCacheManager.exportCache(Path) → int (entries exported)");
            log.info("  TritonCacheManager.importCache(Path) → int (entries imported)");
            log.info("  TritonCacheManager.importCache(Path, validateArch) → int (-2 = arch mismatch)");
            log.info("  TritonCacheManager.inspectBundle(Path) → String (JSON manifest)");
        }

        // =====================================================================
        // 6. Triton Compilation Configuration Reference
        // =====================================================================
        log.info("\n=== 6. Triton Compilation Configuration ===");
        log.info("Triton compilation is configured via TritonEnvironmentConfig (interface).");
        log.info("Access through Nd4j.getEnvironment() or system properties:");
        log.info("");
        log.info("  Build settings:");
        log.info("    tritonBuildThreads (default: 8)         — parallel compilation threads");
        log.info("    tritonCacheEnabled (default: true)      — enable kernel caching");
        log.info("    tritonAlwaysCompile (default: false)    — skip cache, always recompile");
        log.info("    tritonCompileAll (default: false)       — compile all segments upfront");
        log.info("");
        log.info("  Kernel quality:");
        log.info("    tritonNumWarps                          — GPU warp count");
        log.info("    tritonNumStages                         — pipeline stages");
        log.info("    tritonNumCTAs                           — cooperative thread arrays");
        log.info("    tritonMaxNreg                           — max registers per thread");
        log.info("    tritonTf32Enabled                       — TensorFloat-32 on Ampere+");
        log.info("");
        log.info("  Fusion control:");
        log.info("    tritonSectionFusion (default: true)     — fuse adjacent kernel sections");
        log.info("    tritonFusionScoring (default: true)     — score-based fusion decisions");
        log.info("    tritonFusionMinScore (default: 5.0)     — minimum fusion benefit score");
        log.info("    tritonFuseIdentityShapes (default: true)— fuse shape-preserving ops");
        log.info("    tritonFuseCastChains (default: true)    — merge cascaded casts");
        log.info("    tritonFuseAttentionNeighborhoods        — fuse around attention patterns");
        log.info("    tritonFusedMatmul                       — fuse bias+activation into matmul");
        log.info("");
        log.info("  Graph capture:");
        log.info("    tritonGraphCapture (default: true)      — CUDA/HIP graph capture");
        log.info("    tritonCooperativeLaunch (default: false) — cooperative kernel launch");
        log.info("    tritonCaptureMinExec                    — min executions before capture");
        log.info("    tritonForceRecapture                    — force graph recapture");
        log.info("    tritonInvalidateOnPlanFree              — clear cache on plan free");
        log.info("");
        log.info("  Debugging:");
        log.info("    tritonDumpSections                      — dump section analysis");
        log.info("    tritonDumpArgs                          — dump kernel arguments");
        log.info("    tritonDumpGraphDot                      — export graph as DOT");
        log.info("    tritonKernelDump                        — dump generated kernels");
        log.info("    tritonVerifyKernels                     — verify kernel correctness");
        log.info("    tritonVerifyFullSnapshot                — full output verification");
        log.info("    tritonSkipKernels                       — skip specific kernels");
        log.info("");
        log.info("  Op filtering:");
        log.info("    tritonExcludeOps                        — comma-separated ops to exclude");
        log.info("    tritonIncludeTypes                      — restrict to specific dtypes");
        log.info("    tritonOverrideArch                      — override GPU architecture");

        // =====================================================================
        // 7. TritonCacheTool CLI Reference
        // =====================================================================
        log.info("\n=== 7. TritonCacheTool CLI ===");
        log.info("The TritonCacheTool (org.nd4j.tools.TritonCacheTool) provides");
        log.info("command-line management of the Triton kernel cache:");
        log.info("");
        log.info("  # Export all cached kernels to a portable bundle");
        log.info("  java -cp ... org.nd4j.tools.TritonCacheTool export output.tkcache");
        log.info("");
        log.info("  # Import a bundle (validates GPU architecture)");
        log.info("  java -cp ... org.nd4j.tools.TritonCacheTool import bundle.tkcache");
        log.info("");
        log.info("  # Import without architecture check (cross-machine deployment)");
        log.info("  java -cp ... org.nd4j.tools.TritonCacheTool import bundle.tkcache --skip-arch-check");
        log.info("");
        log.info("  # Inspect a bundle (show JSON manifest without importing)");
        log.info("  java -cp ... org.nd4j.tools.TritonCacheTool inspect bundle.tkcache");
        log.info("");
        log.info("Workflow:");
        log.info("  1. Develop/compile on a build machine with GPU");
        log.info("  2. Export the Triton cache bundle");
        log.info("  3. Ship the bundle with your application");
        log.info("  4. Import on the target machine — zero compilation at startup");

        // =====================================================================
        // 8. DSP Plan Binary Format Reference
        // =====================================================================
        log.info("\n=== 8. DSP Plan Binary Format ===");
        log.info("Compiled DSP plans are serialized in a compact binary format:");
        log.info("  Magic:   0x44535031 ('DSP1')");
        log.info("  Version: 6 (current DSP_VERSION)");
        log.info("  Encoding: little-endian");
        log.info("");
        log.info("Plan structure:");
        log.info("  DynamicShapeSlot[]  — per-op descriptors with integer-indexed wiring");
        log.info("  int totalOutputSlots — for flat INDArray[] allocation");
        log.info("  int[][] releaseAtStep — pre-computed liveness schedule");
        log.info("  OpContext[] opContextPool — pre-allocated, one per slot");
        log.info("  String[] externalInputKeys — constants/variables/placeholders by index");
        log.info("  byte[] externalInputSourceTypes — CONSTANT, VARIABLE, PLACEHOLDER");
        log.info("  Map<String,Integer> outputNameToSlotIndex — O(1) output collection");
        log.info("");
        log.info("Key DynamicShapePlan methods:");
        log.info("  plan.serialize()              — compact binary for C++ executor");
        log.info("  plan.computeStructureHash()   — FNV-1a hash for caching");
        log.info("  plan.getSummary()             — text summary");
        log.info("  plan.getDetailedSummary()     — detailed summary");
        log.info("  plan.toDot()                  — GraphViz DOT format");
        log.info("  plan.getSegments()            — execution segments");
        log.info("  plan.getParallelGroups()      — parallel execution groups");
        log.info("  plan.assignDevices()          — auto device assignment");
        log.info("  plan.assignDevices(budgets)   — with memory budgets per device");

        // =====================================================================
        // 9. Cache Priority and Lookup Order
        // =====================================================================
        log.info("\n=== 9. Cache Lookup Order ===");
        log.info("When DSP needs a compiled plan, it searches in this order:");
        log.info("  1. In-memory plan cache (fastest, per-SameDiff instance)");
        log.info("  2. Override directory (~/.kompile/cache/dsp/dsp_plan_override/)");
        log.info("     — For deploying pre-compiled plans (CI/CD artifacts)");
        log.info("  3. Standard cache directory (~/.kompile/cache/dsp/dsp_plan_cache/)");
        log.info("     — Populated automatically during first execution");
        log.info("  4. Compile from scratch (slowest, results are cached)");
        log.info("");
        log.info("The override directory is useful for production deployments:");
        log.info("  - Pre-compile plans on a build server");
        log.info("  - Package the cache directory with your application");
        log.info("  - Set -Dnd4j.dsp.planCache.overrideDir=/app/precompiled/");
        log.info("  - Zero compilation latency at startup");

        log.info("\n**************** DSP Disk Cache and Triton Example finished ********************");
    }
}

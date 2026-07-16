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

import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.KernelSelectionConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * DSP Backends, Kernel Selection, and Graph Execution Modes Example.
 *
 * The DSP subsystem supports 19 execution backends, from simple slot-by-slot
 * dispatch to advanced Triton JIT compilation. This example covers:
 *
 * <h3>1. GraphExecutionMode — All 19 Backend Modes</h3>
 * <pre>
 * GPU Backends:
 *   AUTO(0)         — automatically selects best available backend
 *   CUDA_GRAPHS(2)  — CUDA graph capture and replay (eliminates launch overhead)
 *   NVRTC_JIT(3)    — generates CUDA C++ source, compiles via NVRTC at runtime
 *   PTX_JIT(4)      — generates PTX assembly text directly (fastest compile path)
 *   TRITON(5)       — full Triton pipeline: MLIR IR → TTIR → TTGIR → LLVM → PTX/AMDGCN/SPIR-V
 *   HIP_GRAPHS(9)   — AMD ROCm graph capture and replay
 *   LEVEL_ZERO(10)  — Intel oneAPI Level Zero (Xe GPUs)
 *   VULKAN(11)      — Vulkan compute shaders (cross-platform GPU)
 *   METAL(12)       — Apple Metal Performance Shaders
 *
 * CPU Backends:
 *   SLOT_BY_SLOT(1) — per-op dispatch, correctness baseline
 *   MLX(6)          — Apple Silicon via MLX framework (Metal GPU offload)
 *   ARM_HYBRID(7)   — MLIR CPU + optional Vulkan GPU on ARM (NEON/SVE)
 *   OPENVINO(15)    — Intel OpenVINO graph optimization
 *
 * Mobile / Edge:
 *   NNAPI(8)        — Android Neural Networks API (NPU/DSP/GPU)
 *   HEXAGON(14)     — Qualcomm Hexagon HVX via MLIR → hexagon-mlir
 *   TPU(13)         — Google TPU (XLA compilation)
 *
 * Debug / Special:
 *   EMULATED_REPLAY(17)    — slot-by-slot with full graph lifecycle instrumentation
 *   SHAPE_INFERENCE_ONLY(18) — shape propagation without kernel launches
 * </pre>
 *
 * <h3>2. KernelSelectionConfig — Per-Op Engine Selection</h3>
 * Controls which compute engine (CPU, CUDA, oneDNN, MPS, etc.) runs each op.
 *
 * <h3>3. MLIR Compilation Pipeline</h3>
 * How the MLIR-based backends (CPU, ARM, Hexagon) compile ops to native code.
 *
 * <h3>4. Graph Optimization Passes</h3>
 * 24 optimization passes that run before backend compilation.
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.advanced.execution.DSPBackendsAndKernelSelectionExample"
 */
public class DSPBackendsAndKernelSelectionExample {
    private static final Logger log = LoggerFactory.getLogger(DSPBackendsAndKernelSelectionExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. GraphExecutionMode — All 19 Modes
        // =====================================================================
        log.info("=== 1. GraphExecutionMode — All Backend Modes ===");

        // List all available modes
        for (GraphExecutionMode mode : GraphExecutionMode.values()) {
            log.info("  {} (nativeCode={}) — requiresGraphBackend={}, isSlotBySlot={}",
                    mode.name(), mode.getNativeCode(),
                    mode.requiresGraphBackend(), mode.isSlotBySlot());
        }

        log.info("\nBackend selection priority (AUTO mode on CUDA):");
        log.info("  1. Triton     — highest quality fused kernels (MLIR pipeline)");
        log.info("  2. NVRTC      — CUDA C++ runtime compilation");
        log.info("  3. PTX        — PTX assembly text generation (fastest compile)");
        log.info("  4. CUDA Graphs — graph capture and replay");
        log.info("  5. Slot-by-slot — per-op dispatch fallback");

        // =====================================================================
        // 2. Setting Execution Mode on SameDiff
        // =====================================================================
        log.info("\n=== 2. Setting Execution Mode ===");

        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w = sd.var("w", Nd4j.randn(64, 32).muli(0.01));
        SDVariable b = sd.var("b", Nd4j.zeros(32));
        SDVariable z = input.mmul(w).add(b);
        SDVariable output = sd.nn.softmax("output", z, -1);

        // Set graph execution mode
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        log.info("Current mode: {}", sd.getGraphExecutionMode());

        // DSP auto-compile settings
        sd.setDspAutoCompileEnabled(true);   // Auto-compile on first execution
        log.info("DSP auto-compile: {}", sd.isDspAutoCompileEnabled());

        sd.setDspNativeAutoCompileEnabled(true);  // Use native C++ executor
        log.info("DSP native auto-compile: {}", sd.isDspNativeAutoCompileEnabled());

        // Triton fallback — if Triton isn't available, fall back to AUTO
        sd.setDspFallbackToAutoIfTritonUnavailable(true);
        log.info("Triton fallback to AUTO: {}", sd.isDspFallbackToAutoIfTritonUnavailable());

        // Execute with default AUTO mode
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(8, 64));
        INDArray result = sd.outputSingle(ph, "output");
        log.info("Output shape: {}", Arrays.toString(result.shape()));

        // =====================================================================
        // 2b. LIVE segment backend inspection
        // =====================================================================
        // Run a few more times to let the plan progress past warmup.
        // On CPU, AUTO maps to EMULATED_REPLAY — segmentBackendName() returns "emulated_replay".
        // On CUDA without Triton: returns "cuda_graphs".
        // On CUDA with Triton: returns "triton", "nvrtc", or "cuda_graphs" per segment.
        for (int i = 0; i < 3; i++) {
            ph.put("input", Nd4j.randn(8, 64));
            sd.outputSingle(ph, "output");
        }
        DspHandle dspLive = sd.dsp();
        if (dspLive.isCompiled()) {
            log.info("\nLive segment backend inspection (after {} executions):", dspLive.executeCount());
            int numSegs = dspLive.numSegments();
            for (int s = 0; s < numSegs; s++) {
                log.info("  Segment {}: backend='{}', phase={}, replays={}, capturable={}",
                        s, dspLive.segmentBackendName(s),
                        dspLive.segmentExecutionPhase(s),
                        dspLive.segmentReplayCount(s),
                        dspLive.isSegmentCapturable(s));
            }
            log.info("  planPhase={} (0=SLOT_BY_SLOT, 1=SHAPES_FROZEN, 2=REPLAYING)",
                    dspLive.planPhase());
        } else {
            log.info("Plan not yet compiled — run sd.output() first");
        }

        // =====================================================================
        // 3. Explicit DSP Compilation
        // =====================================================================
        log.info("\n=== 3. Explicit DSP Compilation ===");

        SameDiff sd2 = SameDiff.create();
        SDVariable in2 = sd2.placeHolder("in", DataType.FLOAT, -1, 48);
        SDVariable w2var = sd2.var("w2", Nd4j.randn(48, 24).muli(0.01));
        SDVariable out2 = sd2.nn.relu(in2.mmul(w2var).rename("matmul_out"), 0).rename("out");

        // Compile with default mode
        GraphExecutionMode compiledMode = sd2.compileNativeDynamicShapePlan("out");
        log.info("Compiled to mode: {}", compiledMode);

        // Compile with a specific compilation mode
        // DspCompilationMode controls optimization level:
        //   REDUCE_OVERHEAD — minimize JIT startup time
        //   SPLIT_STITCH    — balanced Triton compilation
        //   MAX_AUTOTUNE    — maximum throughput, best kernel selection
        compiledMode = sd2.compileNativeDynamicShapePlan(DspCompilationMode.REDUCE_OVERHEAD, "out");
        log.info("Compiled with REDUCE_OVERHEAD: {}", compiledMode);

        // Compile targeting a specific execution mode with fallback
        compiledMode = sd2.compileNativeDynamicShapePlan(
                Collections.singleton("out"),           // requested outputs
                GraphExecutionMode.TRITON,              // preferred mode
                true                                    // fallback to AUTO if unavailable
        );
        log.info("Compiled targeting TRITON (with fallback): {}", compiledMode);

        // =====================================================================
        // 4. Shape Freezing (CUDA Graph Capture)
        // =====================================================================
        log.info("\n=== 4. Shape Freezing ===");

        log.info("Shape freezing locks input shapes, enabling:");
        log.info("  - CUDA graph capture and replay");
        log.info("  - Zero-copy output via stable GPU pointers");
        log.info("  - Maximum inference throughput");
        log.info("");
        log.info("  sd.setDspShapesFrozen(true)   — freeze shapes");
        log.info("  sd.isDspShapesFrozen()         — check frozen state");
        log.info("");
        log.info("WARNING: After freezing, all inputs must have the same shape.");
        log.info("Passing a different shape will cause an error, not recompilation.");

        // =====================================================================
        // 4b. BenchmarkConfig Production Presets
        // =====================================================================
        log.info("\n=== 4b. BenchmarkConfig Production Presets ===");
        log.info("BenchmarkConfig provides tested, named execution configurations.");
        log.info("Use BenchmarkConfigApplier.apply(config) to activate one.");
        log.info("");

        // CPU presets — all work on CPU machines; GPU presets require CUDA.
        BenchmarkConfig cpuSlotBySlot = BenchmarkConfig.cpuSlotBySlot();
        log.info("cpuSlotBySlot: mode={}, tritonSectionFusion={}",
                cpuSlotBySlot.getExecutionMode(), cpuSlotBySlot.isTritonSectionFusion());

        BenchmarkConfig cpuCascade = BenchmarkConfig.cpuCascade();
        log.info("cpuCascade:    mode={}, tritonSectionFusion={}",
                cpuCascade.getExecutionMode(), cpuCascade.isTritonSectionFusion());

        BenchmarkConfig cpuOpenVino = BenchmarkConfig.cpuOpenVino();
        log.info("cpuOpenVino:   mode={}, tritonSectionFusion={}",
                cpuOpenVino.getExecutionMode(), cpuOpenVino.isTritonSectionFusion());

        BenchmarkConfig optimal = BenchmarkConfig.optimal();
        log.info("optimal:       mode={}, tritonSectionFusion={}, tritonIncludeTypes={}",
                optimal.getExecutionMode(), optimal.isTritonSectionFusion(),
                optimal.getTritonIncludeTypes());

        // Apply the cpuCascade preset to the Nd4j environment.
        // This wires all DSP system properties to match the preset,
        // exactly as run-benchmark.sh --config SLOT_BY_SLOT would do.
        BenchmarkConfigApplier.apply(cpuCascade);
        log.info("Applied cpuCascade preset via BenchmarkConfigApplier.apply()");
        log.info("  (sets nd4j.dsp.* system properties to match the preset config)");

        // Reset to AUTO so the rest of the example runs in default mode.
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        log.info("Reset to optimal preset.");
        log.info("");
        log.info("CRITICAL RULE — why hardcoding SLOT_BY_SLOT as a workaround is BANNED:");
        log.info("  Calling sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT)");
        log.info("  prevents DSP from advancing through its lifecycle:");
        log.info("    SLOT_BY_SLOT -> SHAPES_FROZEN -> REPLAYING");
        log.info("  This destroys CUDA graph capture/replay and causes 5-10x perf regression.");
        log.info("  SLOT_BY_SLOT mode is ONLY legal via BenchmarkConfigApplier as a measurement");
        log.info("  baseline. If a bug appears during capture/replay, FIX the capture/replay code");
        log.info("  — do NOT force SLOT_BY_SLOT to hide the bug.");

        // =====================================================================
        // 5. KernelSelectionConfig — Per-Op Engine Selection
        // =====================================================================
        log.info("\n=== 5. KernelSelectionConfig ===");

        // Default configuration — auto-tune with fastest strategy
        KernelSelectionConfig defaultConfig = KernelSelectionConfig.defaultConfig();
        log.info("Default config:");
        log.info("  Strategy: {}", defaultConfig.getStrategy());
        log.info("  Auto-tune: {}", defaultConfig.isAutoTuneEnabled());
        log.info("  Warmup runs: {}", defaultConfig.getWarmupRuns());
        log.info("  Benchmark runs: {}", defaultConfig.getBenchmarkRuns());
        log.info("  Preferred engine: {}", defaultConfig.getPreferredEngine());
        log.info("  Fallback to native: {}", defaultConfig.isAllowFallbackToNative());

        // From environment variables
        // SD_KERNEL_STRATEGY, SD_KERNEL_AUTOTUNE, SD_KERNEL_PREFERRED_ENGINE, etc.
        KernelSelectionConfig envConfig = KernelSelectionConfig.fromEnvironment();
        log.info("\nEnvironment config:");
        log.info("  Strategy: {}", envConfig.getStrategy());

        // Custom configuration with builder
        KernelSelectionConfig customConfig = KernelSelectionConfig.builder()
                .strategy(KernelSelectionConfig.Strategy.FASTEST)
                .autoTuneEnabled(true)
                .warmupRuns(3)
                .benchmarkRuns(10)
                .maxBenchmarkTimeMs(2000)
                .preferredEngine(KernelSelectionConfig.Engine.CPU)
                .allowFallbackToNative(true)
                .performanceThreshold(1.5)   // 1.5x faster required to switch engines
                .verbose(true)
                .build();
        log.info("\nCustom config: strategy={}, engine={}, verbose={}",
                customConfig.getStrategy(), customConfig.getPreferredEngine(),
                customConfig.isVerbose());

        // Disable specific engines
        customConfig.disableEngine(KernelSelectionConfig.Engine.VULKAN);
        customConfig.disableEngine(KernelSelectionConfig.Engine.OPENCL);
        log.info("Disabled engines: VULKAN={}, OPENCL={}",
                customConfig.isEngineDisabled(KernelSelectionConfig.Engine.VULKAN),
                customConfig.isEngineDisabled(KernelSelectionConfig.Engine.OPENCL));

        // Set engine preference priority
        customConfig.setEnginePreference(KernelSelectionConfig.Engine.CUDA, 1);
        customConfig.setEnginePreference(KernelSelectionConfig.Engine.ONEDNN, 2);
        customConfig.setEnginePreference(KernelSelectionConfig.Engine.CPU, 3);

        log.info("\nAll available engines:");
        for (KernelSelectionConfig.Engine engine : KernelSelectionConfig.Engine.values()) {
            log.info("  {} — priority: {}", engine.name(),
                    customConfig.getEnginePreference(engine));
        }

        log.info("\nSelection strategies:");
        log.info("  FASTEST          — benchmark and select fastest engine");
        log.info("  FIRST_AVAILABLE  — use first available engine in priority order");
        log.info("  ROUND_ROBIN      — distribute ops across engines");
        log.info("  MEMORY_OPTIMIZED — minimize memory transfers");
        log.info("  POWER_OPTIMIZED  — minimize power consumption (mobile)");

        // =====================================================================
        // 6. Execution Phases and Plan Lifecycle
        // =====================================================================
        log.info("\n=== 6. Execution Phases and Plan Lifecycle ===");

        log.info("ExecutionPhase — per-segment compilation state:");
        for (ExecutionPhase phase : ExecutionPhase.values()) {
            log.info("  {} (code={})", phase.name(), phase.getNativeCode());
        }

        log.info("\nPlanPhase — overall plan compilation state:");
        for (PlanPhase phase : PlanPhase.values()) {
            log.info("  {} (code={})", phase.name(), phase.getNativeCode());
        }

        log.info("\nDspCompilationMode — optimization level:");
        for (DspCompilationMode mode : DspCompilationMode.values()) {
            log.info("  {}", mode.name());
        }

        log.info("\nPlan lifecycle:");
        log.info("  1. SLOT_BY_SLOT    — initial per-op execution, collecting shape info");
        log.info("  2. SHAPES_FROZEN   — shapes locked, segments can be compiled");
        log.info("  3. REPLAYING       — compiled segments are being replayed");
        log.info("  4. REPLAY_BLOCKED  — replay blocked (shape change or error)");

        // =====================================================================
        // 7. Backend Architecture — GPU Compilation Pipelines
        // =====================================================================
        log.info("\n=== 7. Backend Architecture ===");

        log.info("GPU JIT Compilation Pipelines (3 tiers):");
        log.info("");
        log.info("  Triton Backend (highest quality):");
        log.info("    SameDiff ops → Op categorization → TritonIRBuilder → MLIR IR");
        log.info("    → TTIR → TTGIR → LLVM IR → PTX (NVIDIA) / AMDGCN (AMD) / SPIR-V (Intel)");
        log.info("    - Max 768 ops per segment (auto-splits into sub-kernels)");
        log.info("    - Fusion scoring: evaluates register pressure, shared memory,");
        log.info("      intermediate memory savings, launch overhead (15us per kernel)");
        log.info("    - Auto-tune: tries up to 3 configurations, settles on fastest");
        log.info("    - Disk cache: ~/.kompile/cache/ with FNV-1a hash keys");
        log.info("");
        log.info("  NVRTC Backend (medium quality):");
        log.info("    SameDiff ops → NvrtcKernelBuilder → CUDA C++ source");
        log.info("    → nvrtcCompileProgram → PTX → cuModuleLoad");
        log.info("    - Faster compilation than Triton");
        log.info("    - Good for simple elementwise and reduction patterns");
        log.info("");
        log.info("  PTX Backend (fastest compile):");
        log.info("    SameDiff ops → PTX assembly text templates → cuModuleLoadDataEx");
        log.info("    - No compilation step — direct text template expansion");
        log.info("    - Fastest compile time, lowest optimization quality");

        log.info("\nCPU Compilation Backends:");
        log.info("");
        log.info("  MLIR CPU Backend:");
        log.info("    SameDiff ops → CpuIRBuilder → MLIR (memref/scf/arith/linalg)");
        log.info("    → LLVM JIT → native code");
        log.info("    - Supports 40+ op categories (binary, unary, matmul, conv, etc.)");
        log.info("");
        log.info("  OneDNN Backend:");
        log.info("    SameDiff ops → Intel oneDNN Graph API → optimized x86 kernels");
        log.info("    - Best for Intel CPUs (AVX-512, AMX)");
        log.info("");
        log.info("  ARM Compute Library (ACL) Backend:");
        log.info("    SameDiff ops → ARM Compute Library → NEON/SVE SIMD kernels");
        log.info("    - Optimized for ARM Cortex-A (mobile, embedded)");
        log.info("");
        log.info("  Apple MLX Backend:");
        log.info("    SameDiff ops → MlxIRBuilder → mlx::core::array lazy graphs");
        log.info("    → Metal GPU via MLX runtime");
        log.info("    - Includes LLM-specific fusions: RoPE, fused_rms_norm_swiglu,");
        log.info("      fused_bias_dropout_residual, fused attention");

        log.info("\nMobile / Edge Backends:");
        log.info("");
        log.info("  ARM Hybrid Backend:");
        log.info("    MLIR CPU (NEON/SVE) + optional Vulkan GPU offload");
        log.info("    - Best for ARM devices with integrated GPU");
        log.info("");
        log.info("  NNAPI Backend:");
        log.info("    SameDiff ops → Android NNAPI → device NPU/DSP/GPU");
        log.info("    - Delegates to hardware accelerators on Android");
        log.info("");
        log.info("  Hexagon Backend:");
        log.info("    SameDiff ops → HexagonIRBuilder → MLIR → hexagon-mlir runtime");
        log.info("    - Targets Qualcomm Hexagon HVX DSP via DMA + TCM staging");

        // =====================================================================
        // 8. Op Classification System
        // =====================================================================
        log.info("\n=== 8. Op Classification (OpCategoryTable) ===");
        log.info("All backends classify ops into 17 categories for dispatch:");
        log.info("  BINARY_ELEMENTWISE  — add, mul, sub, div, pow, max, min, ...");
        log.info("  UNARY_ELEMENTWISE   — relu, sigmoid, tanh, exp, log, sqrt, ...");
        log.info("  COMPARISON          — eq, neq, gt, lt, gte, lte");
        log.info("  LOGICAL             — and, or, not, xor");
        log.info("  TERNARY             — where, clamp");
        log.info("  IDENTITY            — reshape, squeeze, unsqueeze, permute");
        log.info("  MATMUL              — matmul, gemm, batch_matmul");
        log.info("  REDUCTION           — reduce_sum, reduce_mean, reduce_max, ...");
        log.info("  NORMALIZATION       — layer_norm, batch_norm, rms_norm");
        log.info("  CAST                — dtype conversions (float→half, etc.)");
        log.info("  FUSED_ATTENTION     — DotProductAttentionV2");
        log.info("  SHAPE_MANIPULATION  — concat, split, slice, gather, scatter");
        log.info("  DATA_MOVEMENT       — copy, transpose");
        log.info("  CONSTANT_GENERATION — zeros, ones, range, fill");
        log.info("  CONVOLUTION         — conv1d, conv2d, conv3d");
        log.info("  ROPE                — rotary position encoding");
        log.info("  FUSED_LLM           — SwiGLU, fused_rms_norm_swiglu");

        // =====================================================================
        // 9. Graph Optimization Passes (24 Passes)
        // =====================================================================
        log.info("\n=== 9. Graph Optimization Passes ===");
        log.info("GraphOptimizer applies 24 optimization passes before execution:");
        log.info("");
        log.info("  Cleanup:");
        log.info("    1.  UnusedFunctionOptimizations      — dead op elimination");
        log.info("    2.  ConstantFunctionOptimizations     — constant folding");
        log.info("    3.  IdentityFunctionOptimizations     — identity removal");
        log.info("    4.  RedundancyEliminationOptimizations — redundant op removal");
        log.info("");
        log.info("  Algebraic:");
        log.info("    5.  BroadcastEliminationOptimizations — redundant broadcasts");
        log.info("    6.  ReorderingOptimizations           — reassociate constants");
        log.info("    7.  AlgebraicOptimizations            — x+0→x, x*1→x, x*0→0");
        log.info("    8.  PeepholeOptimizations             — idempotent ops, inverse pairs");
        log.info("    9.  ArithmeticChainOptimizations      — add(add(x,c1),c2)→add(x,c1+c2)");
        log.info("    10. StrengthReductionOptimizations    — pow(x,2)→square, div→mul");
        log.info("");
        log.info("  Shape:");
        log.info("    11. ConcatSplitOptimizations          — flatten nested concat");
        log.info("    12. SelectWhereOptimizations          — constant condition");
        log.info("    13. ShapeFunctionOptimizations        — shape op simplification");
        log.info("    14. CommonSubexpressionElimination    — deduplicate identical ops");
        log.info("");
        log.info("  Fusion:");
        log.info("    15. AttentionFusionOptimizations      — Q*K^T→scale→softmax→V fusion");
        log.info("    16. HorizontalFusionOptimizations     — parallel matmuls sharing input");
        log.info("    17. MatMulChainOptimizations          — fold constants, absorb transpose");
        log.info("    18. ActivationFusionOptimizations     — sigmoid(x)*x→swish, SwiGLU");
        log.info("    19. NormalizationFusionOptimizations  — RMSNorm detection");
        log.info("    20. GatedDeltaNetFusionOptimizations  — GDN pattern fusion");
        log.info("    21. LinearFusionOptimizations         — linear function fusion");
        log.info("");
        log.info("  Memory & Backend:");
        log.info("    22. RematerializationOptimizations    — recompute cheap ops to save memory");
        log.info("    23. QuantizationOptimizations         — FP16/BF16 weight quantization");
        log.info("    24. CuDNNFunctionOptimizations        — cuDNN-specific fusions");
        log.info("");
        log.info("Control:");
        log.info("  -Dnd4j.optimizer.maxIterations=3      — max pass iterations");
        log.info("  -Dnd4j.optimizer.logApplied            — log each applied optimization");
        log.info("  -Dnd4j.optimizer.skip=PassA,PassB      — skip named passes");
        log.info("  -Dnd4j.optimizer.fp16 (default: true)  — FP16 weight quantization");
        log.info("  -Dnd4j.optimizer.bf16=true             — BF16 instead of FP16");

        // =====================================================================
        // 10. Environment Variables Reference
        // =====================================================================
        log.info("\n=== 10. Kernel Selection Environment Variables ===");
        log.info("  SD_KERNEL_STRATEGY=FASTEST             — selection strategy");
        log.info("  SD_KERNEL_AUTOTUNE=true                — enable auto-tuning");
        log.info("  SD_KERNEL_WARMUP_RUNS=2                — warmup iterations");
        log.info("  SD_KERNEL_BENCHMARK_RUNS=5             — benchmark iterations");
        log.info("  SD_KERNEL_CACHE_PATH=/path/cache       — benchmark result cache");
        log.info("  SD_KERNEL_PREFERRED_ENGINE=CUDA         — preferred compute engine");
        log.info("  SD_KERNEL_FORCE_ENGINE=ONEDNN           — force specific engine");
        log.info("  SD_KERNEL_VERBOSE=true                 — verbose logging");
        log.info("  SD_KERNEL_DISABLE_ENGINES=VULKAN,OPENCL — disable engines");

        log.info("\n**************** DSP Backends and Kernel Selection Example finished ********************");
    }
}

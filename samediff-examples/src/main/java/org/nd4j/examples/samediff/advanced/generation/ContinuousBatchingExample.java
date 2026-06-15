/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.batch.BatchGenerationState;
import org.eclipse.deeplearning4j.llm.batch.ChunkedPrefillEngine;
import org.eclipse.deeplearning4j.llm.batch.ContinuousBatchScheduler;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Continuous Batching for LLM Serving — API Reference
 *
 * Continuous batching (also called "iteration-level scheduling" or "in-flight batching")
 * is the key technique for efficient multi-user LLM serving.  Instead of processing one
 * request at a time or waiting for all sequences in a static batch to finish before
 * starting new ones, the scheduler dynamically inserts new requests into free KV-cache
 * slots as soon as any existing sequence completes.
 *
 * This maximises GPU utilisation because the GPU is never idle waiting for stragglers
 * in a fixed batch.
 *
 * <h3>Slot lifecycle:</h3>
 * <pre>
 *   FREE → PREFILL → DECODE → DONE → FREE
 * </pre>
 *
 * <h3>Key classes:</h3>
 * <ul>
 *   <li><b>ContinuousBatchScheduler</b> — queues requests, assigns KV-cache slots,
 *       drives the PREFILL/DECODE/DONE state machine.</li>
 *   <li><b>BatchGenerationState</b>    — snapshot of one slot's state at a given step
 *       (token IDs so far, KV-cache position, sampling config, finish status).</li>
 *   <li><b>ChunkedPrefillEngine</b>   — handles very long prompts by processing the
 *       prompt in fixed-size chunks so peak memory is bounded.</li>
 *   <li><b>GenerationPipeline</b>     — high-level pipeline that uses
 *       ContinuousBatchScheduler internally when batchSize &gt; 1.</li>
 * </ul>
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java \
 *     -Dexec.mainClass="org.nd4j.examples.samediff.advanced.generation.ContinuousBatchingExample"
 */
public class ContinuousBatchingExample {

    public static void main(String[] args) throws Exception {

        System.out.println("=== Continuous Batching for LLM Serving — API Reference ===");
        System.out.println();
        System.out.println("Continuous batching maximises GPU utilisation by filling empty");
        System.out.println("KV-cache slots immediately when any sequence finishes.");
        System.out.println("All requests see lower average latency compared to static batching.");
        System.out.println();

        // ============================================================
        // 1. LOAD THE MODEL
        // ============================================================
        System.out.println("=== 1. Load model ===");
        System.out.println();

        // AutoModel.fromPretrained() detects the format (GGUF, ONNX, SafeTensors, SDZ)
        // automatically. Replace the path with a real file to run this example.
        //
        //   SameDiff sd = AutoModel.fromPretrained("/models/llama3-8b.gguf");

        System.out.println("  // Load the model — any supported format:");
        System.out.println("  SameDiff sd = AutoModel.fromPretrained(\"/models/llama3-8b.gguf\");");
        System.out.println();

        // ============================================================
        // 2. CREATE CONTINUOUSBATCHSCHEDULER
        // ============================================================
        System.out.println("=== 2. ContinuousBatchScheduler — creation ===");
        System.out.println();

        // ContinuousBatchScheduler is the central coordinator.
        // It maintains a fixed pool of KV-cache slots and a waiting queue for incoming requests.
        //
        // Key configuration:
        //   maxBatchSize — maximum number of sequences running concurrently (= number of KV slots)
        //   maxSeqLen    — maximum token length per sequence (controls KV-cache allocation)
        //   prefillChunkSize — max tokens per prefill chunk (for ChunkedPrefillEngine integration)

        System.out.println("  ContinuousBatchScheduler scheduler = ContinuousBatchScheduler.builder()");
        System.out.println("      .model(sd)");
        System.out.println("      .maxBatchSize(8)          // up to 8 concurrent sequences");
        System.out.println("      .maxSeqLen(2048)          // max tokens per sequence");
        System.out.println("      .prefillChunkSize(512)    // process long prompts in 512-token chunks");
        System.out.println("      .build();");
        System.out.println();

        // Conceptual object — replace sd with a real SameDiff model to run:
        //
        // ContinuousBatchScheduler scheduler = ContinuousBatchScheduler.builder()
        //         .model(sd)
        //         .maxBatchSize(8)
        //         .maxSeqLen(2048)
        //         .prefillChunkSize(512)
        //         .build();

        System.out.println("  After construction:");
        System.out.println("    scheduler.getFreeSlotCount()   — initially equals maxBatchSize (8)");
        System.out.println("    scheduler.getActiveCount()     — 0 (no requests yet)");
        System.out.println("    scheduler.getWaitingCount()    — 0 (queue is empty)");
        System.out.println();

        // ============================================================
        // 3. SUBMIT REQUESTS
        // ============================================================
        System.out.println("=== 3. Submitting requests ===");
        System.out.println();

        // Each submit() call enqueues a generation request.
        // The scheduler assigns it to a free slot immediately if one is available,
        // or queues it until a slot becomes free.
        //
        // submit() returns a CompletableFuture<GenerationResult> that resolves
        // when the sequence finishes (EOS or maxNewTokens reached).

        System.out.println("  // Token IDs from tokenizer.encode(text, true).getIds():");
        System.out.println("  int[] prompt1 = {1, 15043, 29892, 3186, 29991};  // \"Hello, world!\"");
        System.out.println("  int[] prompt2 = {1, 5816,  338,  3271,  29973};  // \"What is Paris?\"");
        System.out.println("  int[] prompt3 = {1, 12008,  263,  2498,  26413}; // \"Write a story\"");
        System.out.println();
        System.out.println("  // Each request can have its own sampling config:");
        System.out.println("  SamplingConfig config1 = SamplingConfig.greedy();");
        System.out.println("  SamplingConfig config2 = SamplingConfig.topP(0.9);");
        System.out.println("  SamplingConfig config3 = SamplingConfig.creative();");
        System.out.println();
        System.out.println("  // Submit returns a CompletableFuture resolved when generation finishes:");
        System.out.println("  CompletableFuture<GenerationResult> future1 = scheduler.submit(prompt1, config1);");
        System.out.println("  CompletableFuture<GenerationResult> future2 = scheduler.submit(prompt2, config2);");
        System.out.println("  CompletableFuture<GenerationResult> future3 = scheduler.submit(prompt3, config3);");
        System.out.println();
        System.out.println("  // Alternatively, submit with a simple maxNewTokens cap:");
        System.out.println("  scheduler.submit(prompt1, SamplingConfig.greedy(), 100 /*maxNewTokens*/);");
        System.out.println();

        // ============================================================
        // 4. POLLING FOR COMPLETED RESULTS
        // ============================================================
        System.out.println("=== 4. Polling for completed results ===");
        System.out.println();

        // poll() returns all sequences that finished in the most recent step.
        // Call it after each scheduler.step() to collect results.
        // Results include the full generated text, token IDs, and throughput metrics.

        System.out.println("  // The serving loop drives one decode step at a time:");
        System.out.println("  while (scheduler.hasWork()) {");
        System.out.println("      scheduler.step();                           // one decode step for all active slots");
        System.out.println("      List<GenerationResult> done = scheduler.poll(); // collect newly finished requests");
        System.out.println("      for (GenerationResult result : done) {");
        System.out.println("          System.out.println(\"Finished: '\" + result.getText() + \"'\");");
        System.out.println("          System.out.println(\"  tokens: \" + result.getGeneratedTokenCount()");
        System.out.println("              + \", speed: \" + result.getTokensPerSecond() + \" tok/s\");");
        System.out.println("      }");
        System.out.println("  }");
        System.out.println();
        System.out.println("  // Alternatively, block on the CompletableFuture returned by submit():");
        System.out.println("  GenerationResult r1 = future1.get();");
        System.out.println("  System.out.println(r1.getText());");
        System.out.println();

        // ============================================================
        // 5. BATCHGENERATIONSTATE — PER-SLOT INSPECTION
        // ============================================================
        System.out.println("=== 5. BatchGenerationState — per-slot state inspection ===");
        System.out.println();

        // BatchGenerationState represents the instantaneous state of one active slot.
        // It is produced by the scheduler at each step and consumed by the decode engine.
        // You rarely use this class directly unless writing a custom serving loop.

        System.out.println("  // During a custom serving loop you can inspect slot states:");
        System.out.println("  List<BatchGenerationState> activeStates = scheduler.getActiveStates();");
        System.out.println("  for (BatchGenerationState state : activeStates) {");
        System.out.println("      System.out.println(\"Slot \" + state.getSlotIndex() + \":\");");
        System.out.println("      System.out.println(\"  prompt tokens:    \" + state.getPromptLength());");
        System.out.println("      System.out.println(\"  generated so far: \" + state.getGeneratedLength());");
        System.out.println("      System.out.println(\"  KV cache pos:     \" + state.getKvCachePosition());");
        System.out.println("      System.out.println(\"  phase:            \" + state.getPhase()); // PREFILL or DECODE");
        System.out.println("      System.out.println(\"  sampling config:  \" + state.getSamplingConfig());");
        System.out.println("  }");
        System.out.println();

        // BatchGenerationState.Phase:
        System.out.println("  BatchGenerationState.Phase values:");
        System.out.println("    PREFILL — processing the prompt (compute-bound, sequential)");
        System.out.println("    DECODE  — generating new tokens (memory-bound, batched)");
        System.out.println("    DONE    — sequence finished; slot will be freed on next step");
        System.out.println();

        // ============================================================
        // 6. CHUNKEDPREFILLENGINE FOR LONG PROMPTS
        // ============================================================
        System.out.println("=== 6. ChunkedPrefillEngine — handling long prompts ===");
        System.out.println();

        // For prompts of thousands of tokens, processing the entire prompt in a single
        // forward pass can require enormous amounts of memory (O(seq_len^2) attention).
        // ChunkedPrefillEngine splits the prompt into chunks of at most prefillChunkSize
        // tokens and processes each chunk in turn, accumulating KV cache entries.
        //
        // This bounds peak memory to O(chunk_size * seq_len) instead of O(seq_len^2),
        // enabling very long context windows on memory-constrained hardware.
        //
        // ChunkedPrefillEngine is integrated into ContinuousBatchScheduler automatically
        // when prefillChunkSize is set.  You can also use it standalone.

        System.out.println("  // Standalone ChunkedPrefillEngine usage:");
        System.out.println("  ChunkedPrefillEngine prefillEngine = ChunkedPrefillEngine.builder()");
        System.out.println("      .model(sd)");
        System.out.println("      .chunkSize(512)           // process 512 tokens at a time");
        System.out.println("      .build();");
        System.out.println();
        System.out.println("  // A 4096-token prompt is processed in 8 chunks of 512:");
        System.out.println("  int[] longPromptTokenIds = /* ... 4096 token IDs ... */;");
        System.out.println("  ChunkedPrefillEngine.PrefillResult prefillResult =");
        System.out.println("      prefillEngine.prefill(longPromptTokenIds);");
        System.out.println();
        System.out.println("  // prefillResult contains:");
        System.out.println("  //   getKvCache()     — the fully built KV cache, ready for decode");
        System.out.println("  //   getLogits()      — logits for the last prompt token");
        System.out.println("  //   getChunksProcessed() — how many chunks were needed");
        System.out.println();
        System.out.println("  // When integrated with ContinuousBatchScheduler, the scheduler");
        System.out.println("  // interleaves prefill chunks with decode steps for other sequences,");
        System.out.println("  // so decode throughput is not blocked during long prompt ingestion.");
        System.out.println();

        // ============================================================
        // 7. SCHEDULER STATISTICS
        // ============================================================
        System.out.println("=== 7. Scheduler statistics ===");
        System.out.println();

        System.out.println("  // Query live scheduler state:");
        System.out.println("  int  active    = scheduler.getActiveCount();    // slots currently generating");
        System.out.println("  int  waiting   = scheduler.getWaitingCount();   // requests in queue");
        System.out.println("  int  freeSlots = scheduler.getFreeSlotCount();  // available slots");
        System.out.println("  long completed = scheduler.getTotalCompleted();  // all-time finished count");
        System.out.println();
        System.out.println("  // Per-step throughput metrics:");
        System.out.println("  ContinuousBatchScheduler.Stats stats = scheduler.getStats();");
        System.out.println("  double throughput = stats.getMeanThroughputTokensPerSec();");
        System.out.println("  double latencyP50 = stats.getLatencyPercentileMs(50);");
        System.out.println("  double latencyP99 = stats.getLatencyPercentileMs(99);");
        System.out.println("  double gpuUtil    = stats.getMeanBatchOccupancy();  // fraction of maxBatchSize used");
        System.out.println();

        // ============================================================
        // 8. GENERATIONPIPELINE WITH BATCH SUPPORT
        // ============================================================
        System.out.println("=== 8. GenerationPipeline with batch support ===");
        System.out.println();

        // GenerationPipeline is the high-level wrapper. When batchSize > 1 it uses
        // ContinuousBatchScheduler internally, so most users don't need to use
        // the scheduler directly.

        System.out.println("  // GenerationPipeline uses continuous batching when batchSize > 1:");
        System.out.println("  GenerationPipelineConfig config = GenerationPipelineConfig.builder()");
        System.out.println("      .decoder(sd)");
        System.out.println("      .tokenizer(tokenizer)");
        System.out.println("      .batchSize(8)             // enable continuous batching with 8 slots");
        System.out.println("      .maxSeqLen(2048)");
        System.out.println("      .samplingConfig(SamplingConfig.greedy())");
        System.out.println("      .maxNewTokens(200)");
        System.out.println("      .build();");
        System.out.println();
        System.out.println("  GenerationPipeline pipeline = GenerationPipeline.create(config);");
        System.out.println();
        System.out.println("  // Single-request usage (works identically to batchSize=1):");
        System.out.println("  GenerationResult result = pipeline.generate(\"Explain quantum computing\");");
        System.out.println();
        System.out.println("  // Multi-request batch usage (served concurrently via continuous batching):");
        System.out.println("  CompletableFuture<GenerationResult> f1 =");
        System.out.println("      pipeline.generateAsync(\"What is deep learning?\");");
        System.out.println("  CompletableFuture<GenerationResult> f2 =");
        System.out.println("      pipeline.generateAsync(\"Explain gradient descent.\");");
        System.out.println("  CompletableFuture.allOf(f1, f2).join();");
        System.out.println("  System.out.println(f1.get().getText());");
        System.out.println("  System.out.println(f2.get().getText());");
        System.out.println();
        System.out.println("  pipeline.close();");
        System.out.println();

        // ============================================================
        // 9. SERVING PATTERN SUMMARY
        // ============================================================
        System.out.println("=== 9. Production serving pattern summary ===");
        System.out.println();

        System.out.println("  The recommended production pattern:");
        System.out.println();
        System.out.println("  1. Create a ContinuousBatchScheduler (or GenerationPipeline with batchSize>1)");
        System.out.println("     once at server startup.  Keep it running for the lifetime of the service.");
        System.out.println();
        System.out.println("  2. For each incoming HTTP request (or gRPC call):");
        System.out.println("       int[] tokenIds = tokenizer.encode(userPrompt, true).getIds();");
        System.out.println("       CompletableFuture<GenerationResult> f =");
        System.out.println("           scheduler.submit(tokenIds, samplingConfig);");
        System.out.println("       GenerationResult r = f.get();  // wait for completion");
        System.out.println("       return r.getText();");
        System.out.println();
        System.out.println("  3. The scheduler's background thread batches requests automatically:");
        System.out.println("       - New requests fill free slots immediately (no wait if slot available)");
        System.out.println("       - Long prompts are chunked to bound memory");
        System.out.println("       - Decode steps run all active slots together (maximises GPU occupancy)");
        System.out.println("       - Finished slots are freed and reassigned to queued requests");
        System.out.println();
        System.out.println("  Key metrics to monitor in production:");
        System.out.println("    - Batch occupancy      — fraction of slots in use (target: > 80%)");
        System.out.println("    - Queue depth          — waiting requests (high = need more capacity)");
        System.out.println("    - P99 latency          — tail latency visible to clients");
        System.out.println("    - Token throughput     — total tokens/sec across all sequences");
        System.out.println();

        System.out.println("ContinuousBatchingExample completed.");
        System.out.println("Use GenerationPipeline with batchSize > 1 for the simplest path to");
        System.out.println("continuous batching. Use ContinuousBatchScheduler directly when you");
        System.out.println("need fine-grained slot control or custom serving loop logic.");
    }
}

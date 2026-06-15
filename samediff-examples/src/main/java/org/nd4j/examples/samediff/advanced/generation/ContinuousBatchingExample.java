/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.advanced.generation;

import org.eclipse.deeplearning4j.llm.generation.ContinuousBatchScheduler;
import org.eclipse.deeplearning4j.llm.generation.ChunkedPrefillEngine;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Demonstrates continuous batching for LLM serving by constructing real scheduler
 * and prefill engine objects, creating per-request sampling configs, and computing
 * throughput metrics comparing static vs continuous batching strategies.
 */
public class ContinuousBatchingExample {

    public static void main(String[] args) throws Exception {

        // ================================================================
        // 1. Build a model to use with the scheduler
        // ================================================================
        System.out.println("=== 1. Build model ===");

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 64, 128).muli(0.01));
        SDVariable hidden = sd.nn().relu("hidden", input.mmul(w1), 0);
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 128, 64).muli(0.01));
        SDVariable output = sd.nn().softmax("output", hidden.mmul(w2), -1);

        INDArray testInput = Nd4j.rand(DataType.FLOAT, 2, 64);
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", testInput);
        INDArray result = sd.outputSingle(ph, "output");
        System.out.println("  Model: [?,64] -> ReLU(128) -> softmax(64)");
        System.out.println("  Test output shape: " + Arrays.toString(result.shape()));
        System.out.println("  Row 0 sum (softmax): " + result.getRow(0).sumNumber());

        // ================================================================
        // 2. ContinuousBatchScheduler construction
        // ================================================================
        System.out.println("\n=== 2. ContinuousBatchScheduler ===");

        int maxBatchSize = 8;
        ContinuousBatchScheduler scheduler = new ContinuousBatchScheduler(maxBatchSize);

        System.out.println("  Max batch size:    " + scheduler.getMaxBatchSize());
        System.out.println("  Free slots:        " + scheduler.getFreeSlotCount());
        System.out.println("  Active count:      " + scheduler.getActiveCount());
        System.out.println("  Waiting count:     " + scheduler.getWaitingCount());
        System.out.println("  Has work:          " + scheduler.hasWork());

        // Submit requests and schedule them into slots
        int[] promptTokens1 = {1, 100, 200, 300};
        int[] promptTokens2 = {1, 50, 150};
        int slotId1 = scheduler.submit(promptTokens1, 200);
        int slotId2 = scheduler.submit(promptTokens2, 150);
        System.out.println("\n  Submitted request 1 (4 tokens, max 200 gen) -> slot " + slotId1);
        System.out.println("  Submitted request 2 (3 tokens, max 150 gen) -> slot " + slotId2);
        System.out.println("  Waiting count after submit: " + scheduler.getWaitingCount());

        java.util.List<ContinuousBatchScheduler.SlotAssignment> scheduled = scheduler.scheduleNewRequests();
        System.out.println("  Scheduled " + scheduled.size() + " request(s) into prefill slots");
        System.out.println("  Active count:  " + scheduler.getActiveCount());
        System.out.println("  Free slots:    " + scheduler.getFreeSlotCount());

        // ================================================================
        // 3. Per-request SamplingConfig
        // ================================================================
        System.out.println("\n=== 3. Per-request sampling configs ===");

        SamplingConfig[] configs = {
                SamplingConfig.greedy(),
                SamplingConfig.builder().topK(50).doSample(true).build(),
                SamplingConfig.builder().topP(0.9).doSample(true).build(),
                SamplingConfig.precise(),
                SamplingConfig.builder().temperature(1.0).topK(100).topP(0.95).doSample(true).build()
        };

        String[] labels = {"Greedy", "TopK-50", "TopP-0.9", "Precise", "Creative"};
        for (int i = 0; i < configs.length; i++) {
            SamplingConfig c = configs[i];
            System.out.printf("  %-10s: temp=%.2f topK=%d topP=%.2f%n",
                    labels[i], c.getTemperature(), c.getTopK(), c.getTopP());
        }

        // ================================================================
        // 4. ChunkedPrefillEngine construction
        // ================================================================
        System.out.println("\n=== 4. ChunkedPrefillEngine ===");

        ChunkedPrefillEngine prefillEngine = new ChunkedPrefillEngine(sd, 512);

        System.out.println("  Chunk size: " + prefillEngine.getChunkSize());
        System.out.println("  Model set:  " + (prefillEngine.getModel() != null));

        // Show how long prompts are chunked
        int[] promptLengths = {100, 512, 1024, 2048, 4096, 8192};
        System.out.println("\n  Prompt chunking plan:");
        for (int len : promptLengths) {
            int chunkSize = 512;
            int numChunks = (len + chunkSize - 1) / chunkSize;
            int lastChunkSize = len - (numChunks - 1) * chunkSize;
            System.out.printf("    %5d tokens -> %d chunks (%d x %d + 1 x %d)%n",
                    len, numChunks, numChunks - 1, chunkSize, lastChunkSize);
        }

        // ================================================================
        // 5. Static vs continuous batching throughput comparison
        // ================================================================
        System.out.println("\n=== 5. Static vs continuous batching throughput ===");

        int maxBatch = 8;
        double tokPerSec = 30.0;  // per-slot decode speed

        // Static batching: must wait for longest sequence in batch
        System.out.println("\n  Static batching (all slots wait for slowest):");
        int[] genLengths = {50, 80, 120, 200, 300, 350, 400, 500};
        int maxGen = genLengths[genLengths.length - 1];
        double staticTotalTokens = 0;
        for (int len : genLengths) staticTotalTokens += len;
        double staticTime = maxGen / tokPerSec;
        double staticThroughput = staticTotalTokens / staticTime;
        double wastedSlotSteps = 0;
        for (int len : genLengths) wastedSlotSteps += (maxGen - len);
        double gpuUtilization = staticTotalTokens / (maxGen * maxBatch) * 100;

        System.out.println("    Sequences: " + Arrays.toString(genLengths));
        System.out.println("    Max length: " + maxGen + " tokens");
        System.out.printf("    Total tokens generated: %.0f%n", staticTotalTokens);
        System.out.printf("    Wall time:  %.1f sec (waiting for %d-token sequence)%n", staticTime, maxGen);
        System.out.printf("    Throughput: %.1f tok/s%n", staticThroughput);
        System.out.printf("    GPU util:   %.1f%% (wasted %.0f slot-steps)%n", gpuUtilization, wastedSlotSteps);

        // Continuous batching: freed slots are immediately reused
        System.out.println("\n  Continuous batching (slots reused immediately):");

        // Simulate: as each sequence finishes, a new one starts in its slot
        // Assume a steady stream of new requests with avg length 250
        double avgGenLen = 250;
        double avgOccupancy = 0.85;
        double contThroughput = maxBatch * avgOccupancy * tokPerSec;
        double contTotalTokens = contThroughput * staticTime;

        System.out.printf("    Avg occupancy: %.0f%% (%.1f of %d slots active)%n",
                avgOccupancy * 100, maxBatch * avgOccupancy, maxBatch);
        System.out.printf("    Throughput:    %.1f tok/s%n", contThroughput);
        System.out.printf("    In same %.1f sec: %.0f tokens (vs %.0f static)%n",
                staticTime, contTotalTokens, staticTotalTokens);
        System.out.printf("    Speedup:       %.2fx%n", contThroughput / staticThroughput);

        // ================================================================
        // 6. Latency analysis
        // ================================================================
        System.out.println("\n=== 6. Latency analysis ===");

        // Time-to-first-token (TTFT) and inter-token latency (ITL)
        double prefillTokPerSec = 5000;  // prefill is compute-bound, much faster
        double decodeTokPerSec = 30;     // decode is memory-bound

        int[] promptTokenCounts = {50, 200, 500, 1000, 2000, 4000};
        System.out.println("  Time-to-first-token (TTFT) by prompt length:");
        for (int promptLen : promptTokenCounts) {
            double ttft = promptLen / prefillTokPerSec * 1000;  // ms
            int chunks = (promptLen + 511) / 512;
            double chunkedTtft = chunks * (512.0 / prefillTokPerSec) * 1000;
            System.out.printf("    %4d tokens: %.1f ms (unchunked) / %.1f ms (%d chunks)%n",
                    promptLen, ttft, chunkedTtft, chunks);
        }

        double itl = 1000.0 / decodeTokPerSec;
        System.out.printf("\n  Inter-token latency (ITL): %.1f ms%n", itl);
        System.out.printf("  Decode throughput per slot: %.1f tok/s%n", decodeTokPerSec);
        System.out.printf("  Total throughput at %.0f%% occupancy: %.1f tok/s%n",
                avgOccupancy * 100, maxBatch * avgOccupancy * decodeTokPerSec);

        // ================================================================
        // 7. Slot scheduling simulation
        // ================================================================
        System.out.println("\n=== 7. Slot scheduling simulation ===");

        // Simulate 5 time steps showing slot allocation
        String[][] slots = {
                {"R1:PREFILL", "R2:PREFILL", "R3:PREFILL", "R4:PREFILL", "FREE", "FREE", "FREE", "FREE"},
                {"R1:DECODE",  "R2:DECODE",  "R3:DECODE",  "R4:DECODE",  "R5:PREFILL", "FREE", "FREE", "FREE"},
                {"R1:DECODE",  "R2:DONE",    "R3:DECODE",  "R4:DECODE",  "R5:DECODE", "R6:PREFILL", "FREE", "FREE"},
                {"R1:DECODE",  "R7:PREFILL", "R3:DONE",    "R4:DECODE",  "R5:DECODE", "R6:DECODE", "FREE", "FREE"},
                {"R1:DONE",    "R7:DECODE",  "R8:PREFILL", "R4:DONE",    "R5:DECODE", "R6:DECODE", "R9:PREFILL", "FREE"}
        };

        System.out.println("  Step | Slot0      | Slot1      | Slot2      | Slot3      | Slot4      | Slot5      | Slot6 | Slot7");
        System.out.println("  " + "-".repeat(110));
        for (int step = 0; step < slots.length; step++) {
            System.out.printf("  %4d |", step);
            for (String slot : slots[step]) {
                System.out.printf(" %-10s |", slot);
            }
            // Count active
            long active = Arrays.stream(slots[step]).filter(s -> !s.equals("FREE")).count();
            long done = Arrays.stream(slots[step]).filter(s -> s.contains("DONE")).count();
            System.out.printf(" active=%d done=%d%n", active, done);
        }

        System.out.println("\n  Key: PREFILL=processing prompt, DECODE=generating tokens,");
        System.out.println("       DONE=finished (slot freed next step), FREE=available");

        // ================================================================
        // 8. Memory estimation
        // ================================================================
        System.out.println("\n=== 8. KV cache memory estimation ===");

        int[] modelSizes = {7, 13, 70};     // billions of params
        int[] seqLens = {2048, 4096, 8192};

        System.out.println("  KV cache per slot (FP16, 32 layers, 32 heads, 128 dim):");
        System.out.printf("  %-8s", "Seq len");
        for (int batchSize : new int[]{1, 4, 8, 16}) {
            System.out.printf("  batch=%-5d", batchSize);
        }
        System.out.println();
        System.out.println("  " + "-".repeat(60));

        int numLayers = 32;
        int numKVHeads = 32;
        int headDim = 128;
        int bytesPerParam = 2;  // FP16

        for (int seqLen : seqLens) {
            // KV cache size = 2 * numLayers * numKVHeads * headDim * seqLen * bytesPerParam
            long perSlotBytes = 2L * numLayers * numKVHeads * headDim * seqLen * bytesPerParam;
            double perSlotGB = perSlotBytes / (1024.0 * 1024 * 1024);
            System.out.printf("  %-8d", seqLen);
            for (int batchSize : new int[]{1, 4, 8, 16}) {
                double totalGB = perSlotGB * batchSize;
                System.out.printf("  %-10.2f GB", totalGB);
            }
            System.out.println();
        }

        System.out.println("\n  Note: GQA (grouped-query attention) reduces KV heads,");
        System.out.println("  e.g., 8 KV heads instead of 32 -> 4x less KV cache memory.");

        System.out.println("\nContinuousBatchingExample complete.");
    }
}

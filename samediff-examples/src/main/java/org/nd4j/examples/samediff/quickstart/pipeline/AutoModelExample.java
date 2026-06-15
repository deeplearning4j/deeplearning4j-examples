/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.quickstart.pipeline;

import org.eclipse.deeplearning4j.pipeline.AutoModel;
import org.eclipse.deeplearning4j.pipeline.ModelFormat;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader.LoadConfig;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Demonstrates AutoModel format-agnostic model loading by building a SameDiff
 * model, saving it to disk, then loading it back via AutoModel.fromPretrained().
 * Shows LoadConfig options, ModelFormat detection, and round-trip verification.
 */
public class AutoModelExample {

    public static void main(String[] args) throws Exception {

        // ================================================================
        // 1. Build a small SameDiff model and run inference
        // ================================================================
        System.out.println("=== 1. Build a SameDiff model ===");

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 3).muli(0.1));
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 3));
        SDVariable logits = input.mmul(weights).add("logits", bias);
        SDVariable output = sd.nn().softmax("output", logits, -1);

        INDArray testInput = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", testInput);
        INDArray originalOutput = sd.outputSingle(placeholders, "output");

        System.out.println("  Input shape:  " + Arrays.toString(testInput.shape()));
        System.out.println("  Output shape: " + Arrays.toString(originalOutput.shape()));
        System.out.println("  Output row 0: " + originalOutput.getRow(0));
        System.out.println("  Output row 1: " + originalOutput.getRow(1));
        System.out.println("  Variables:    " + sd.variableNames());

        // ================================================================
        // 2. Save model to .sdz file
        // ================================================================
        System.out.println("\n=== 2. Save model as .sdz ===");

        File tmpFile = File.createTempFile("automodel-demo", ".sdz");
        tmpFile.deleteOnExit();
        sd.save(tmpFile, false);

        System.out.println("  Saved to: " + tmpFile.getAbsolutePath());
        System.out.println("  File size: " + tmpFile.length() + " bytes");

        // ================================================================
        // 3. Load via AutoModel.fromPretrained() — format auto-detected
        // ================================================================
        System.out.println("\n=== 3. AutoModel.fromPretrained() round-trip ===");

        SameDiff loaded = AutoModel.fromPretrained(tmpFile.getAbsolutePath());

        INDArray reloadedOutput = loaded.outputSingle(placeholders, "output");
        double maxDelta = originalOutput.sub(reloadedOutput).amaxNumber().doubleValue();

        System.out.println("  Loaded variables: " + loaded.variableNames());
        System.out.println("  Reloaded output:  " + reloadedOutput.getRow(0));
        System.out.println("  Max delta from original: " + maxDelta);
        System.out.println("  Round-trip match: " + (maxDelta < 1e-6 ? "YES" : "NO (delta=" + maxDelta + ")"));

        // ================================================================
        // 4. LoadConfig builder — caching, device, mmap options
        // ================================================================
        System.out.println("\n=== 4. LoadConfig options ===");

        LoadConfig config = LoadConfig.builder()
                .cacheConvertedModel(true)
                .device("0")
                .useMmap(true)
                .convertToFloat32(false)
                .dequantize(false)
                .build();

        System.out.println("  cacheConvertedModel: " + config.cacheConvertedModel());
        System.out.println("  device:              " + config.getDevice());
        System.out.println("  useMmap:             " + config.useMmap());
        System.out.println("  convertToFloat32:    " + config.convertToFloat32());
        System.out.println("  dequantize:          " + config.dequantize());

        SameDiff loadedWithConfig = AutoModel.fromPretrained(tmpFile.getAbsolutePath(), config);
        INDArray configOutput = loadedWithConfig.outputSingle(placeholders, "output");
        System.out.println("  Loaded with config, output matches: " +
                (originalOutput.sub(configOutput).amaxNumber().doubleValue() < 1e-6));

        // ================================================================
        // 5. ModelFormat enum — all supported formats
        // ================================================================
        System.out.println("\n=== 5. ModelFormat values ===");

        for (ModelFormat fmt : ModelFormat.values()) {
            System.out.println("  " + fmt.name() + " (ext=" + fmt.getExtension()
                    + ", " + fmt.getDescription() + ")");
        }

        ModelFormat detected = ModelFormat.fromFilename(tmpFile.getName());
        System.out.println("  Detected format for saved .sdz: " + detected);

        // ================================================================
        // 6. Inspect loaded model variables
        // ================================================================
        System.out.println("\n=== 6. Variable inspection ===");

        for (String varName : loaded.variableNames()) {
            SDVariable var = loaded.getVariable(varName);
            INDArray arr = var.getArr();
            if (arr != null) {
                System.out.println("  " + varName + ": shape=" +
                        Arrays.toString(arr.shape()) + " dtype=" + arr.dataType());
            } else {
                System.out.println("  " + varName + ": placeholder (no array)");
            }
        }

        // ================================================================
        // 7. Build a larger model and verify round-trip
        // ================================================================
        System.out.println("\n=== 7. Larger model round-trip ===");

        SameDiff sd2 = SameDiff.create();
        SDVariable in2 = sd2.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd2.var("w1", Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1));
        SDVariable b1 = sd2.var("b1", Nd4j.zeros(DataType.FLOAT, 32));
        SDVariable hidden = sd2.nn().relu("hidden", in2.mmul(w1).add(b1), 0);
        SDVariable w2 = sd2.var("w2", Nd4j.randn(DataType.FLOAT, 32, 5).muli(0.1));
        SDVariable b2 = sd2.var("b2", Nd4j.zeros(DataType.FLOAT, 5));
        SDVariable out2 = sd2.nn().softmax("output", hidden.mmul(w2).add(b2), -1);

        File tmpFile2 = File.createTempFile("automodel-larger", ".sdz");
        tmpFile2.deleteOnExit();
        sd2.save(tmpFile2, false);

        INDArray testIn2 = Nd4j.rand(DataType.FLOAT, 3, 16);
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("input", testIn2);
        INDArray origOut2 = sd2.outputSingle(ph2, "output");

        SameDiff loaded2 = AutoModel.fromPretrained(tmpFile2.getAbsolutePath());
        INDArray reloadOut2 = loaded2.outputSingle(ph2, "output");

        System.out.println("  Model: 16->32->5 with ReLU hidden layer");
        System.out.println("  Input:  [3, 16]");
        System.out.println("  Output: " + Arrays.toString(reloadOut2.shape()));
        System.out.println("  Row sums (should be ~1.0 for softmax):");
        for (int i = 0; i < 3; i++) {
            System.out.println("    Row " + i + " sum = " + reloadOut2.getRow(i).sumNumber());
        }
        double delta2 = origOut2.sub(reloadOut2).amaxNumber().doubleValue();
        System.out.println("  Round-trip match: " + (delta2 < 1e-6));
        System.out.println("  File size: " + tmpFile2.length() + " bytes");

        System.out.println("\nAutoModelExample complete.");
    }
}

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

package org.nd4j.examples.samediff.quickstart.operations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.enums.DataFormat;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * SameDiff CNN Operations (sd.cnn namespace) - Complete API Reference
 *
 * This example covers all convolution and pooling operations:
 *
 *   1. Conv1D - 1D convolution for sequence/temporal data
 *   2. Conv2D - 2D convolution for images (standard CNN)
 *   3. Conv3D - 3D convolution for video/volumetric data
 *   4. Deconv2D / Deconv3D - Transposed (fractionally-strided) convolutions
 *   5. DepthwiseConv2D - Depthwise separable convolution (MobileNet)
 *   6. SeparableConv2D - Depth + pointwise convolution
 *   7. Pooling - Max/Avg 2D/3D, adaptive pooling, maxPoolWithArgmax
 *   8. Space/Depth/Batch transforms - spaceToDepth, depthToSpace, spaceToBatch, batchToSpace
 *   9. Upsampling - 2D and 3D nearest-neighbor upsampling
 *  10. Im2Col / Col2Im - Low-level patch extraction
 *  11. Local Response Normalization
 *  12. Dilation2D - Morphological dilation
 *
 * Key config classes:
 *   - Conv1DConfig, Conv2DConfig, Conv3DConfig
 *   - Pooling2DConfig, Pooling3DConfig
 *   - PaddingMode (VALID, SAME, CAUSAL)
 *   - DataFormat (NCHW, NHWC)
 */
public class CNNOpsExample {

    public static void main(String[] args) {

        int batch = 2;

        // ============================================================
        // 1. CONV1D - 1D Convolution
        // ============================================================
        System.out.println("=== Conv1D ===");
        {
            SameDiff sd = SameDiff.create();
            int inChannels = 3, outChannels = 8, seqLen = 20, kernelSize = 5;

            // Input: [batch, channels, length] (NCW format)
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inChannels, seqLen);
            // Weights: [kernelSize, inChannels, outChannels]
            SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, kernelSize, inChannels, outChannels).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outChannels));

            Conv1DConfig config = Conv1DConfig.builder()
                    .k(kernelSize)  // kernel size
                    .s(1)           // stride
                    .p(0)           // padding
                    .d(1)           // dilation
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCW")
                    .build();

            SDVariable conv1d = sd.cnn().conv1d("conv1d", input, weights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inChannels, seqLen)),
                    "conv1d");
            System.out.println("  Input shape:  [" + batch + ", " + inChannels + ", " + seqLen + "] (NCW)");
            System.out.println("  Output shape: " + java.util.Arrays.toString(result.get("conv1d").shape()));
            System.out.println("  Kernel=" + kernelSize + ", stride=1, SAME padding");
        }

        // ============================================================
        // 2. CONV2D - 2D Convolution
        // ============================================================
        System.out.println("\n=== Conv2D ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 3, outCh = 16, h = 28, w = 28;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);
            // Weights: [kH, kW, inCh, outCh] (YXIO format by default)
            SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 3, 3, inCh, outCh).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outCh));

            // Standard 3x3 convolution with SAME padding
            Conv2DConfig config = Conv2DConfig.builder()
                    .kH(3).kW(3)        // kernel size
                    .sH(1).sW(1)        // stride
                    .pH(0).pW(0)        // padding (ignored when SAME)
                    .dH(1).dW(1)        // dilation
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();

            SDVariable conv2d = sd.cnn().conv2d("conv2d", input, weights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "conv2d");
            System.out.println("  Input:  [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("conv2d").shape()));

            // Dilated convolution (atrous convolution)
            Conv2DConfig dilatedConfig = Conv2DConfig.builder()
                    .kH(3).kW(3)
                    .sH(1).sW(1)
                    .dH(2).dW(2)  // dilation rate 2 => effective kernel 5x5
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();
            SDVariable dilatedConv = sd.cnn().conv2d("dilated_conv2d", input, weights, bias, dilatedConfig);
            System.out.println("  Dilated conv (rate=2): effective 5x5 receptive field");

            // Strided convolution (downsampling)
            Conv2DConfig stridedConfig = Conv2DConfig.builder()
                    .kH(3).kW(3)
                    .sH(2).sW(2)  // stride 2 => output is half size
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();
            SDVariable stridedConv = sd.cnn().conv2d("strided_conv2d", input, weights, bias, stridedConfig);

            Map<String, INDArray> result2 = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "strided_conv2d");
            System.out.println("  Strided conv (stride=2): " + java.util.Arrays.toString(result2.get("strided_conv2d").shape()));
        }

        // ============================================================
        // 3. CONV3D - 3D Convolution
        // ============================================================
        System.out.println("\n=== Conv3D ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 1, outCh = 8, d = 10, h = 16, w = 16;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, d, h, w);
            SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 3, 3, 3, inCh, outCh).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outCh));

            Conv3DConfig config = Conv3DConfig.builder()
                    .kD(3).kH(3).kW(3)
                    .sD(1).sH(1).sW(1)
                    .pD(0).pH(0).pW(0)
                    .dD(1).dH(1).dW(1)
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCDHW")
                    .build();

            SDVariable conv3d = sd.cnn().conv3d("conv3d", input, weights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, d, h, w)),
                    "conv3d");
            System.out.println("  Input:  [" + batch + ", " + inCh + ", " + d + ", " + h + ", " + w + "]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("conv3d").shape()));
        }

        // ============================================================
        // 4. DECONV2D (Transposed Convolution)
        // ============================================================
        System.out.println("\n=== Deconv2D (Transposed Convolution) ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 16, outCh = 8, h = 14, w = 14;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);
            SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 4, inCh, outCh).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outCh));

            // Transposed conv with stride 2 => doubles spatial dims
            DeConv2DConfig config = DeConv2DConfig.builder()
                    .kH(4).kW(4)
                    .sH(2).sW(2)
                    .isSameMode(true)
                    .dataFormat("NCHW")
                    .build();

            SDVariable deconv = sd.cnn().deconv2d("deconv2d", input, weights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "deconv2d");
            System.out.println("  Input:  [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("deconv2d").shape()));
            System.out.println("  Stride=2 transposed conv doubles spatial dimensions (upsampling)");
        }

        // ============================================================
        // 5. DEPTHWISE CONV2D (MobileNet-style)
        // ============================================================
        System.out.println("\n=== Depthwise Conv2D ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 32, h = 28, w = 28;
            int depthMultiplier = 1;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);
            // Depthwise weights: [kH, kW, inCh, depthMultiplier]
            SDVariable depthWeights = sd.var("depthWeights",
                    Nd4j.randn(DataType.FLOAT, 3, 3, inCh, depthMultiplier).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, inCh * depthMultiplier));

            Conv2DConfig config = Conv2DConfig.builder()
                    .kH(3).kW(3)
                    .sH(1).sW(1)
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();

            SDVariable dwConv = sd.cnn().depthWiseConv2d("dw_conv2d", input, depthWeights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "dw_conv2d");
            System.out.println("  Input:  [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("dw_conv2d").shape()));
            System.out.println("  Each channel convolved independently (MobileNet pattern)");
        }

        // ============================================================
        // 6. SEPARABLE CONV2D (Depthwise + Pointwise)
        // ============================================================
        System.out.println("\n=== Separable Conv2D ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 16, outCh = 32, h = 28, w = 28;
            int depthMultiplier = 1;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);
            SDVariable depthWeights = sd.var("depthWeights",
                    Nd4j.randn(DataType.FLOAT, 3, 3, inCh, depthMultiplier).muli(0.1));
            // Pointwise weights: [1, 1, inCh*depthMultiplier, outCh]
            SDVariable pointWeights = sd.var("pointWeights",
                    Nd4j.randn(DataType.FLOAT, 1, 1, inCh * depthMultiplier, outCh).muli(0.1));
            SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outCh));

            Conv2DConfig config = Conv2DConfig.builder()
                    .kH(3).kW(3)
                    .sH(1).sW(1)
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();

            SDVariable sepConv = sd.cnn().separableConv2d("sep_conv2d", input, depthWeights,
                    pointWeights, bias, config);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "sep_conv2d");
            System.out.println("  Input:  [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("sep_conv2d").shape()));
            System.out.println("  Depthwise 3x3 + Pointwise 1x1 (Xception/MobileNet pattern)");
        }

        // ============================================================
        // 7. POOLING OPERATIONS
        // ============================================================
        System.out.println("\n=== Pooling Operations ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 16, h = 28, w = 28;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);

            // --- Max Pooling 2D ---
            Pooling2DConfig maxPoolConfig = Pooling2DConfig.builder()
                    .kH(2).kW(2)
                    .sH(2).sW(2)
                    .paddingMode(PaddingMode.VALID)
                    .type(Pooling2D.Pooling2DType.MAX)
                    .isNHWC(false)
                    .build();
            SDVariable maxPool = sd.cnn().maxPooling2d("max_pool", input, maxPoolConfig);

            // --- Average Pooling 2D ---
            Pooling2DConfig avgPoolConfig = Pooling2DConfig.builder()
                    .kH(2).kW(2)
                    .sH(2).sW(2)
                    .paddingMode(PaddingMode.VALID)
                    .type(Pooling2D.Pooling2DType.AVG)
                    .isNHWC(false)
                    .build();
            SDVariable avgPool = sd.cnn().avgPooling2d("avg_pool", input, avgPoolConfig);

            // --- Max Pool with Argmax (returns indices of max values) ---
            SDVariable[] maxPoolArgmax = sd.cnn().maxPoolWithArgmax(
                    new String[]{"maxpool_values", "maxpool_indices"}, input, maxPoolConfig);

            // --- Adaptive Average Pooling (output fixed size regardless of input) ---
            SDVariable adaptiveAvg = sd.cnn().adaptiveAvgPooling2d("adaptive_avg", input, 7, 7);

            // --- Adaptive Max Pooling ---
            SDVariable adaptiveMax = sd.cnn().adaptiveMaxPooling2d("adaptive_max", input, 1, 1);

            Map<String, INDArray> placeholders = java.util.Collections.singletonMap(
                    "input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w));
            Map<String, INDArray> result = sd.output(placeholders,
                    "max_pool", "avg_pool", "adaptive_avg", "adaptive_max");

            System.out.println("  Max Pool 2x2:     " + java.util.Arrays.toString(result.get("max_pool").shape()));
            System.out.println("  Avg Pool 2x2:     " + java.util.Arrays.toString(result.get("avg_pool").shape()));
            System.out.println("  Adaptive Avg 7x7: " + java.util.Arrays.toString(result.get("adaptive_avg").shape()));
            System.out.println("  Adaptive Max 1x1: " + java.util.Arrays.toString(result.get("adaptive_max").shape()));
            System.out.println("  (Global avg pool = adaptiveAvgPooling2d with output 1x1)");
        }

        // ============================================================
        // 8. SPACE/DEPTH/BATCH TRANSFORMS
        // ============================================================
        System.out.println("\n=== Space/Depth Transforms ===");
        {
            SameDiff sd = SameDiff.create();

            // --- Space to Depth (PixelUnshuffle) ---
            // Rearranges spatial blocks into depth: [N, C, H, W] -> [N, C*block^2, H/block, W/block]
            SDVariable input1 = sd.placeHolder("input_s2d", DataType.FLOAT, batch, 3, 8, 8);
            SDVariable s2d = sd.cnn().spaceToDepth("s2d", input1, 2, DataFormat.NCHW);

            // --- Depth to Space (PixelShuffle / sub-pixel convolution) ---
            // Inverse of spaceToDepth: [N, C*block^2, H, W] -> [N, C, H*block, W*block]
            SDVariable input2 = sd.placeHolder("input_d2s", DataType.FLOAT, batch, 12, 4, 4);
            SDVariable d2s = sd.cnn().depthToSpace("d2s", input2, 2, DataFormat.NCHW);

            // --- Space to Batch ---
            SDVariable input3 = sd.placeHolder("input_s2b", DataType.FLOAT, batch, 1, 4, 4);
            SDVariable s2b = sd.cnn().spaceToBatch("s2b", input3,
                    new int[]{2, 2},   // block sizes
                    new int[]{0, 0},   // padding top
                    new int[]{0, 0});  // padding bottom

            java.util.Map<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("input_s2d", Nd4j.randn(DataType.FLOAT, batch, 3, 8, 8));
            ph.put("input_d2s", Nd4j.randn(DataType.FLOAT, batch, 12, 4, 4));
            ph.put("input_s2b", Nd4j.randn(DataType.FLOAT, batch, 1, 4, 4));

            Map<String, INDArray> result = sd.output(ph, "s2d", "d2s", "s2b");
            System.out.println("  SpaceToDepth [2,3,8,8] block=2 -> " + java.util.Arrays.toString(result.get("s2d").shape()));
            System.out.println("  DepthToSpace [2,12,4,4] block=2 -> " + java.util.Arrays.toString(result.get("d2s").shape()));
            System.out.println("  SpaceToBatch [2,1,4,4] block=[2,2] -> " + java.util.Arrays.toString(result.get("s2b").shape()));
        }

        // ============================================================
        // 9. UPSAMPLING (nearest-neighbor)
        // ============================================================
        System.out.println("\n=== Upsampling ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 8, h = 7, w = 7;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);

            // Uniform 2x upsampling
            SDVariable up2x = sd.cnn().upsampling2d("up2x", input, 2);

            // Non-uniform upsampling (different H/W scale)
            SDVariable upHW = sd.cnn().upsampling2d("upHW", input, 3, 4, true); // scaleH=3, scaleW=4, nchw=true

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "up2x", "upHW");
            System.out.println("  Input:         [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Upsample 2x:   " + java.util.Arrays.toString(result.get("up2x").shape()));
            System.out.println("  Upsample 3x4:  " + java.util.Arrays.toString(result.get("upHW").shape()));
        }

        // ============================================================
        // 10. IM2COL / COL2IM
        // ============================================================
        System.out.println("\n=== Im2Col / Col2Im ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 3, h = 8, w = 8;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);

            Conv2DConfig im2colConfig = Conv2DConfig.builder()
                    .kH(3).kW(3)
                    .sH(1).sW(1)
                    .paddingMode(PaddingMode.SAME)
                    .dataFormat("NCHW")
                    .build();

            // Im2Col: extracts all patches as columns
            // Output: [batch, inCh, kH, kW, outH, outW]
            SDVariable cols = sd.cnn().im2Col("im2col", input, im2colConfig);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "im2col");
            System.out.println("  Input shape:  [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Im2Col shape: " + java.util.Arrays.toString(result.get("im2col").shape()));
            System.out.println("  (Extracts sliding window patches as columns for manual convolution)");
        }

        // ============================================================
        // 11. LOCAL RESPONSE NORMALIZATION
        // ============================================================
        System.out.println("\n=== Local Response Normalization ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 16, h = 14, w = 14;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);

            LocalResponseNormalizationConfig lrnConfig = LocalResponseNormalizationConfig.builder()
                    .alpha(1e-4)   // scaling parameter
                    .beta(0.75)    // exponent
                    .bias(1.0)     // bias (k)
                    .depth(5)      // neighborhood size
                    .build();

            SDVariable lrn = sd.cnn().localResponseNormalization("lrn", input, lrnConfig);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "lrn");
            System.out.println("  LRN output: " + java.util.Arrays.toString(result.get("lrn").shape()));
            System.out.println("  Formula: x / (bias + alpha * sum(x_neighbors^2))^beta");
        }

        // ============================================================
        // 12. EXTRACT IMAGE PATCHES (ViT-style)
        // ============================================================
        System.out.println("\n=== Extract Image Patches ===");
        {
            SameDiff sd = SameDiff.create();
            int inCh = 3, h = 32, w = 32;
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inCh, h, w);

            // Extract non-overlapping 16x16 patches (like Vision Transformer)
            SDVariable patches = sd.cnn().extractImagePatches("patches", input,
                    16, 16,  // kH, kW (patch size)
                    16, 16,  // sH, sW (stride = patch size for non-overlapping)
                    1, 1,    // rH, rW (dilation rates)
                    true);   // sameMode

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, batch, inCh, h, w)),
                    "patches");
            System.out.println("  Input: [" + batch + ", " + inCh + ", " + h + ", " + w + "]");
            System.out.println("  Patches: " + java.util.Arrays.toString(result.get("patches").shape()));
            System.out.println("  (Non-overlapping 16x16 patches, ViT patchification)");
        }

        // ============================================================
        // 13. BUILDING A SMALL CNN (putting it together)
        // ============================================================
        System.out.println("\n=== Mini CNN Architecture ===");
        {
            SameDiff sd = SameDiff.create();
            int h = 28, w = 28;

            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1, h, w);

            // Conv block 1: Conv2D -> ReLU -> MaxPool
            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 3, 3, 1, 16).muli(0.1));
            SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 16));
            SDVariable conv1 = sd.cnn().conv2d(input, w1, b1,
                    Conv2DConfig.builder().kH(3).kW(3).paddingMode(PaddingMode.SAME).dataFormat("NCHW").build());
            SDVariable relu1 = sd.nn().relu(conv1, 0);
            SDVariable pool1 = sd.cnn().maxPooling2d(relu1,
                    Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build());

            // Conv block 2: Conv2D -> ReLU -> MaxPool
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 3, 3, 16, 32).muli(0.1));
            SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 32));
            SDVariable conv2 = sd.cnn().conv2d(pool1, w2, b2,
                    Conv2DConfig.builder().kH(3).kW(3).paddingMode(PaddingMode.SAME).dataFormat("NCHW").build());
            SDVariable relu2 = sd.nn().relu(conv2, 0);
            SDVariable pool2 = sd.cnn().maxPooling2d(relu2,
                    Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build());

            // Global average pooling (adaptive to 1x1)
            SDVariable gap = sd.cnn().adaptiveAvgPooling2d("gap", pool2, 1, 1);

            Map<String, INDArray> result = sd.output(
                    java.util.Collections.singletonMap("input", Nd4j.randn(DataType.FLOAT, 4, 1, h, w)),
                    "gap");
            System.out.println("  Architecture: Conv3x3(1->16) -> ReLU -> MaxPool2x2");
            System.out.println("                Conv3x3(16->32) -> ReLU -> MaxPool2x2");
            System.out.println("                Global Avg Pool");
            System.out.println("  Input:  [4, 1, 28, 28]");
            System.out.println("  Output: " + java.util.Arrays.toString(result.get("gap").shape()));
        }

        System.out.println("\nAll CNN operations demonstrated successfully.");
    }
}

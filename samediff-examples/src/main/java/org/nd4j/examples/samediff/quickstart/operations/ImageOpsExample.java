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
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

/**
 * SameDiff Image Operations (sd.image()) - Complete API Example
 *
 * The SDImage namespace provides differentiable image processing operations
 * for use in computer vision pipelines.
 *
 * Operations covered:
 *   Color Space Conversions:
 *     - rgbToHsv / hsvToRgb
 *     - rgbToYiq / yiqToRgb
 *     - rgbToYuv / yuvToRgb
 *
 *   Color Adjustments:
 *     - adjustContrast
 *     - adjustHue
 *     - adjustSaturation
 *
 *   Spatial Transforms:
 *     - imageResize (with multiple interpolation methods)
 *     - resizeBiLinear / resizeBiCubic
 *     - cropAndResize
 *     - randomCrop
 *     - affineGrid
 *     - extractImagePatches
 *
 *   Other:
 *     - nonMaxSuppression (object detection NMS)
 *     - pad
 */
public class ImageOpsExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. COLOR SPACE CONVERSIONS
        // ============================================================
        System.out.println("=== Color Space Conversions ===");
        {
            SameDiff sd = SameDiff.create();

            // Image in NHWC format: [batch, height, width, channels]
            SDVariable rgbImage = sd.placeHolder("rgb", DataType.FLOAT, -1, 64, 64, 3);

            // RGB <-> HSV
            SDVariable hsv = sd.image().rgbToHsv("toHsv", rgbImage);
            SDVariable backToRgb = sd.image().hsvToRgb("backToRgb", hsv);

            // RGB <-> YIQ (luminance, in-phase, quadrature)
            SDVariable yiq = sd.image().rgbToYiq("toYiq", rgbImage);
            SDVariable backFromYiq = sd.image().yiqToRgb("fromYiq", yiq);

            // RGB <-> YUV (luminance, blue-diff chrominance, red-diff chrominance)
            SDVariable yuv = sd.image().rgbToYuv("toYuv", rgbImage);
            SDVariable backFromYuv = sd.image().yuvToRgb("fromYuv", yuv);

            // Pixel values should be in [0,1] range for color conversions
            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 64, 64, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("rgb", imageData),
                    "toHsv", "backToRgb", "toYiq", "toYuv");

            System.out.println("  RGB input shape:    " + imageData.shapeInfoToString());
            System.out.println("  HSV output shape:   " + result.get("toHsv").shapeInfoToString());
            System.out.println("  YIQ output shape:   " + result.get("toYiq").shapeInfoToString());
            System.out.println("  YUV output shape:   " + result.get("toYuv").shapeInfoToString());
            System.out.println("  Round-trip RGB shape: " + result.get("backToRgb").shapeInfoToString());
        }

        // ============================================================
        // 2. COLOR ADJUSTMENTS
        // ============================================================
        System.out.println("\n=== Color Adjustments ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 64, 64, 3);

            // Adjust contrast: factor > 1 increases contrast, < 1 decreases
            SDVariable highContrast = sd.image().adjustContrast("highContrast", image, 2.0);
            SDVariable lowContrast = sd.image().adjustContrast("lowContrast", image, 0.5);

            // Adjust hue: delta in [-1, 1] rotates the hue channel
            SDVariable hueShifted = sd.image().adjustHue("hueShifted", image, 0.2);

            // Adjust saturation: factor > 1 increases saturation, < 1 decreases
            SDVariable saturated = sd.image().adjustSaturation("saturated", image, 1.5);
            SDVariable desaturated = sd.image().adjustSaturation("desaturated", image, 0.3);

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 2, 64, 64, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData),
                    "highContrast", "lowContrast", "hueShifted", "saturated", "desaturated");

            System.out.println("  High contrast shape: " + result.get("highContrast").shapeInfoToString());
            System.out.println("  Hue shifted shape:   " + result.get("hueShifted").shapeInfoToString());
            System.out.println("  Saturated shape:     " + result.get("saturated").shapeInfoToString());
        }

        // ============================================================
        // 3. IMAGE RESIZE - Multiple interpolation methods
        // ============================================================
        System.out.println("\n=== Image Resize ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 64, 64, 3);
            SDVariable targetSize = sd.constant("targetSize", Nd4j.createFromArray(128, 128));

            // imageResize with different interpolation methods
            SDVariable bilinear = sd.image().imageResize("bilinear", image, targetSize,
                    ImageResizeMethod.ResizeBilinear);
            SDVariable bicubic = sd.image().imageResize("bicubic", image, targetSize,
                    ImageResizeMethod.ResizeBicubic);
            SDVariable nearest = sd.image().imageResize("nearest", image, targetSize,
                    ImageResizeMethod.ResizeNearest);

            // With preserveAspectRatio and antialias options
            SDVariable resizedFull = sd.image().imageResize("resizedFull", image, targetSize,
                    false, true, ImageResizeMethod.ResizeBilinear);

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 64, 64, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData),
                    "bilinear", "bicubic", "nearest");

            System.out.println("  Input:     64x64 -> Target: 128x128");
            System.out.println("  Bilinear:  " + result.get("bilinear").shapeInfoToString());
            System.out.println("  Bicubic:   " + result.get("bicubic").shapeInfoToString());
            System.out.println("  Nearest:   " + result.get("nearest").shapeInfoToString());
        }

        // ============================================================
        // 4. RESIZE BILINEAR / BICUBIC - Direct methods
        // ============================================================
        System.out.println("\n=== ResizeBiLinear / ResizeBiCubic ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 32, 32, 3);

            // resizeBiLinear: takes height, width as ints
            SDVariable upscaled = sd.image().resizeBiLinear("upscaled", image,
                    128, 128, false, false);

            // resizeBiCubic: takes SDVariable size, alignCorners, alignPixelCenters
            SDVariable size = sd.constant("size", Nd4j.createFromArray(96, 96));
            SDVariable cubicUp = sd.image().resizeBiCubic("cubicUp", image, size, false, false);

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 32, 32, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData), "upscaled", "cubicUp");
            System.out.println("  BiLinear 32->128: " + result.get("upscaled").shapeInfoToString());
            System.out.println("  BiCubic  32->96:  " + result.get("cubicUp").shapeInfoToString());
        }

        // ============================================================
        // 5. CROP AND RESIZE - Region-of-interest extraction
        // ============================================================
        System.out.println("\n=== Crop and Resize ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 100, 100, 3);

            // Crop boxes: [numBoxes, 4] with normalized coords [y1, x1, y2, x2]
            SDVariable cropBoxes = sd.constant("cropBoxes", Nd4j.createFromArray(new float[][]{
                    {0.1f, 0.1f, 0.5f, 0.5f},   // Top-left crop
                    {0.5f, 0.5f, 0.9f, 0.9f}     // Bottom-right crop
            }));

            // Box indices: maps each box to a batch image
            SDVariable boxIndices = sd.constant("boxIndices", Nd4j.createFromArray(0, 0));

            // Output size for each crop
            SDVariable cropSize = sd.constant("cropSize", Nd4j.createFromArray(32, 32));

            // With extrapolation value for out-of-bounds
            SDVariable crops = sd.image().cropAndResize("crops", image, cropBoxes, boxIndices,
                    cropSize, 0.0);

            // Without extrapolation value
            SDVariable cropsDefault = sd.image().cropAndResize("cropsDefault", image, cropBoxes,
                    boxIndices, cropSize);

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 100, 100, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData), "crops");
            System.out.println("  2 crop boxes from 100x100 -> 32x32 each");
            System.out.println("  Crops output shape: " + result.get("crops").shapeInfoToString());
        }

        // ============================================================
        // 6. RANDOM CROP
        // ============================================================
        System.out.println("\n=== Random Crop ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 256, 256, 3);

            // Random crop to specified shape
            SDVariable cropShape = sd.constant("cropShape",
                    Nd4j.createFromArray(1, 224, 224, 3).castTo(DataType.INT));
            SDVariable cropped = sd.image().randomCrop("cropped", image, cropShape);

            System.out.println("  Random crop: 256x256 -> random 224x224 subregion");
            System.out.println("  Use for data augmentation during training");
        }

        // ============================================================
        // 7. EXTRACT IMAGE PATCHES
        // ============================================================
        System.out.println("\n=== Extract Image Patches ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 64, 64, 3);

            // Extract 8x8 patches with stride 8 (non-overlapping)
            SDVariable patches = sd.image().extractImagePatches("patches", image,
                    new int[]{1, 8, 8, 1},   // kSizes: [1, kH, kW, 1]
                    new int[]{1, 8, 8, 1},   // strides: [1, sH, sW, 1]
                    new int[]{1, 1, 1, 1},   // rates (dilation): [1, rH, rW, 1]
                    true);                    // sameMode

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 64, 64, 3);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData), "patches");
            System.out.println("  64x64 image -> 8x8 patches:");
            System.out.println("  Patches shape: " + result.get("patches").shapeInfoToString());
            System.out.println("  Use for Vision Transformer (ViT) patch embedding");
        }

        // ============================================================
        // 8. AFFINE GRID - Spatial transformer network support
        // ============================================================
        System.out.println("\n=== Affine Grid ===");
        {
            SameDiff sd = SameDiff.create();

            // Theta: [batch, 2, 3] affine transformation matrix
            // Identity: [[1,0,0],[0,1,0]]
            SDVariable theta = sd.placeHolder("theta", DataType.FLOAT, -1, 2, 3);
            SDVariable size = sd.constant("size", Nd4j.createFromArray(1, 3, 64, 64));

            // Generate sampling grid for spatial transformer
            SDVariable grid = sd.image().affineGrid("grid", theta, size, true);

            // Without alignCorners
            SDVariable gridNoAlign = sd.image().affineGrid("gridNoAlign", theta, size);

            // Identity transform
            INDArray thetaData = Nd4j.zeros(DataType.FLOAT, 1, 2, 3);
            thetaData.putScalar(0, 0, 0, 1.0f); // scale x
            thetaData.putScalar(0, 1, 1, 1.0f); // scale y
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("theta", thetaData), "grid");
            System.out.println("  Affine grid shape: " + result.get("grid").shapeInfoToString());
            System.out.println("  Use with grid_sample for Spatial Transformer Networks");
        }

        // ============================================================
        // 9. NON-MAX SUPPRESSION - Object detection post-processing
        // ============================================================
        System.out.println("\n=== Non-Max Suppression ===");
        {
            SameDiff sd = SameDiff.create();

            // Boxes: [numBoxes, 4] with [y1, x1, y2, x2]
            SDVariable boxes = sd.placeHolder("boxes", DataType.FLOAT, -1, 4);
            // Scores: [numBoxes]
            SDVariable scores = sd.placeHolder("scores", DataType.FLOAT, -1);

            // NMS parameters: maxOutputSize, iouThreshold, scoreThreshold
            SDVariable selected = sd.image().nonMaxSuppression("selected", boxes, scores,
                    10,       // max output boxes
                    0.5,      // IoU threshold
                    0.3);     // score threshold

            // Simulate 20 detection boxes
            INDArray boxData = Nd4j.rand(DataType.FLOAT, 20, 4);
            INDArray scoreData = Nd4j.rand(DataType.FLOAT, 20);
            java.util.HashMap<String, INDArray> placeholders = new java.util.HashMap<>();
            placeholders.put("boxes", boxData);
            placeholders.put("scores", scoreData);

            Map<String, INDArray> result = sd.output(placeholders, "selected");
            System.out.println("  20 input boxes -> NMS selected indices: " + result.get("selected"));
            System.out.println("  IoU threshold: 0.5, Score threshold: 0.3, Max output: 10");
        }

        // ============================================================
        // 10. PAD - Image padding
        // ============================================================
        System.out.println("\n=== Image Padding ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable image = sd.placeHolder("image", DataType.FLOAT, -1, 28, 28, 1);

            // Padding: [rank, 2] specifying [before, after] for each dimension
            // Pad 2 pixels on each side of H and W, no padding on batch or channel
            SDVariable padding = sd.constant("padding", Nd4j.createFromArray(new int[][]{
                    {0, 0},  // batch: no padding
                    {2, 2},  // height: +2 top, +2 bottom
                    {2, 2},  // width: +2 left, +2 right
                    {0, 0}   // channels: no padding
            }));

            // Constant padding with value 0.0
            SDVariable padded = sd.image().pad("padded", image, padding,
                    org.nd4j.enums.Mode.CONSTANT, 0.0);

            INDArray imageData = Nd4j.rand(DataType.FLOAT, 1, 28, 28, 1);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("image", imageData), "padded");
            System.out.println("  28x28 + 2px padding -> " + result.get("padded").shapeInfoToString());
        }

        // ============================================================
        // PIPELINE: Data augmentation for training
        // ============================================================
        System.out.println("\n=== Image Augmentation Pipeline ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 256, 256, 3);

            // Step 1: Random crop to 224x224
            SDVariable cropShape = sd.constant("cs",
                    Nd4j.createFromArray(1, 224, 224, 3).castTo(DataType.INT));
            SDVariable cropped = sd.image().randomCrop("aug_crop", input, cropShape);

            // Step 2: Random color augmentation
            SDVariable augContrast = sd.image().adjustContrast("aug_contrast", cropped, 1.2);
            SDVariable augHue = sd.image().adjustHue("aug_hue", augContrast, 0.05);
            SDVariable augSat = sd.image().adjustSaturation("aug_sat", augHue, 1.1);

            // Step 3: Convert to HSV for feature extraction
            SDVariable hsvFeatures = sd.image().rgbToHsv("aug_hsv", augSat);

            System.out.println("  Augmentation pipeline: crop -> contrast -> hue -> saturation -> HSV");
            System.out.println("  All operations are differentiable for end-to-end training");
        }

        System.out.println("\nAll image operations demonstrated successfully.");
    }
}

/*******************************************************************************
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

package org.deeplearning4j.examples.advanced.modelling.densenet.imageUtils;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;

public class BlurTransform extends BaseImageTransform<Mat> {

    private int kSize;
    private double deviation;

    public BlurTransform(int kSize, double deviation) {
        this(null, kSize, deviation);
        this.converter = new OpenCVFrameConverter.ToMat();
    }


    public BlurTransform(Random random, int kSize, double deviation) {
        super(random);
        this.kSize = kSize;
        this.deviation = deviation;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());
        Mat result = new Mat();
        int value = kSize % 2 == 0 ? kSize + 1 : kSize;
        try {
            GaussianBlur(mat, result, new Size(value, value), random != null ? random.nextDouble() * deviation : deviation);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        return coordinates;
    }
}

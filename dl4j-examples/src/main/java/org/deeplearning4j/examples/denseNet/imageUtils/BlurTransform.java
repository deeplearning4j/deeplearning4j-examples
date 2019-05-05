package org.deeplearning4j.examples.denseNet.imageUtils;

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

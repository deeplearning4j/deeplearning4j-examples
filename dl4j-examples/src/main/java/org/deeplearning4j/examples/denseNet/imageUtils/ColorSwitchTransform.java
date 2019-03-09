package org.deeplearning4j.examples.denseNet.imageUtils;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class ColorSwitchTransform extends BaseImageTransform<Mat> {

    private int[] colorCodes;

    public ColorSwitchTransform(int... colorCodes) {
        this(null, colorCodes);
    }


    public ColorSwitchTransform(Random random, int... colorCodes) {
        super(random);
        this.colorCodes = colorCodes;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());
        Mat result = new Mat();

        try {
            cvtColor(mat, result, random != null ? colorCodes[random.nextInt(colorCodes.length)] : colorCodes[0]);
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

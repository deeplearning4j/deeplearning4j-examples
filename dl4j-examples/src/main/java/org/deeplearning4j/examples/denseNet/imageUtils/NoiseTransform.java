package org.deeplearning4j.examples.denseNet.imageUtils;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

public class NoiseTransform extends BaseImageTransform<Mat> {

    private int numPix;

    public NoiseTransform(int numPix) {
        this(null, numPix);
    }

    public NoiseTransform(Random random, int numPix) {
        super(random);
        this.numPix = numPix;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());

        int nbChannels = mat.channels();
        UByteIndexer idx = mat.createIndexer();

        for (int i = 0; i < numPix; i++) {
            int row = random.nextInt(mat.rows());
            int col = random.nextInt(mat.cols());
            for (int j = 0; j < nbChannels; j++) {
                //int current = idx.get(row, col);
                //idx.put(row, col, j, random.nextInt(getMaxValue(current) - getMinValue(current) + 1) + getMinValue(current));
                idx.put(row, col, j, random.nextInt(255));
            }
        }
        return new ImageWritable(converter.convert(mat));
    }

    @Override
    public float[] query(float... coordinates) {
        return coordinates;
    }
}

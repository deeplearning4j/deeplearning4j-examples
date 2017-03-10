package org.deeplearning4j.examples.multigpu.crash;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author raver119@gmail.com
 */
public class CrashIterator implements MultiDataSetIterator {
    @Override
    public MultiDataSet next(int num) {
        INDArray featutres[] = new INDArray[]{Nd4j.create(num, 814)};
        INDArray labels[] = new INDArray[4];

        INDArray labels2 = Nd4j.create(num, 2).addiRowVector(Nd4j.create(new float[]{1.0f, 0.0f}));
        INDArray labels3 = Nd4j.create(num, 3).addiRowVector(Nd4j.create(new float[]{1.0f, 0.0f, 1.0f}));
        INDArray labels13 = Nd4j.create(num, 13).addiRowVector(Nd4j.create(13).putScalar(0, 1.0));

        for (int i = 0; i < 4; i++)
            labels[i] = labels2.dup();

        labels[2] = labels3.dup();
        labels[3] = labels13.dup();

        org.nd4j.linalg.dataset.MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(featutres, labels, null, null);
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {

    }

    @Override
    public boolean hasNext() {
        return true;
    }

    @Override
    public MultiDataSet next() {
        return next(32);
    }

    @Override
    public void remove() {

    }
}

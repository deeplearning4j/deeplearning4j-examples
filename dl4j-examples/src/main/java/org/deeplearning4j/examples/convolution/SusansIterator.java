package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SusansIterator implements MultiDataSetIterator {
    private AtomicInteger counter = new AtomicInteger(0);
    private int numBatches;
    private int batchSize;

    private int shapes[] = new int[]{8469, 9173, 8403, 8855, 8517, 9062, 9013, 8801, 8910, 8820, 8854, 8787, 8669, 9048, 8819, 8500, 8858, 857, 8642, 8361,  9141, 9026, 9087, 8918, 8796, 8709, 8721, 8721, 8891, 8921, 8983, 8624, 8956, 8831, 8963, 8959, 9083, 8726, 9110, 9035, 9094, 9038, 8865, 8259, 8548, 8510};

    public SusansIterator(int numBatches, int batchSize) {
        this.numBatches = numBatches;
        this.batchSize = batchSize;
    }

    @Override
    public MultiDataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
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
        counter.set(0);
    }

    @Override
    public boolean hasNext() {
        return counter.get() < shapes.length;
    }

    @Override
    public MultiDataSet next() {

        int localMaxima = shapes[counter.get()];//7500 + counter.get(); //RandomUtils.nextInt(7500, 9000);

        log.info("Creating dataset for shape: {}", localMaxima);

        int[] shapeFeatures = new int[]{batchSize, 20, localMaxima};
        int[] shapeLabels = new int[] {batchSize, 2, localMaxima};
        int[] shapeFMasks = new int[] {batchSize, localMaxima};
        int[] shapeLMasks = new int[] {batchSize, localMaxima};

        log.info("Allocating dataset seqnum: {}", counter.get());

        INDArray features = Nd4j.create(shapeFeatures).assign(counter.get());
        INDArray labels = Nd4j.create(shapeLabels).assign(counter.get() + 0.25);
        INDArray fMasks = Nd4j.create(shapeFMasks).assign(counter.get() + 0.50);
        INDArray lMasks = Nd4j.create(shapeLMasks).assign(counter.get() + 0.75);
/*
        INDArray features = Nd4j.rand(shapeFeatures).assign(counter.get());
        INDArray labels = Nd4j.rand(shapeLabels).assign(counter.get() + 0.25);
        INDArray fMasks = Nd4j.rand(shapeFMasks).assign(counter.get() + 0.50);
        INDArray lMasks = Nd4j.rand(shapeLMasks).assign(counter.get() + 0.75);
*/
        counter.getAndIncrement();

        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{features}, new INDArray[]{labels.dup(), labels.dup(), labels.dup(), labels.dup()}, new INDArray[]{fMasks}, new INDArray[]{lMasks.dup(), lMasks.dup(), lMasks.dup(), lMasks.dup()});
        //return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{features}, new INDArray[]{labels, labels, labels, labels}, new INDArray[]{fMasks}, new INDArray[]{lMasks, lMasks, lMasks, lMasks});
    }

    @Override
    public void remove() {

    }
}

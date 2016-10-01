package org.deeplearning4j.examples.convolution.debug;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class DummyIterator implements DataSetIterator {
    private int[] shapeFeatures;
    private int[] shapeLabels;
    private int batchSize;


    public DummyIterator(int[] shapeFeatures, int[] shapeLabels) {
        this.shapeFeatures = shapeFeatures;
        this.shapeLabels = shapeLabels;
    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    @Override
    public DataSet next(int num) {
        return null;
    }

    /**
     * Total examples in the iterator
     *
     * @return
     */
    @Override
    public int totalExamples() {
        return 0;
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return 0;
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return 0;
    }

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    @Override
    public boolean resetSupported() {
        return true;
    }

    /**
     * Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
     * Most DataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
     * iterator that does asynchronous prefetching. For example, it would not make sense to use asynchronous
     * prefetching for the following types of iterators:
     * (a) Iterators that store their full contents in memory already
     * (b) Iterators that re-use features/labels arrays (as future next() calls will overwrite past contents)
     * (c) Iterators that already implement some level of asynchronous prefetching
     * (d) Iterators that may return different data depending on when the next() method is called
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    @Override
    public boolean asyncSupported() {
        return true;
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {

    }

    /**
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return 0;
    }

    /**
     * The current cursor if applicable
     *
     * @return
     */
    @Override
    public int cursor() {
        return 0;
    }

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    @Override
    public int numExamples() {
        return 0;
    }

    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    /**
     * Returns preprocessors, if defined
     *
     * @return
     */
    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    /**
     * Get dataset iterator record reader labels
     */
    @Override
    public List<String> getLabels() {
        return null;
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return true;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public DataSet next() {

        INDArray features = Nd4j.zeros(shapeFeatures);
        INDArray labels = Nd4j.rand('c', shapeLabels);

        return new DataSet(features, labels);
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {

    }
}

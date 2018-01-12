package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**simple iterator provide unsupervised traning and testing
 * @author wangfeng
 */
public class MnistIterator implements DataSetIterator {

    private int batchSize = 0;
    private int batchNum = 0;
    private int numExample = 0;
    private MnistLoader load;
    private DataSetPreProcessor preProcessor;

    public MnistIterator() {
        load = new MnistLoader();
    }
    public MnistIterator(int batchSize, boolean train) {
        this.batchSize = batchSize;
        load = new MnistLoader(train);
        numExample = load.totalExamples();
    }

    @Override
    public DataSet next(int i) {
        batchNum += i;
        DataSet ds = load.next(i);
        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }
        return ds;
    }

    @Override
    public int totalExamples() {
        return numExample;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        batchNum = 0;
        load.reset();
    }
    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        if(batchNum < numExample){
            return true;
        } else {
            return false;
        }
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}

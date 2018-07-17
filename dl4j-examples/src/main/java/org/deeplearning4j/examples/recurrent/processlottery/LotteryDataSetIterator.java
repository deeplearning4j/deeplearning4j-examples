package org.deeplearning4j.examples.recurrent.processlottery;



import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.File;
import java.util.*;

/**
 * @author WangFeng
 */
public class LotteryDataSetIterator implements DataSetIterator {

    private BaseDataSetReader recordReader;
    private int batchSize;
    private DataSet last;
    private boolean useCurrent;

    public LotteryDataSetIterator(String filePath, int batchSize, boolean modelType) {
        this.recordReader = modelType? new LotteryCombinationDataSetReader(new File(filePath)): new LotteryCharacterSequenceDataSetReader(new File(filePath));
        this.batchSize = batchSize;
    }

    @Override
    public DataSet next(int i) {
        return recordReader.next(i);
    }

    @Override
    public int totalExamples() {
        return recordReader.totalExamples();
    }

    @Override
    public int inputColumns() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        } else {
            return last.numInputs();
        }
    }

    @Override
    public int totalOutcomes() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        } else {
            return last.numOutcomes();
        }
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
        recordReader.reset();
        last = null;
        useCurrent = false;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return recordReader.cursor();
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not support the function");
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public DataSet next() {
        if (useCurrent) {
            useCurrent = false;
            return last;
        } else {
            return next(batchSize);
        }
    }

}

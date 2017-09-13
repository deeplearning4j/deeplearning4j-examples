package org.deeplearning4j.examples;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Stack;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.examples.conf.Constants;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * This class returns a MultiDataSet each time its {@link next} method is called. The data in HDFS is structured as
 * follows: A main directory which contains three sub-folders: /train, /test, and /predict. Depending on the value set
 * for the variable flag, this class will iterate over one of the sub-folders and create MultiDataSets. In the
 * sub-folders, each HDFS file consists of rows representing time-steps (as in an LSTM) starting with a time index. Each
 * file is therefore an input sequence (as in an LSTM). You must specify the time index for this to re-order the
 * sequence or pad it in case some time-steps are missing.
 *
 * @author: Ousmane A. Dia
 */
public class MDSIterator extends BaseDataSetIterator implements MultiDataSetIterator {

    private int vectorSize = 0;
    private int labelSize = 0;
    private final int batchSize;
    private int numSteps = 6;
    private Stack<Path> stack = new Stack<Path>();
    private MultiDataSetPreProcessor preProcessor;

    private StackSequenceRecordReader ssRecordReader;

    private static final long serialVersionUID = -2132071188514707198L;

    /**
     *
     * @param dataDirectory Hadoop directory that contains either training/validation/testing data. The dataDirectory
     *        should be of the following form: hdfs://your_cluster:port/folder
     * @param batchSize Size of the mini batch
     * @param vectorSize Number of features of your embeddings
     * @param labelSize Number of classes
     * @param numSteps
     * @param flag Takes values 0 (training, 1 (validation), 2 (testing) and helps set the start and end of your
     *        sequence.
     */
    public MDSIterator(Configuration configuration, String dataDirectory, int batchSize, int vectorSize, int labelSize,
                    int numSteps, int flag) {
        super(configuration, dataDirectory);
        this.batchSize = batchSize;
        this.vectorSize = vectorSize;
        this.labelSize = labelSize;
        int start = Constants.START_SEQ;
        int end = Constants.END_SEQ;
        start = flag == 2 ? start + 1 : start;
        end = flag == 2 ? end : end - 1;
        ssRecordReader = new StackSequenceRecordReader(fs, start, end);
        this.numSteps = numSteps;
    }

    @Override
    public boolean hasNext() {
        try {
            return hdfsIterator != null && hdfsIterator.hasNext();
        } catch (IOException e) {
            return false;
        }
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public MultiDataSet next(int num) {
        try {
            if (!hdfsIterator.hasNext())
                throw new NoSuchElementException();
            MultiDataSet mds = nextMultiDataSet(num);
            while (mds == null && hdfsIterator.hasNext()) {
                mds = nextMultiDataSet(num);
            }
            return mds;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    private void pushAndClear(Path path, String index) {
        String p = stack.isEmpty() ? "" : stack.peek().toUri().toString();
        if (p.contains(index.split("_")[0])) {
            stack.push(path);
        } else {
            ssRecordReader.newRecord(stack);
            stack.push(path);
        }
        ssRecordReader.newRecord(stack);
    }

    private MultiDataSet nextMultiDataSet(int num) throws IOException {
        String previousPath = stack.isEmpty() ? "" : stack.peek().toUri().getPath();

        for (int i = 0; i < num && hdfsIterator.hasNext(); i++) {
            for (int j = 0; j < numSteps; j++) {
                if (!hdfsIterator.hasNext())
                    break;
                LocatedFileStatus next = hdfsIterator.next();
                Path path = next.getPath();

                String currentPath = path.toUri().getPath();
                String index = getRelativeFilename(currentPath);

                if (previousPath.contains(index.split("_")[0])) {
                    if (j >= numSteps - 1 || !hdfsIterator.hasNext()) {
                        pushAndClear(path, index);
                    } else {
                        stack.push(path);
                    }
                    previousPath = currentPath;
                } else {
                    if (j >= numSteps - 1 || !hdfsIterator.hasNext()) {
                        pushAndClear(path, index);
                    }
                    ssRecordReader.newRecord(stack);
                    stack.push(path);
                    if (!previousPath.isEmpty()) {
                        break;
                    }
                    previousPath = currentPath;
                }
            }
        }
        return ssRecordReader.toMultiDataSet(vectorSize, labelSize);
    }

    @Override
    public void reset() {
        super.initIterator(hdfsUrl);
    }

    @Override
    public boolean resetSupported() {
        return true;
    }


    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove not yet supported");
    }
}

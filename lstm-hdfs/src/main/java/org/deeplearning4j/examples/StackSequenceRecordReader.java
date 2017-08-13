package org.deeplearning4j.examples;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.examples.conf.Constants;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author: Ousmane A. Dia
 */
public class StackSequenceRecordReader extends FileRecordReader {

    private Stack<List<Path>> sequenceStack = new Stack<List<Path>>();
    private FileSystem fs;

    private List<Double> featureMasks, labelMasks;

    private static final long serialVersionUID = 2949161683514893388L;
    private CSVSequenceRecordReader csvRecordReader = new CSVSequenceRecordReader();

    private List<String> timeSteps = new ArrayList<String>();
    private int timeIndex = Constants.TIME_INDEX;

    private static final Logger LOG = LoggerFactory.getLogger(StackSequenceRecordReader.class);

    public StackSequenceRecordReader(FileSystem fs, int startSeq, int endSeq) {
        if (startSeq > endSeq)
            throw new RuntimeException("Start is greater than End.");
        this.fs = fs;
        for (int i = startSeq; i <= endSeq; i++) {
            timeSteps.add(String.valueOf(i));
        }
    }

    public void newRecord(Stack<Path> pathStack) {
        newRecord(pathStack, true);
    }

    public void newRecord(Stack<Path> pathStack, boolean reverse) {

        List<Path> paths = new ArrayList<Path>();

        if (pathStack.isEmpty())
            return;
        while (!pathStack.isEmpty()) {
            paths.add(pathStack.pop());
        }
        if (reverse)
            Collections.reverse(paths);
        this.sequenceStack.push(paths);
    }

    public boolean hasNext() {
        return !sequenceStack.isEmpty() && !sequenceStack.peek().isEmpty();
    }

    @Override
    public List<Writable> next() {
        featureMasks = new ArrayList<Double>(Collections.nCopies(timeSteps.size(), 0.0));
        labelMasks = new ArrayList<Double>(Collections.nCopies(timeSteps.size(), 0.0));
        List<Writable> writable = new ArrayList<Writable>();
        List<Path> paths = sequenceStack.pop();
        DataInputStream in = null;
        for (Path p : paths) {
            try {
                in = fs.open(p);
                URI uri = p.toUri();

                String pathParts[] = uri.toString().split("_");
                String currStep = pathParts == null || pathParts.length < 3 ? "" : pathParts[2];
                int index = timeSteps.indexOf(currStep);
                index = index < 0 ? timeSteps.size() - 1 : index;

                List<String> seq = new ArrayList<String>();
                List<List<Writable>> steps = csvRecordReader.sequenceRecord(uri, in);
                if (!CollectionUtils.isEmpty(steps)) {
                    for (List<Writable> instance : steps) {
                        if (timeIndex >= 1 && timeIndex <= instance.size()) {
                            seq.add(instance.get(timeIndex - 1).toString());
                            if (timeIndex - 1 > 0)
                                writable.addAll(instance.subList(0, timeIndex - 1));
                            writable.addAll(instance.subList(timeIndex, instance.size()));
                        } else {
                            writable.addAll(instance);
                        }
                    }
                    if (!seq.isEmpty()) {
                        String maxIndex = seq.get(0);
                        for (int i = 0; i < seq.size(); i++) {
                            if (Integer.parseInt(seq.get(i)) > Integer.parseInt(maxIndex))
                                maxIndex = seq.get(i);
                            featureMasks.set(timeSteps.indexOf(seq.get(i)), 1.0);
                        }
                        labelMasks.set(timeSteps.indexOf(maxIndex),
                                        timeSteps.get(timeSteps.size() - 1) == maxIndex || paths.size() == 1 ? 1.0
                                                        : 0.0);
                    } else {
                        featureMasks.set(index, 1.0);
                        labelMasks.set(index, timeSteps.get(timeSteps.size() - 1) == currStep || paths.size() == 1 ? 1.0
                                        : 0.0);
                    }
                }
            } catch (IOException e) {
                LOG.error(e.getLocalizedMessage(), e);
            } finally {
                if (in != null) {
                    IOUtils.closeQuietly(in);
                }
            }
        }
        return writable;
    }

    private List<List<Writable>> records(List<List<Double>> lMasks, List<List<Double>> rMasks) {
        List<List<Writable>> writables = new ArrayList<List<Writable>>();
        while (hasNext()) {
            List<Writable> w = next();
            if (w.isEmpty())
                continue;
            writables.add(w);
            lMasks.add(featureMasks);
            rMasks.add(labelMasks);
        }
        return writables;
    }

    private MultiDataSet zeroMDS(int numFeatures, int numLabels) {
        return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {Nd4j.create(new double[numFeatures])},
                        new INDArray[] {Nd4j.create(new double[numLabels])},
                        new INDArray[] {Nd4j.create(new double[1])}, new INDArray[] {Nd4j.create(new double[1])});
    }

    public MultiDataSet toMultiDataSet(int numFeatures, int numLabels) {
        List<List<Double>> featureMasks = new ArrayList<List<Double>>();
        List<List<Double>> labelMasks = new ArrayList<List<Double>>();
        List<List<Writable>> records = records(featureMasks, labelMasks);

        List<MultiDataSet> list = new ArrayList<MultiDataSet>(records.size());

        int currHotIndex = 0;
        for (int i = 0; i < records.size(); i++) {

            Pair<INDArray[], INDArray[]> pair = toPair(records.get(i), numFeatures, numLabels);

            for (int j = 0; j < timeSteps.size(); j++) {
                if (featureMasks.get(i).get(j) != 1.0) {
                    list.add(zeroMDS(numFeatures, numLabels));
                } else {
                    INDArray feature = pair.getFirst()[currHotIndex];
                    INDArray label = pair.getSecond()[currHotIndex++];
                    INDArray fMask = Nd4j.create(new double[] {featureMasks.get(i).get(j)});
                    INDArray lMask = Nd4j.create(new double[] {labelMasks.get(i).get(j)});

                    list.add(new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {feature}, new INDArray[] {label},
                                    new INDArray[] {fMask}, new INDArray[] {lMask}));
                }
            }
            currHotIndex = 0;
        }
        return list.isEmpty() ? null : org.nd4j.linalg.dataset.MultiDataSet.merge(list);
    }

    private Pair<INDArray[], INDArray[]> toPair(Collection<Writable> record, int numFeatures, int numLabels) {
        Iterator<Writable> writables = record.iterator();
        Writable firstWritable = writables.next();
        INDArray vector1 = Nd4j.create(numFeatures);
        INDArray vector2 = Nd4j.create(numLabels);

        INDArray[] array1 = new INDArray[record.size() / (numFeatures + numLabels)];
        INDArray[] array2 = new INDArray[record.size() / (numFeatures + numLabels)];

        vector1.putScalar(0, firstWritable.toDouble());

        int count1 = 1, count2 = 0, i = 0;
        while (writables.hasNext()) {
            Writable w = writables.next();
            if (count1 < numFeatures) {
                double val = Double.isNaN(w.toDouble()) ? 0.0 : w.toDouble();
                vector1.putScalar(count1++, val);
            } else {
                if (count2 < numLabels) {
                    double val = Double.isNaN(w.toDouble()) ? 0.0 : w.toDouble();
                    vector2.putScalar(count2++, val);
                }
            }
            if (count1 == numFeatures && count2 == numLabels) {
                array1[i] = vector1;
                array2[i++] = vector2;
                count1 = 0;
                count2 = 0;
            }
        }
        return new Pair<INDArray[], INDArray[]>(array1, array2);
    }
}

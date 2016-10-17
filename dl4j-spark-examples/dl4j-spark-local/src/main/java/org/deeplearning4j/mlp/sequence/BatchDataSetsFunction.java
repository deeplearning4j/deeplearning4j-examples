package org.deeplearning4j.mlp.sequence;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Function used to batch DataSet objects together.
 * Use like:
 * <pre>
 * {@code
 *      RDD<DataSet> mySingleExampleDataSets = ...;
 *      RDD<DataSet> batchData = mySingleExampleDataSets.mapPartitions(new BatchDataSetsFunction(batchSize));
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class BatchDataSetsFunction implements FlatMapFunction<Iterator<DataSet>,DataSet> {
    private final int minibatchSize;

    public BatchDataSetsFunction(int minibatchSize) {
        this.minibatchSize = minibatchSize;
    }

    @Override
    public Iterable<DataSet> call(Iterator<DataSet> iter) throws Exception {
        List<DataSet> out = new ArrayList<>();
        while(iter.hasNext()) {
            List<org.nd4j.linalg.dataset.DataSet> list = new ArrayList<>();

            int count = 0;
            while (count < minibatchSize && iter.hasNext()) {
                org.nd4j.linalg.dataset.DataSet ds = iter.next();
                count += ds.getFeatureMatrix().size(0);
                list.add(ds);
            }

            DataSet next;
            if (list.size() == 0) next = list.get(0);
            else next = org.nd4j.linalg.dataset.DataSet.merge(list);

            out.add(next);
        }
        return out;
    }
}

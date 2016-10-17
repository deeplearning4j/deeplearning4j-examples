package org.deeplearning4j.mlp.sequence;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.ByteArrayOutputStream;
import java.util.UUID;

/**
 * Created by Alex on 01/08/2016.
 */
public class ToSequenceFilePairFunction implements PairFunction<DataSet,Text,BytesWritable> {
    @Override
    public Tuple2<Text, BytesWritable> call(DataSet dataSet) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        dataSet.save(baos);
        byte[] bytes = baos.toByteArray();

        return new Tuple2<>(new Text(UUID.randomUUID().toString()), new BytesWritable(bytes));
    }
}

package org.deeplearning4j.mlp.sequence;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.ByteArrayInputStream;

/**
 * Created by Alex on 01/08/2016.
 */
public class FromSequenceFilePairFunction implements Function<Tuple2<Text,BytesWritable>,DataSet> {
    @Override
    public DataSet call(Tuple2<Text, BytesWritable> v1) throws Exception {
        DataSet ds = new DataSet();
        ByteArrayInputStream bais = new ByteArrayInputStream(v1._2().getBytes());
        ds.load(bais);
        return ds;
    }
}

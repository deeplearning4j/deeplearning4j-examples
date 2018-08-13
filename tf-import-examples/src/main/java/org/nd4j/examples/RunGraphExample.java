package org.nd4j.examples;

import com.github.os72.protobuf351.util.JsonFormat;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs a tensorflow graph using the tensorflow graph runner.
 *
 * @author Adam Gibson
 */
public class RunGraphExample {

    public static void main(String...args) throws Exception {
        //input name (usually with place holders)
        List<String> inputs = Arrays.asList("flatten_2_input");
        //load the graph from the classpath
        byte[] content = IOUtils.toByteArray(new ClassPathResource("Mnist/mnist.pb").getInputStream());
        DataSetIterator dataSetIterator = new MnistDataSetIterator(1,1);
        INDArray predict = dataSetIterator.next().getFeatures();
        //run the graph using nd4j
        try(GraphRunner graphRunner = new GraphRunner(content,inputs)) {
            Map<String,INDArray> inputMap = new HashMap<>();
            inputMap.put(inputs.get(0),predict);
            Map<String, INDArray> run = graphRunner.run(inputMap);
            System.out.println("Run result " + run);
        }

    }
}

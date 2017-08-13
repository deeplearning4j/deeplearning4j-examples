package org.deeplearning4j.examples.inference;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

/**
 * This examples shows use of ParallelInference mechanism
 *
 * @author raver119@gmail.com
 */
public class ParallelInferenceExample {

    public static void main(String[] args) throws Exception {

        // use path to your model here, or just instantiate it anywhere
        Model model = ModelSerializer.restoreComputationGraph("PATH_TO_YOUR_MODEL_FILE", false);

        ParallelInference pi = new ParallelInference.Builder(model)
            // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
            .inferenceMode(InferenceMode.BATCHED)

            // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
            .batchLimit(32)

            // set this value to number of available computational devices, either CPUs or GPUs
            .workers(2)

            .build();


        // PLEASE NOTE: this output() call is just a placeholder, you should pass data in the same dimensionality you had during training
        INDArray result = pi.output(new float[] {0.1f, 0.1f, 0.1f, 0.2f, 0,3f });
    }
}

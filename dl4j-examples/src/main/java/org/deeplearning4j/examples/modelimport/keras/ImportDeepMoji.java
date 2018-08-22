package org.deeplearning4j.examples.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URL;

import static java.io.File.createTempFile;

/**
 * Test import of DeepMoji application
 *
 * @author Max Pumperla
 */
public class ImportDeepMoji {

    public static void main(String[] args) throws Exception {
        // First, register the Keras layer wrapped around our custom SameDiff attention layer
        KerasLayer.registerCustomLayer("AttentionWeightedAverage", KerasDeepMojiAttention.class);

        // Then, download the model from azure (check if it's cached)
        String modelUrl = "http://blob.deeplearning4j.org/models/deepmoji.h5";
        File cachedKerasFile = createTempFile("deepmoji", ".h5");

        if (!cachedKerasFile.exists()) {
            System.out.println("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(modelUrl), cachedKerasFile);
            cachedKerasFile.deleteOnExit();
        }

        // Finally, import the model and test on artificial inputs.
        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());;
        INDArray input = Nd4j.create(new int[] {10, 30});
        graph.output(input);
    }
}

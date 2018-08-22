package org.deeplearning4j.examples.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
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
@Slf4j
public class ImportDeepMoji {

    public static void main(String[] args) throws Exception {
        //KerasLayer.registerCustomLayer("AttentionWeightedAverage", KerasDeepMojiAttention.class);

        String modelUrl = "http://blob.deeplearning4j.org/models/deepmoji.h5";
        File cachedKerasFile = createTempFile("deepmoji", ".h5");


        if (!cachedKerasFile.exists()) {
            log.info("Downloading model to " + cachedKerasFile.toString());
            FileUtils.copyURLToFile(new URL(modelUrl), cachedKerasFile);
            cachedKerasFile.deleteOnExit();
        }

        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(cachedKerasFile.getAbsolutePath());;

        INDArray input = Nd4j.create(new int[] {10, 30});
        graph.output(input);
    }
}

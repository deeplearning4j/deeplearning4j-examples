package org.deeplearning4j.examples.modelimport.keras;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.io.ClassPathResource;

/**
 * Simple example showing how to load resnet50 from keras into dl4j
 */
public class LoadResNet50 {

    public final static String MODEL_PATH = "modelimport/keras/resnet50_weights_tf_dim_ordering_tf_kernels.h5";

    public static void main(String[] args) throws Exception {

        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(new ClassPathResource(MODEL_PATH).getFile().getPath());

        System.out.println(model.summary());

    }
}

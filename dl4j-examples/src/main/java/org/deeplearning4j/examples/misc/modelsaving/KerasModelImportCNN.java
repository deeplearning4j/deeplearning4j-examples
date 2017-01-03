package org.deeplearning4j.examples.misc.modelsaving;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.string.NDArrayStrings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class KerasModelImportCNN {

    private static final Logger log = LoggerFactory.getLogger(KerasModelImportCNN.class);

    public static void main(String[] args) throws Exception {

        String jsnPath =  "/Users/susaneraly/SKYMIND/kerasImport/examples/mnist_model.json";
        String hdfPath =  "/Users/susaneraly/SKYMIND/kerasImport/examples/mnist_model.h5";
        String validationPath = "/Users/susaneraly/SKYMIND/kerasImport/examples/validation_28x28.txt";

        //MultiLayerNetwork model = KerasModelImport.importSequentialModel(jsnPath, hdfPath);
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(jsnPath,hdfPath);

        // Load the validation image
        INDArray img = Nd4j.readNumpy(validationPath, " ");
        List<INDArray> imgs = new ArrayList<INDArray>();
        imgs.add(img);
        int[] shape = {1,1,28,28};
        INDArray img4d = Nd4j.create(imgs, shape);

        // Get class probabilities
        INDArray output = model.output(img4d, false);
        log.info("{}", output);
        System.out.println(new NDArrayStrings(6).format(output));

    }
}

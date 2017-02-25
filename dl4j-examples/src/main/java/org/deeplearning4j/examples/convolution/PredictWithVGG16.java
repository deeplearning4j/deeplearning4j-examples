package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * Created by raver119 on 22.02.17.
 */
@Slf4j
public class PredictWithVGG16 {
    public static void main(String [] args) throws Exception {


/*
        //Helper for trained deep learning models
        TrainedModelHelper helper = new TrainedModelHelper(TrainedModels.VGG16);

        //Load the model into dl4j
        ComputationGraph vgg16 = helper.loadModel();

*/

        String modelJsonFilename = "/home/raver119/Downloads/inception.json";
        String weightsHdf5Filename = "/home/raver119/Downloads/inception_v3_weights_th_dim_ordering_th_kernels.h5";

        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelJsonFilename, weightsHdf5Filename, false);

        // Make the image dataset iterator and ttach the VGG16 specific preprocessor to the dataset iterator for the mean shifting required
          DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
        //  dataIter.setPreProcessor(preProcessor);

        NativeImageLoader loader = new NativeImageLoader(299, 299, 3);
        INDArray image = loader.asMatrix(new File("/home/raver119/tank.jpg"));
//        DataSet ds = new DataSet(image, Nd4j.createUninitialized(10));
//        preProcessor.preProcess(ds);

        long time1 = 0;
        long time2 = 0;
        log.info("Going for it");
        for (int i = 0; i < 100; i++) {
            time1 = System.currentTimeMillis();
            model.output(image);
            time2 = System.currentTimeMillis();
        }
        log.info("Last time: {} ms", time2 - time1);

    }
}

package org.deeplearning4j.examples.modelimport;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;

import java.io.IOException;

/**
 * Created by susaneraly on 2/28/17.
 */
@Slf4j
public class TransferLearningExample {

    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
        ComputationGraph vgg16 = modelImportHelper.loadModel();

        //print model summary - will gave layer names, nIn, nOut etc
        log.info(vgg16.summary());

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16).setFeatureExtractor("fc1").build;

        log.info(vgg16Transfer.summary());

    }
}

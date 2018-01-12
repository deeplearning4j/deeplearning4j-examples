package org.deeplearning4j.examples.multigpu.vgg16.vgg16;

import org.deeplearning4j.examples.multigpu.vgg16.dataHelpers.FeaturizedPreSave;
import org.deeplearning4j.examples.multigpu.vgg16.dataHelpers.FlowerDataSetIteratorFeaturized;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;

/**
 * @author susaneraly on 3/10/17.
 *
 * Important:
 * Run the class "FeaturizePreSave" before attempting to run this. The outputs at the boundary of the frozen and unfrozen
 * vertices of a model are saved. These are referred to as "featurized" datasets in this description.
 * On a dataset of about 3000 images which is what is downloaded this can take "a while"
 *
 * Here we see how the transfer learning helper can be used to fit from a featurized datasets.
 * We attempt to train the same model architecture as the one in "EditLastLayerOthersFrozen".
 * Since the helper avoids the forward pass through the frozen layers we save on computation time when running multiple epochs.
 * In this manner, users can iterate quickly tweaking learning rates, weight initialization etc` to settle on a model that gives good results.
 */
public class FitFromFeaturized {

    public static final String featureExtractionLayer = FeaturizedPreSave.featurizeExtractionLayer;
    protected static final long seed = 12345;
    protected static final int numClasses = 5;
    protected static final int nEpochs = 3;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FitFromFeaturized.class);

    public static void main(String [] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        // temp workaround for backend initialization

        CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
            .allowMultiGPU(true)

            // we're allowing larger memory caches
            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

            // cross-device access is used for faster model averaging over pcie
            .allowCrossDeviceAccess(true);

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(3e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();
        log.info(vgg16Transfer.summary());

        DataSetIterator trainIter = FlowerDataSetIteratorFeaturized.trainIterator();
        DataSetIterator testIter = FlowerDataSetIteratorFeaturized.testIterator();
        System.out.println("Env information " + Nd4j.getExecutioner().getEnvironmentInformation());
         //Instantiate the transfer learning helper to fit and output from the featurized dataset
        //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
        //If using with a UI or a listener attach them directly to the unfrozenGraph instance
        //With each iteration updated params from unfrozenGraph are copied over to the original model
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
        log.info(transferLearningHelper.unfrozenGraph().summary());
        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(transferLearningHelper.unfrozenGraph())
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(24)

            // set number of workers equal to number of available devices
            .workers(Nd4j.getAffinityManager().getNumberOfDevices())

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(3)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(true)

            .build();

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            wrapper.fit(trainIter);
            trainIter.reset();
            log.info("Epoch #" + epoch +" complete");
        }
        log.info("Model build complete");
    }
}

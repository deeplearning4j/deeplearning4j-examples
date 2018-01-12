package org.deeplearning4j.examples.transferlearning.vgg16;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FeaturizedPreSave;
import org.deeplearning4j.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIteratorFeaturized;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FitFromFeaturized.class);

    public static final String featureExtractionLayer = FeaturizedPreSave.featurizeExtractionLayer;
    protected static final long seed = 12345;
    protected static final int numClasses = 5;
    protected static final int nEpochs = 3;

    public static void main(String [] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

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
                            .weightInit(WeightInit.DISTRIBUTION)
                            .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                            .activation(Activation.SOFTMAX).build(),
                       "fc2")
            .build();
        log.info(vgg16Transfer.summary());

        DataSetIterator trainIter = FlowerDataSetIteratorFeaturized.trainIterator();
        DataSetIterator testIter = FlowerDataSetIteratorFeaturized.testIterator();

        //Instantiate the transfer learning helper to fit and output from the featurized dataset
        //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
        //If using with a UI or a listener attach them directly to the unfrozenGraph instance
        //With each iteration updated params from unfrozenGraph are copied over to the original model
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
        log.info(transferLearningHelper.unfrozenGraph().summary());

        for (int epoch = 0; epoch < nEpochs; epoch++) {
            if (epoch == 0) {
                Evaluation eval = transferLearningHelper.unfrozenGraph().evaluate(testIter);
                log.info("Eval stats BEFORE fit.....");
                log.info(eval.stats()+"\n");
                testIter.reset();
            }
            int iter = 0;
            while (trainIter.hasNext()) {
                transferLearningHelper.fitFeaturized(trainIter.next());
                if (iter % 10 == 0) {
                    log.info("Evaluate model at iter " + iter + " ....");
                    Evaluation eval = transferLearningHelper.unfrozenGraph().evaluate(testIter);
                    log.info(eval.stats());
                    testIter.reset();
                }
                iter++;
            }
            trainIter.reset();
            log.info("Epoch #"+epoch+" complete");
        }
        log.info("Model build complete");
    }
}

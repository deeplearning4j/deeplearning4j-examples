package org.deeplearning4j.examples.arbiter;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * This is a basic hyperparameter optimization example using Arbiter to conduct random search on two network hyperparameters.
 * The two hyperparameters are learning rate and layer size, and the search is conducted for a simple multi-layer perceptron
 * on MNIST data.
 *
 * Note that this example is set up to use Arbiter's UI: http://localhost:9000/arbiter
 *
 * @author Alex Black
 */
public class BasicHyperparameterOptimizationExample {

    public static void main(String[] args) throws Exception {


        //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
        // fixed values or values to optimize, for each hyperparameter

        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(16,256);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
            //These next few options: fixed values for all models
            .weightInit(WeightInit.XAVIER)
            .regularization(true)
            .l2(0.0001)
            //Learning rate hyperparameter: search over different values, applied to all models
            .learningRate(learningRateHyperparam)
            .addLayer( new DenseLayerSpace.Builder()
                    //Fixed values for this layer:
                    .nIn(784)  //Fixed input: 28x28=784 pixels for MNIST
                    .activation(Activation.LEAKYRELU)
                    //One hyperparameter to infer: layer size
                    .nOut(layerSizeHyperparam)
                    .build())
            .addLayer( new OutputLayerSpace.Builder()
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build())
            .build();


        //Now: We need to define a few configuration options
        // (a) How are we going to generate candidates? (random search or grid search)
        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

        // (b) How are going to provide data? We'll use a simple data provider that returns MNIST data
        int nTrainEpochs = 2;
        int batchSize = 64;
        DataProvider dataProvider = new ExampleDataProvider(nTrainEpochs, batchSize);

        // (c) How we are going to save the models that are generated and tested?
        //     In this example, let's save them to disk the working directory
        //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if(f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);

        // (d) What are we actually trying to optimize?
        //     In this example, let's use classification accuracy on the test set
        //     See also ScoreFunctions.testSetF1(), ScoreFunctions.testSetRegression(regressionValue) etc
        ScoreFunction scoreFunction = new TestSetAccuracyScoreFunction();


        // (e) When should we stop searching? Specify this with termination conditions
        //     For this example, we are stopping the search at 15 minutes or 10 candidates - whichever comes first
        TerminationCondition[] terminationConditions = {
            new MaxTimeCondition(15, TimeUnit.MINUTES),
            new MaxCandidatesCondition(10)};



        //Given these configuration options, let's put them all together:
        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        //And set up execution locally on this machine:
        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());


        //Start the UI. Arbiter uses the same storage and persistence approach as DL4J's UI
        //Access at http://localhost:9000/arbiter
        StatsStorage ss = new FileStatsStorage(new File("arbiterExampleUiStats.dl4j"));
        runner.addListeners(new ArbiterStatusListener(ss));
        UIServer.getInstance().attach(ss);


        //Start the hyperparameter optimization
        runner.execute();


        //Print out some basic stats regarding the optimization procedure
        String s = "Best score: " + runner.bestScore() + "\n" +
            "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
            "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork)bestResult.getResult();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());


        //Wait a while before exiting
        Thread.sleep(60000);
        UIServer.getInstance().stop();
    }


    public static class ExampleDataProvider implements DataProvider {
        private int numEpochs;
        private int batchSize;

        public ExampleDataProvider(@JsonProperty("numEpochs") int numEpochs, @JsonProperty("batchSize") int batchSize){
            this.numEpochs = numEpochs;
            this.batchSize = batchSize;
        }

        private ExampleDataProvider(){

        }


        @Override
        public Object trainData(Map<String, Object> dataParameters) {
            try{
                return new MultipleEpochsIterator(numEpochs, new MnistDataSetIterator(batchSize,true,12345));
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        @Override
        public Object testData(Map<String, Object> dataParameters) {
            try{
                return new MnistDataSetIterator(batchSize,false,12345);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        @Override
        public Class<?> getDataType() {
            return DataSetIterator.class;
        }
    }
}

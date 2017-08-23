package org.deeplearning4j.examples.feedforward.classification.detectgender;

/**
 * Created by KIT Solutions (www.kitsol.com) on 9/28/2016.
 */

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;

/**
 * - Notes:
 *  - Data files are stored at following location
 *  .\dl4j-0.4-examples-master\dl4j-examples\src\main\resources\PredictGender\Data folder
 */

public class PredictGenderTrain
{
    public String filePath;

    public static void main(String args[])
    {

        PredictGenderTrain dg = new PredictGenderTrain();
        dg.filePath =  System.getProperty("user.dir") + "\\src\\main\\resources\\PredictGender\\Data";
        dg.train();
    }

    /**
     * This function uses GenderRecordReader and passes it to RecordReaderDataSetIterator for further training.
     */
    public void train()
    {
        int seed = 123456;
        double learningRate = 0.005;// was .01 but often got errors: "o.d.optimize.solvers.BaseOptimizer - Hit termination condition on iteration 0"
        int batchSize = 100;
        int nEpochs = 100;
        int numInputs = 0;
        int numOutputs = 0;
        int numHiddenNodes = 0;

        try(GenderRecordReader rr = new GenderRecordReader(new ArrayList<String>() {{add("M");add("F");}}))
        {
            long st = System.currentTimeMillis();
            System.out.println("Preprocessing start time : " + st);

            rr.initialize(new FileSplit(new File(this.filePath)));

            long et = System.currentTimeMillis();
            System.out.println("Preprocessing end time : " + et);
            System.out.println("time taken to process data : " + (et-st) + " ms");

            numInputs = rr.maxLengthName * 5;  // multiplied by 5 as for each letter we use five binary digits like 00000
            numOutputs = 2;
            numHiddenNodes = 2 * numInputs + numOutputs;


            GenderRecordReader rr1 = new GenderRecordReader(new ArrayList<String>() {{add("M");add("F");}});

            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2);
            DataSetIterator testIter = new RecordReaderDataSetIterator(rr1, batchSize, numInputs, 2);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .biasInit(1)
                .regularization(true).l2(1e-4)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.setListeners(new StatsListener(statsStorage));

            for ( int n = 0; n < nEpochs; n++)
            {
                while(trainIter.hasNext())
                {
                    model.fit(trainIter.next());
                }
                trainIter.reset();
            }

            ModelSerializer.writeModel(model,this.filePath + "PredictGender.net",true);

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(numOutputs);
            while(testIter.hasNext()){
                DataSet t = testIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray predicted = model.output(features,false);

                eval.eval(lables, predicted);

            }

            //Print the evaluation statistics
            System.out.println(eval.stats());
        }
        catch(Exception e)
        {
            System.out.println("Exception111 : " + e.getMessage());
        }
    }
}

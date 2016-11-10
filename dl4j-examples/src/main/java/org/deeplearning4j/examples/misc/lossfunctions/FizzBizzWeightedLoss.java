package org.deeplearning4j.examples.misc.lossfunctions;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by susaneraly on 10/12/16.
 */
public class WeightedLoss {
    private static Logger log = LoggerFactory.getLogger(WeightedLoss.class);

    public static void main(String[] args) throws Exception {

        int numInputs = 10;
        int numOutputs = 4;

        int numHiddenNodes = 200;

        double learningRate = 0.01;
        int nEpochs = 1000;
        int iterations = 1;
        int seed = 123;
        int batchSize = 64;

        /*=========================================================
            Load csv into a dataset/datasetiterator
        ==========================================================*/
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("fizzbuzztrain.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,10,4);
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("fizzbuzztest.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,10,4);

        /*=========================================================
           Setup NN
           Initialize model and set up listeners - score and histogram
        ==========================================================*/

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.RMSPROP)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .activation("relu")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation("softmax")
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        model.setListeners(new HistogramIterationListener(1));


        /*=========================================================
          Train your NN
        ==========================================================*/

        for ( int n = 0; n < nEpochs; n++) {
            while (trainIter.hasNext()) {
                DataSet tr = trainIter.next();
                model.fit(tr);
            }
            trainIter.reset();
        }

        /*=========================================================
         Evaluate your NN
        ==========================================================*/
        System.out.println("Evaluate model....");
        model.evaluate(testIter);

    }


}

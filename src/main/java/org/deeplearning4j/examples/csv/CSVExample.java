package org.deeplearning4j.examples.csv;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class CSVExample {

    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    public static void main(String[] args) throws  Exception {
        RecordReader recordReader = new CSVRecordReader(0,",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        //reader,label index,number of possible labels
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,4,3);
        //get the dataset using the record reader. The datasetiterator handles vectorization
        DataSet next = iterator.next();
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 3;
        long seed = 6;
        int listenerFreq = iterations;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true).useDropConnect(true)
                .learningRate(1e-1)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("relu").dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        next.normalizeZeroMeanZeroUnitVariance();
        next.shuffle();
        //split test and train
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.6);
        model.fit(testAndTrain.getTrain());

        //evaluate the model
        Evaluation eval = new Evaluation(3);
        DataSet test = testAndTrain.getTest();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());

    }

}

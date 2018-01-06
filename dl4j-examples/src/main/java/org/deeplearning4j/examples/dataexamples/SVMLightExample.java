package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SVMLightExample {
    private static Logger log = LoggerFactory.getLogger(SVMLightExample.class);

    public static void main(String[] args) throws Exception {

        int numOfFeatures = 784;     // For MNIST data set, each row is a 1D expansion of a handwritten digits picture of size 28x28 pixels = 784
        int numOfClasses = 10;       // 10 classes (types of senders) in the data set. Zero indexing. Classes have integer values 0, 1 or 2 ... 9
        int batchSize = 10;          // 1000 examples, with batchSize is 10, around 100 iterations per epoch
        int printIterationsNum = 20; // print score every 20 iterations

        int hiddenLayer1Num = 200;
        int iterations = 1;
        long seed = 42;
        int nEpochs = 4;

        Configuration config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, true);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, numOfFeatures);

        SVMLightRecordReader trainRecordReader = new SVMLightRecordReader();
        trainRecordReader.initialize(config, new FileSplit(new ClassPathResource("/DataExamples/MnistSVMLightExample/mnist_svmlight_train_1000.txt").getFile()));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, numOfFeatures, numOfClasses);

        SVMLightRecordReader testRecordReader = new SVMLightRecordReader();
        testRecordReader.initialize(config, new FileSplit(new ClassPathResource("/DataExamples/MnistSVMLightExample/mnist_svmlight_test_100.txt").getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, numOfFeatures, numOfClasses);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
            .iterations(iterations)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.02)
            .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
                .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numOfFeatures).nOut(hiddenLayer1Num)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(hiddenLayer1Num).nOut(numOfClasses).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(printIterationsNum));

        for ( int n = 0; n < nEpochs; n++) {

            model.fit(trainIter);

            log.info(String.format("Epoch %d finished training", n + 1));

            // evaluate the model on test data, once every second epoch
            if ((n + 1) % 2 == 0) {
                //evaluate the model on the test set
                Evaluation eval = new Evaluation(numOfClasses);
                testIter.reset();
                while(testIter.hasNext()) {
                    DataSet t = testIter.next();
                    INDArray features = t.getFeatures();
                    INDArray labels = t.getLabels();
                    INDArray predicted = model.output(features, false);
                    eval.eval(labels, predicted);
                }
                log.info(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                    n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
                log.info(eval.stats());
            }
        }
        System.out.println("Finished...");
    }
}


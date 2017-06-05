package org.deeplearning4j.examples.convolution;

    import org.apache.commons.io.FileUtils;
    import org.apache.commons.io.IOUtils;
    import org.datavec.api.berkeley.Pair;
    import org.datavec.api.records.reader.SequenceRecordReader;
    import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
    import org.datavec.api.split.NumberedFileInputSplit;
    import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
    import org.deeplearning4j.eval.Evaluation;
    import org.deeplearning4j.nn.api.OptimizationAlgorithm;
    import org.deeplearning4j.nn.conf.*;
    import org.deeplearning4j.nn.conf.layers.GravesLSTM;
    import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
    import org.deeplearning4j.nn.weights.WeightInit;
    import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
    import org.nd4j.linalg.activations.Activation;
    import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
    import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
    import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
    import org.nd4j.linalg.lossfunctions.LossFunctions;
    import org.slf4j.Logger;
    import org.slf4j.LoggerFactory;

    import java.io.File;
    import java.net.URL;
    import java.util.ArrayList;
    import java.util.Collections;
    import java.util.List;
    import java.util.Random;

public class NetLeak {
    private static final Logger log = LoggerFactory.getLogger(NetLeak.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    public void RunAndDropNet(DataSetIterator trainData,DataSetIterator testData) throws Exception {

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)    //Random number generator seed for improved repeatability. Optional.
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(0.005)
            .trainingWorkspaceMode(WorkspaceMode.SINGLE)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nIn(10).nOut(6).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 10;
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
            net.evaluate(testData);
            testData.reset();
            trainData.reset();
        }
    }

    public static void main(String[] args) throws Exception {
        downloadUCIData();

        // ----- Load the training data -----
        //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        int miniBatchSize = 200;
        int numLabelClasses = 6;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        int i = 1;
        while (true) {
            log.info("run "+String.valueOf(i++));

            new NetLeak().RunAndDropNet(trainData,testData);
            trainData.reset();
            testData.reset();
        }
    }


    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    private static void downloadUCIData() throws Exception {
        if (baseDir.exists()) return;    //Data already exists, don't download it again

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 450;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}

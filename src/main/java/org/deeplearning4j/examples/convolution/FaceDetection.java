package org.deeplearning4j.examples.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * MSRA-CFW Dataset of Celebrity Faces on the Web is a data set created by MicrosoftResearch.
 * This is based of of the thumbnails data set which is a smaller subset. It includes 2215 images
 * and 10 classifications with each image only including one face.
 *
 * More information and the data set can be found at: http://research.microsoft.com/en-us/projects/msra-cfw/
 *
 */
public class FaceDetection {
    private static final Logger log = LoggerFactory.getLogger(FaceDetection.class);

    // based on small sample
    public final static int NUM_IMAGES = 2215; // # examples per person range 50 to 700
    public final static int NUM_ABELS = 10;
    public final static int WIDTH = 80; // size varies
    public final static int HEIGHT = 80;
    public final static int CHANNELS = 3;

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numExamples",usage="Number of examples",aliases="-nE")
    protected int numExamples = 100;
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 20;
    @Option(name="--epochs",usage="Number of epochs",aliases="-ep")
    protected int epochs = 2;
    @Option(name="--iter",usage="Number of iterations",aliases="-i")
    protected int iterations = 2;
    @Option(name="--numLabels",usage="Number of categories",aliases="-nL")
    protected int numLabels = 4;

    @Option(name="--weightInit",usage="How to initialize weights",aliases="-wI")
    protected WeightInit weightInit = WeightInit.XAVIER;
    @Option(name="--activation",usage="Activation function to use",aliases="-a")
    protected String activation = "relu";
    @Option(name="--updater",usage="Updater to apply gradient changes",aliases="-up")
    protected Updater updater = Updater.NESTEROVS;
    @Option(name="--learningRate", usage="Learning rate", aliases="-lr")
    protected double lr = 1e-2;
    @Option(name="--momentum",usage="Momentum rate",aliases="-mu")
    protected double mu = 0.9;
    @Option(name="--lambda",usage="L2 weight decay",aliases="-l2")
    protected double l2 = 1e-3;
    @Option(name="--regularization",usage="Boolean to apply regularization",aliases="-reg")
    protected boolean regularization = true;
    @Option(name="--split",usage="Percent to split for training",aliases="-split")
    protected double split = 0.8;

    public void run(String[] args) {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        int seed = 123;
        int listenerFreq = 1;
        boolean appendLabels = true;
        int splitTrainNum = (int) (batchSize*split);

        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        DataSet dsNext;


        // TODO setup to download and untar the example - currently needs manual download
        log.info("Load data....");
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample"); // 10 labels
        File mainPath = new File(BaseImageLoader.BASE_DIR, "ms_sample"); // 4 labels
        RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, appendLabels);
        try {
            recordReader.initialize(new LimitFileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, numExamples, numLabels, null, new Random(123)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, numLabels);

//        List<String> labels = dataIter.getLabels();
        List<String> labels = Arrays.asList(new String[]{"liv_tyler", "michelle_obama", "aaron_carter", "al_gore"});

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(lr)
                .momentum(mu)
                .regularization(regularization)
                .l2(l2)
                .updater(updater)
                .useDropConnect(true)
                //////////// ? forgot where I got this ////////////
//                .list(11)
//                .layer(0, new ConvolutionLayer.Builder(7, 7)
//                        .name("cnn1")
//                        .nIn(CHANNELS)
//                        .stride(1, 1)
//                        .nOut(48)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool1")
//                        .build())
//                .layer(2, new LocalResponseNormalization.Builder().build())
//                .layer(3, new ConvolutionLayer.Builder(5, 5)
//                        .name("cnn2")
//                        .stride(1, 1)
//                        .nOut(128)
//                        .build())
//                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool2")
//                        .build())
//                .layer(5, new LocalResponseNormalization.Builder().build())
//                .layer(6, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn3")
//                        .stride(1, 1)
//                        .nOut(192)
//                        .build())
//                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool3")
//                        .build())
//                .layer(8, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(256)
//                        .dropOut(0.5)
//                        .build())
//                .layer(9, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(256)
//                        .dropOut(0.5)
//                        .build())
//                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(NUM_LABELS)
//                        .activation("softmax")
//                        .build())
//
                    //////////// MNIST version ////////////
//                .list(3)
//                .layer(0, new ConvolutionLayer.Builder(10, 10)
//                        .stride(2,2)
//                        .nIn(CHANNELS)
//                        .nOut(6)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
//                        .build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(NUM_IMAGES)
//                        .activation("softmax")
//                        .build())

                ///////////// Paper on tested approaches ////////////
                .list(10)
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(128)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(128)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(1, 1)
                        .nOut(64)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .stride(1, 1)
                        .build())
                .layer(6, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(400)
                        .dropOut(0.5)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(400)
                        .dropOut(0.5)
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn3")
                        .nOut(200)
                        .dropOut(0.5)
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(HEIGHT, WIDTH, CHANNELS);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        // Listeners
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(listenerFreq).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));

        // Early Stopping

//        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
//        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
//                .evaluateEveryNEpochs(1)
//                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
//                .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))     //Calculate test set score
//                .modelSaver(saver)
//                .build();
//
//        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,configuration,mnistTrain1024);
//
//        //Conduct early stopping training:
//        EarlyStoppingResult result = trainer.fit();
//        System.out.println("Termination reason: " + result.getTerminationReason());
//        System.out.println("Termination details: " + result.getTerminationDetails());
//        System.out.println("Total epochs: " + result.getTotalEpochs());
//        System.out.println("Best epoch number: " + result.getBestModelEpoch());
//        System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//        //Print score vs. epoch
//        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
//        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
//        Collections.sort(list);
//        System.out.println("Score vs. Epoch:");
//        for( Integer i : list){
//            System.out.println(i + "\t" + scoreVsEpoch.get(i));
//        }


        log.info("Train model....");
        // one epoch
        while (dataIter.hasNext()) {
            dsNext = dataIter.next();
            dsNext.scale();
            trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        // more than 1 epoch for just training
        for(int i = 1; i < epochs; i++) {
            dataIter.reset();
            while (dataIter.hasNext()) {
                dsNext = dataIter.next();
                trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed));
                trainInput = trainTest.getTrain();
                model.fit(trainInput);
            }
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(labels);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

    public static void main(String[] args) throws Exception {
        new FaceDetection().run(args);
    }

}

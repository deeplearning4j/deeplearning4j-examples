package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.File;

/**
 * This example trains a convolutional neural network image classifier on the Tiny ImageNet dataset using Apache Spark
 *
 * The Tiny ImageNet dataset is an image dataset of size 64x64 images, with 200 classes, and 500 images per class,
 * for a total of 100,000 images.
 *
 * Before running this example, you should do ONE (either) of the following to prepare the data for training:
 * 1. Run PreprocessLocal, and copy the output files to remote storage for your cluster (HDFS, S3, Azure storage, etc), OR
 * 2. Run PreprocessSpark on the tiny imagenet source files
 *
 * The CNN classifier trained here is trained from scratch without any pretraining. It is a custom network architecture
 * with 1,077,160 parameters based loosely on the VGG/DarkNet architectures. Improved accuracy is likely possible with
 * a larger network, better selection of hyperparameters, and more epochs.
 *
 * For further details on DL4J's Spark implementation, see the "Distributed Deep Learning" pages at:
 * https://deeplearning4j.org/docs/latest/
 *
 * A local (single machine) version of this example is available in TrainLocal
 *
 *
 * @author Alex Black
 */
public class TrainSparkDebug {
    public static final Logger log = LoggerFactory.getLogger(TrainSparkDebug.class);

    /* --- Required Arguments -- */

    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script." +
        " See PreprocessLocal or PreprocessSpark", required = true)
    private String dataPath;

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String masterIP;

    @Parameter(names = {"--networkMask"}, description = "Network mask for Spark communication. For example, 10.0.0.0/16", required = true)
    private String networkMask;

    @Parameter(names = {"--numNodes"}, description = "Number of Spark nodes (machines)", required = true)
    private int numNodes;

    /* --- Optional Arguments -- */

    @Parameter(names = {"--saveDirectory"}, description = "If set: save the trained network plus evaluation to this directory." +
        " Otherwise, the trained net will not be saved")
    private String saveDirectory = null;

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JTinyImageNetExample";

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 10;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
        "minibatches per worker when fitting")
    private int minibatch = 32;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
    private int numWorkersPerNode = 1;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold. See ")
    private double gradientThreshold = 1E-3;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private int port = 40123;

    public static void main(String[] args) throws Exception {
        new TrainSparkDebug().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        SparkConf conf = new SparkConf();
        conf.setAppName(sparkAppName);
        JavaSparkContext sc = new JavaSparkContext(conf);



        //Set up TrainingMaster for gradient sharing training
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
            .unicastPort(port)                          // Should be open for IN/OUT communications on all Spark nodes
            .networkMask(networkMask)                   // Local network mask - for example, 10.0.0.0/16 - see https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-parameter-server
            .controllerAddress(masterIP)                // IP address of the master/driver node
            .meshBuildMode(MeshBuildMode.PLAIN)
            .build();
        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
            .rngSeed(12345)
            .collectTrainingStats(false)
            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
            .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(this.gradientThreshold))     //Threshold algorithm determines the encoding threshold to be use. See docs for details
            .workersPerNode(numWorkersPerNode)          // Workers per node
            .build();


        ComputationGraph net = getNetwork();
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);
        sparkNet.setListeners(new PerformanceListener(10, true));

        //Create data loader
        int imageHeightWidth = 64;      //64x64 pixel input
        int imageChannels = 3;          //RGB
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(new TinyImageNetDataSetIterator(1).getLabels());
        int numClasses = TinyImageNetFetcher.NUM_LABELS;
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);
        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range


        //Fit the network
        String trainPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "train";
        JavaRDD<String> pathsTrain = SparkUtils.listPaths(sc, trainPath);
        StringBuilder sbEval = new StringBuilder();
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
            sparkNet.fitPaths(pathsTrain, loader);

            //Perform evaluation
            String testPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "test";
            JavaRDD<String> pathsTest = SparkUtils.listPaths(sc, testPath);
            Evaluation evaluation = new Evaluation(TinyImageNetDataSetIterator.getLabels(false), 5); //Set up for top 5 accuracy
            evaluation = (Evaluation) sparkNet.doEvaluation(pathsTest, loader, evaluation)[0];
            String evalStats = evaluation.stats();
            log.info("Evaluation statistics: {}", evalStats);

            if(saveDirectory != null){
                // Save evaluation
                String evalPath = FilenameUtils.concat(saveDirectory, "evaluation.txt");
                sbEval.append("----- Epoch ").append(i+1).append(" of ").append(numEpochs).append(" -----\n")
                    .append(evalStats)
                    .append("\n\n");
                SparkUtils.writeStringToFile(evalPath, sbEval.toString(), sc);

                File f = new File(saveDirectory, "net_epoch" + i + ".zip");
                sparkNet.getNetwork().save(f);
            }
        }



        if (saveDirectory != null && saveDirectory.isEmpty()) {
            log.info("Saving the network and evaluation to directory: {}", saveDirectory);

            // Save network
            String networkPath = FilenameUtils.concat(saveDirectory, "network.bin");
            FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
            try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(networkPath)))) {
                ModelSerializer.writeModel(sparkNet.getNetwork(), os, true);
            }
        }


        log.info("----- Example Complete -----");
    }

    public static ComputationGraph getNetwork() {
        //This network: created for the purposes of this example. It is a simple CNN loosely inspired by the DarkNet
        // architecture, which was in turn inspired by the VGG16/19 networks
        //The performance of this network can likely be improved

        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
            .add(0, 3e-4)
            .add(1, 2e-4)
            .add(3, 1e-4)
            .add(5, 5e-5)
            .add(7, 2e-5).build();

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
            .convolutionMode(ConvolutionMode.Same)
            .l2(1e-4)
            .updater(new AMSGrad(lrSchedule))
            .weightInit(WeightInit.RELU)
            .graphBuilder()
            .addInputs("input")
            .setOutputs("output");

        addLayers(b, 0, 3, 3, 32, 0);     //64x64 out
        addLayers(b, 1, 3, 32, 64, 2);    //32x32 out
        addLayers(b, 2, 2, 64, 128, 0);   //32x32 out
        addLayers(b, 3, 2, 128, 256, 2);   //16x16 out
        addLayers(b, 4, 2, 256, 256, 0);   //16x16 out
        addLayers(b, 5, 2, 256, 512, 2);   //8x8 out

        b.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
            .nIn(512)
            .nOut(TinyImageNetFetcher.NUM_LABELS)
            .weightInit(WeightInit.XAVIER)
            .stride(1, 1)
            .activation(Activation.IDENTITY)
            .build(), "maxpooling2d_5")
            .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
            .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).build(), "globalpooling")
            .setOutputs("loss");

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        return net;
    }


    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, int filterSize, int nIn, int nOut, int poolSize) {
        return addLayers(graphBuilder, layerNumber, filterSize, nIn, nOut, poolSize, poolSize);
    }

    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, int filterSize, int nIn, int nOut, int poolSize, int poolStride) {
        String input = "maxpooling2d_" + (layerNumber - 1);
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "activation_" + (layerNumber - 1);
        }
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "concatenate_" + (layerNumber - 1);
        }
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "input";
        }

        return addLayers(graphBuilder, layerNumber, input, filterSize, nIn, nOut, poolSize, poolStride);
    }

    public static ComputationGraphConfiguration.GraphBuilder addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, int layerNumber, String input, int filterSize, int nIn, int nOut, int poolSize, int poolStride) {
        graphBuilder
            .addLayer("convolution2d_" + layerNumber,
                new ConvolutionLayer.Builder(filterSize,filterSize)
                    .nIn(nIn)
                    .nOut(nOut)
                    .weightInit(WeightInit.XAVIER)
                    .convolutionMode(ConvolutionMode.Same)
                    .hasBias(false)
                    .stride(1,1)
                    .activation(Activation.IDENTITY)
                    .build(),
                input)
//            .addLayer("batchnormalization_" + layerNumber,
//                new BatchNormalization.Builder()
//                    .nIn(nOut).nOut(nOut)
//                    .weightInit(WeightInit.XAVIER)
//                    .activation(Activation.IDENTITY)
//                    .build(),
//                "convolution2d_" + layerNumber)
            .addLayer("activation_" + layerNumber,
                new ActivationLayer.Builder()
                    .activation(new ActivationLReLU(0.1))
                    .build(),
//                "batchnormalization_" + layerNumber);
                "convolution2d_" + layerNumber);
        if (poolSize > 0) {
            graphBuilder
                .addLayer("maxpooling2d_" + layerNumber,
                    new SubsamplingLayer.Builder()
                        .kernelSize(poolSize, poolSize)
                        .stride(poolStride, poolStride)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(),
                    "activation_" + layerNumber);
        }

        return graphBuilder;
    }
}

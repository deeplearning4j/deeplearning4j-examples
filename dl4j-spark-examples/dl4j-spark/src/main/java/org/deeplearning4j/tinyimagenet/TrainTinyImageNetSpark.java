package org.deeplearning4j.tinyimagenet;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.api.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;

public class TrainTinyImageNetSpark {
    public static final Logger log = LoggerFactory.getLogger(TrainTinyImageNetSpark.class);

    /* --- Required Arguments -- */

    @Parameter(names = {"--outputPath"}, description = "Local output path/directory to write results to", required = true)
    private String outputPath = null;

    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script", required = true)
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
    private int numEpochs = 1;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
        "minibatches per worker when fitting")
    private int minibatch = 32;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node")
    private int numWorkersPerNode = 1;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold")
    private double gradientThreshold = 1E-4;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private int port = 40123;

    public static void main(String[] args) throws Exception {
        new TrainTinyImageNetSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        JavaSparkContext sc = new JavaSparkContext();

        //Set up TrainingMaster for gradient sharing training
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
            .unicastPort(port)                          // Should be open for IN/OUT communications on all Spark nodes
            .networkMask(networkMask)                   // Local network mask - for example, 10.0.0.0/16 - see https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-parameter-server
            .controllerAddress(masterIP)                // IP address of the master/driver node
            .build();
        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, numWorkersPerNode, this.gradientThreshold, minibatch)
            .rngSeed(12345)
            .collectTrainingStats(false)
            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
            .updatesThreshold(this.gradientThreshold)   // Encoding threshold (see docs for details)
            .workersPerNode(numWorkersPerNode)          // Workers per node
            .build();


        ComputationGraph net = getNetwork();
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);

        //Create data loader
        int imageHeightWidth = 64;      //64x64 pixel input
        int imageChannels = 3;          //RGB
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(new TinyImageNetDataSetIterator(1).getLabels());
        int numClasses = TinyImageNetFetcher.NUM_LABELS;
        DataSetLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);

        //Fit the network
        JavaRDD<String> pathsTrain = SparkUtils.listPaths(sc, dataPath + "/train");
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
            sparkNet.fitPaths(pathsTrain, loader);
        }

        //Perform evaluation
        Evaluation evaluation = sparkNet.evaluate(dataPath + "/test", loader);
        log.info("Evaluation statistics: {}", evaluation.stats());

        if (saveDirectory != null && saveDirectory.isEmpty()) {
            log.info("Saving the network and evaluation to directory: {}", saveDirectory);

            // Save network
            String networkPath = FilenameUtils.concat(saveDirectory, "network.bin");
            FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
            try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(networkPath)))) {
                ModelSerializer.writeModel(sparkNet.getNetwork(), os, true);
            }

            // Save evaluation
            String evalPath = FilenameUtils.concat(saveDirectory, "evaluation.txt");
            SparkUtils.writeStringToFile(evalPath, evaluation.stats(), sc);
        }


        log.info("----- Example Complete -----");
    }

    public static ComputationGraph getNetwork() {

        //This network: created for the purposes of this example. It is a simple CNN loosely inspired by the DarkNet
        // architecture, which was in turn inspired by the VGG16/19 networks

        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
            .convolutionMode(ConvolutionMode.Same)
            .l2(1e-4)
            .updater(new AMSGrad(1e-3))
            .weightInit(WeightInit.RELU)
            .graphBuilder()
            .addInputs("input")
            .setOutputs("output");

        DarknetHelper.addLayers(b, 0, 3, 3, 32, 0);     //64x64 out
        DarknetHelper.addLayers(b, 1, 3, 32, 64, 2);    //32x32 out
        DarknetHelper.addLayers(b, 2, 2, 64, 128, 0);   //32x32 out
        DarknetHelper.addLayers(b, 3, 2, 128, 256, 2);   //16x16 out
        DarknetHelper.addLayers(b, 4, 2, 256, 256, 0);   //16x16 out
        DarknetHelper.addLayers(b, 5, 2, 256, 512, 2);   //8x8 out

        b.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
            .nIn(1024)
            .nOut(TinyImageNetFetcher.NUM_LABELS)
            .weightInit(WeightInit.XAVIER)
            .stride(1, 1)
            .activation(Activation.IDENTITY)
            .build())
            .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
            .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).build(), "globalpooling")
            .setOutputs("loss");

        ComputationGraphConfiguration conf = b.build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        return net;
    }
}

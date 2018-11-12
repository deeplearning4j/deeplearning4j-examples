package org.deeplearning4j.resnet50Benchmark;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.tinyimagenet.TrainSpark;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.util.*;

public class BenchmarkSpark {
    public static final Logger log = LoggerFactory.getLogger(TrainSpark.class);

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

    @Parameter(names = {"--numBatches"}, description = "Number of batches to use for benchmarking. Should be at least 20x number of nodes, preferably more")
    private int numBatches = 2000;

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JImagenetBenchmark";

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of minibatches per worker when fitting")
    private int minibatch = 128;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
    private int numWorkersPerNode = 1;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private int port = 40123;

    public static void main(String[] args) throws Exception {
        new BenchmarkSpark().entryPoint(args);
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
//            .maxChunkSize(1024*1024)
//            .chunksBufferSize(3L*1024*1024*1024)
            .build();
        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
            .workersPerNode(numWorkersPerNode)          // Workers per node
//            .workerTogglePeriodicGC(false)
            .workerPeriodicGCFrequency(10000)
            .build();

        ComputationGraph net = ResNet50.builder()
            .seed(12345)
            .numClasses(1000)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
            .cacheMode(CacheMode.NONE)
            .workspaceMode(WorkspaceMode.ENABLED)
            .build().init();

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);
        sparkNet.setListeners(new PerformanceListener(1));

        //Create data loader
        int imageHeightWidth = 224;     //224x224 pixel input to network
        int imageChannels = 3;          //RGB
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(getLabels());
        int numClasses = 1000;
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);
        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range


        List<String> allPaths = listPaths(dataPath, sc.hadoopConfiguration());

        //First: perform a small warmup
        List<String> warmupPaths = randomSample(allPaths, 10*numNodes, 12345);
        JavaRDD<String> warmupPathsRdd = sc.parallelize(warmupPaths);
        sparkNet.fitPaths(warmupPathsRdd, loader);

        //Perform benchmark on a subset of the data, specified number of minibatches
        List<String> trainPaths = randomSample(allPaths, numBatches, 12345);
        if(trainPaths.size() != numBatches){
            throw new IllegalStateException("Expected " + numBatches + " batches, got " + trainPaths.size());
        }
        JavaRDD<String> trainPathsRdd = sc.parallelize(trainPaths);
        long timeBefore = System.currentTimeMillis();
        sparkNet.fitPaths(trainPathsRdd, loader);
        long timeAfter = System.currentTimeMillis();

        long totalTimeMillisec = (timeAfter - timeBefore);
        int totalBatches = numBatches;

        double batchesPerSec = totalBatches / (totalTimeMillisec / 1000.0);
        double examplesPerSec = (totalBatches * minibatch) / (totalTimeMillisec / 1000.0);

        DecimalFormat df = new DecimalFormat("#.00");
        System.out.println("Completed " + totalBatches + " in " + df.format(totalTimeMillisec / 1000.0) + " seconds, batch size " + minibatch);
        System.out.println("Batches per second: " + df.format(batchesPerSec));
        System.out.println("Examples per second: " + df.format(examplesPerSec));
    }


    public static List<String> listPaths(String path, Configuration config) throws IOException {
        FileSystem hdfs = FileSystem.get(URI.create(path), config);
        RemoteIterator fileIter = hdfs.listFiles(new Path(path), true);
        List<String> outPaths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String filePath = ((LocatedFileStatus) fileIter.next()).getPath().toString();
            outPaths.add(filePath);
        }
        return outPaths;
    }

    public static <T> List<T> randomSample(List<T> in, int count, long rngSeed){
        if(in.isEmpty()){
            throw new IllegalStateException("Cannot sample from empty list");
        }
        if(count > in.size()){
            //Duplicate when count > amount of data
            log.warn("Sampling {} values from size {} list - values will be repeated", count, in.size());
        }

        int[] order = new int[in.size()];
        for( int i=0; i<order.length; i++ ){
            order[i] = i;
        }
        MathUtils.shuffleArray(order, rngSeed);

        List<T> out = new ArrayList<>();
        for(int i=0; i<count; i++ ){
            out.add(in.get(order[i]));
        }
        return out;
    }


    private List<String> getLabels() throws IOException {
        File validationLabelsFile = new File(System.getProperty("java.io.tmpdir"), PreprocessLocal.VALIDATION_LABEL_MAPPING_FILENAME);
        if(!validationLabelsFile.exists()){
            FileUtils.copyURLToFile(new URL(PreprocessLocal.VALIDATION_LABEL_MAPPING_FILE), validationLabelsFile);
        }

        //Get unique set of labels, then sort alphabetically:
        List<String> labels = new ArrayList<>(new HashSet<>(FileUtils.readLines(validationLabelsFile, StandardCharsets.UTF_8)));
        Collections.sort(labels);

        if (labels.size() != 1000) {
            throw new IllegalStateException("Expected exactly 1000 labels, got " + labels.size());
        }
        return labels;
    }

}

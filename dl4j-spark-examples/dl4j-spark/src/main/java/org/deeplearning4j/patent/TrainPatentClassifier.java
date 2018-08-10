package org.deeplearning4j.patent;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.patent.preprocessing.PatentLabelGenerator;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.patent.utils.data.LoadDataSetsFunction;
import org.deeplearning4j.patent.utils.evaluation.ConvergenceRunnable;
import org.deeplearning4j.patent.utils.evaluation.ToEval;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Train the patent classifier on Spark
 *
 * @author Alex Black
 */
public class TrainPatentClassifier {
    public static final int MILLISEC_PER_SEC = 1000;
    private static final Logger log = LoggerFactory.getLogger(TrainPatentClassifier.class);

    /* --- Required Arguments -- */

    @Parameter(names = {"--outputPath"}, description = "Local output path/directory to write results to", required = true)
    private String outputPath = null;

    @Parameter(names = {"--azureStorageAcct"}, description = "Name of the Azure storage account to use for storage", required = true)
    private String azureStorageAcct;

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
    private String masterIP;

    @Parameter(names = {"--networkMask"}, description = "Network mask for Spark communication. For example, 10.0.0.0/16", required = true)
    private String networkMask;

    @Parameter(names = {"--numNodes"}, description = "Number of Spark nodes (machines)", required = true)
    private int numNodes;

    /* --- Optional Arguments -- */

    @Parameter(names = {"--azureContainerPreproc"}, description = "Name of the container in the specified storage account for the serialized training DataSet files")
    private String azureContainerPreproc = "patentPreprocData";

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JSparkPatentClassifierExample";

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 1;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
        "minibatches per worker when fitting")
    private int minibatch = 32;

    @Parameter(names = {"--maxSequenceLength"}, description = "Maximum number of words in the sequences for generated DataSets")
    private int maxSequenceLength = 1000;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node")
    private int numWorkersPerNode = 1;

    @Parameter(names = {"--listenerFrequency"}, description = "Listener Frequency")
    private int listenerFrequency = 10;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold")
    private double gradientThreshold = 1E-4;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private int port = 40123;

    @Parameter(names = {"--totalExamplesTest"}, description = "Total number of examples for testing. Set to -1 to use all; otherwise a" +
        " (consist between runs) random subset is used. Note that the full set can take a long time to evaluate!")
    private int totalExamplesTest = 10000;

    @Parameter(names = {"--wordVectorsPath"}, description = "Word vectors path")
    private String wordVectorsPath = "wasbs://resources@deeplearning4jblob.blob.core.windows.net/wordvectors/GoogleNews-vectors-negative300.bin.gz";

    @Parameter(names = {"--saveFrequencySec"}, description = "How often (in seconds) to save a copy of the parameters for later evaluation")
    private int saveFreqSec = 180;

    @Parameter(names = {"--evalOnly"}, description = "If set, only evaluation will be performed on all parameter snapshots found;" +
        "no training will occur when this is set", arity = 1)
    private boolean evalOnly = false;

    @Parameter(names = {"--continueTraining"}, description = "If true, training will continue from the last saved checkpoint", arity = 1)
    private boolean continueTraining = false;

    @Parameter(names = {"--maxRuntimeSec"}, description = "Maximum runtime in seconds (training will terminate after completing a subset " +
            "if this is exceeded). Set -1 for no maximum - in which case the full numEpochs epochs will be trained")
    private long maxRuntimeSec = -1;

    @Parameter(names = {"--batchesBtwCheckpoints"}, description = "Number of minibatches between saving model checkpoints." +
            " Note that setting this value too low can result in poor performance. Suggested minimum: 200 * numNodes * numWorkersPerNode." +
            " Set to <= 0 for fitting on all data")
    private int batchesBtwCheckpoints = 5000;

    public static void main(String[] args) throws Exception {
        new TrainPatentClassifier().entryPoint(args);
    }

    /**
     * JCommander entry point
     */
    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        //Azure storage account naming rules: https://blogs.msdn.microsoft.com/jmstall/2014/06/12/azure-storage-naming-rules/
        //The default exceptions aren't helpful, we'll validate this here
        if(!azureStorageAcct.matches("^[a-z0-9]+$") || azureStorageAcct.length() < 3 || azureStorageAcct.length() > 24){
            throw new IllegalStateException("Invalid storage account name: must be alphanumeric, lowercase, " +
                    "3 to 24 characters. Got option azureStorageAcct=\"" + azureStorageAcct + "\"");
        }
        if(!azureContainerPreproc.matches("^[a-z0-9-]+$") || azureContainerPreproc.length() < 3 || azureContainerPreproc.length() > 63){
            throw new IllegalStateException("Invalid Azure container name: must be alphanumeric or dash, lowercase, " +
                    "3 to 63 characters. Got option azureContainerPreproc=\"" + azureContainerPreproc + "\"");
        }


        StringBuilder results = new StringBuilder();    //To store results/timing - will be written to disk on completion

        long startTime = System.currentTimeMillis();

        // Prepare neural net
        ComputationGraph net = new ComputationGraph(NetworkConfiguration.getConf());
        net.init();
        log.info("Parameters: {}", net.params().length());

        // Configure Spark
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName(sparkAppName);
        JavaSparkContext sc = new JavaSparkContext();
        int numWorkers = this.numNodes * this.numWorkersPerNode;

        //Prepare dataset RDDs
        String dirName = "seqLength" + maxSequenceLength + "_mb" + minibatch;
        String containerRoot = "wasbs://" + azureContainerPreproc + "@" + azureStorageAcct + ".blob.core.windows.net/";
        String baseOutPath = containerRoot + dirName;
        String trainDataPathRootDir = baseOutPath + "/train/";
        String testDataPathRootDir = baseOutPath + "/test/";
        JavaRDD<String> trainDataPaths = SparkUtils.listPaths(sc, trainDataPathRootDir);
        JavaRDD<String> testDataPaths = totalExamplesTest <= 0 ? null : listPathsSubset(sc, testDataPathRootDir, totalExamplesTest, 12345);
        trainDataPaths.cache();
        if(testDataPaths != null)
            testDataPaths.cache();


        //If only doing evaluation: perform it here and exit
        if(evalOnly){
            evaluateOnly(sc, net, testDataPaths);
            return;
        }

        //Write configuration to output directory. Also determine output base directory for results
        writeConfig(sc);

        //Set up TrainingMaster for gradient sharing training
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .unicastPort(port)                          // Should be open for IN/OUT communications on all Spark nodes
                .networkMask(networkMask)                   // Local network mask
                .controllerAddress(masterIP)
                .build();
        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, numWorkers, this.gradientThreshold, minibatch)
                .rngSeed(12345)
                .collectTrainingStats(false)
                .batchSizePerWorker(minibatch)              // Minibatch size for each worker
                .updatesThreshold(this.gradientThreshold)   // Encoding threshold (see docs for details)
                .workersPerNode(numWorkersPerNode)          // Workers per node
                .build();
        tm.setCollectTrainingStats(false);

        //If continueTraining==true and checkpoints are available available: Load checkpoint to continue training
        int firstSubsetIdx = 0;
        if (continueTraining) {
            Pair<Integer,ComputationGraph> p = loadCheckpoint();
            if(p != null){
                firstSubsetIdx = p.getFirst();
                net = p.getSecond();
            }
        }


        //Setup saving of parameter snapshots. This is so we can calculate accuracy vs. time
        final AtomicBoolean isTraining = new AtomicBoolean(false);
        final File baseParamSaveDir = new File(outputPath, "paramSnapshots");
        if (!baseParamSaveDir.exists())
            baseParamSaveDir.mkdirs();

        //Prepare Spark version of neural net
        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, tm);
        sparkNet.setCollectTrainingStats(tm.getIsCollectTrainingStats());

        // Add listeners
        sparkNet.setListeners(new PerformanceListener(listenerFrequency, true));

        // Time setup
        long endTimeMs = System.currentTimeMillis();
        double elapsedSec = (endTimeMs - startTime) / MILLISEC_PER_SEC;
        log.info("Setup timing: {} s", elapsedSec);
        results.append("Setup timing: ").append(elapsedSec).append(" sec\n");

        String resultsFile = FilenameUtils.concat(outputPath, "results.txt");
        if (new File(resultsFile).exists()) {
            String str = "\n\n\n============================================================================" + results.toString();
            FileUtils.writeStringToFile(new File(resultsFile), str, Charset.forName("UTF-8"), true);
        } else {
            FileUtils.writeStringToFile(new File(resultsFile), results.toString(), Charset.forName("UTF-8"));
        }

        //Random split into RDDs of exactly "convNumBatches" objects
        long countTrain = trainDataPaths.count();
        JavaRDD<String>[] trainSubsets;
        if(batchesBtwCheckpoints > 1){
            trainSubsets = SparkUtils.balancedRandomSplit((int) countTrain, batchesBtwCheckpoints, trainDataPaths);
        } else {
            trainSubsets = (JavaRDD<String>[])new JavaRDD[]{trainDataPaths};
        }

        DataSetLoader datasetLoader = new LoadDataSetsFunction(wordVectorsPath,
                PatentLabelGenerator.classLabelFilteredCounts().size(),
                300);

        //Before training starts: start the thread to track convergence. This thread asyncronously saves params periodically for later evaluation
        AtomicInteger currentSubset = new AtomicInteger(0);
        Queue<ToEval> toEvalQueue = ConvergenceRunnable.startConvergenceThread(baseParamSaveDir, currentSubset, isTraining, saveFreqSec, sparkNet.getNetwork().params());
        log.info("Network saving thread started: saving copy every {} sec", saveFreqSec);


        boolean firstSave = true;
        long startTrain = System.currentTimeMillis();
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = firstSubsetIdx; i < trainSubsets.length; i++) {
                currentSubset.set(i);
                log.info("Starting training: epoch {} of {}, subset {} of {} ({} minibatches)", (epoch + 1), numEpochs, (i + 1), trainSubsets.length, batchesBtwCheckpoints);
                long start = System.currentTimeMillis();
                isTraining.set(true);
                sparkNet.fitPaths(trainSubsets[i], datasetLoader);
                isTraining.set(false);
                long end = System.currentTimeMillis();
                log.info("Finished training: epoch {} of {}, subset {} of {} ({} minibatches) in {} sec", (epoch + 1), numEpochs, (i + 1), trainSubsets.length, batchesBtwCheckpoints, (end-start)/1000);

                String fileName = "netCheckpoint_" + System.currentTimeMillis() + "_epoch" + epoch + "_subset" + i + ".zip";
                String outpath = FilenameUtils.concat(outputPath, "nets/" + fileName);
                File f = new File(outpath);
                if (firstSave) {
                    firstSave = false;
                    f.getParentFile().mkdirs();
                }
                ModelSerializer.writeModel(sparkNet.getNetwork(), f, true);
                log.info("Saved network checkpoint to {}", outpath);

                //Now, evaluate the saved checkpoint files
                List<ToEval> toEval = new ArrayList<>();
                while(toEvalQueue.size() > 0){
                    toEval.add(toEvalQueue.remove());
                }

                if(totalExamplesTest > 0 && toEval.size() > 0) {
                    log.info("Starting evaluation of {} checkpoint files", toEval.size());
                    ComputationGraph cgForEval = sparkNet.getNetwork().clone();
                    SparkComputationGraph scgForEval = new SparkComputationGraph(sc, cgForEval, null);
                    for (ToEval te : toEval) {
                        INDArray params = Nd4j.readBinary(te.getFile());
                        cgForEval.params().assign(params);

                        long startEval = System.currentTimeMillis();
                        IEvaluation[] evals = scgForEval.doEvaluation(testDataPaths, 4, minibatch, datasetLoader, new Evaluation());
                        long endEval = System.currentTimeMillis();

                        StringBuilder sb = new StringBuilder();
                        Evaluation e = (Evaluation) evals[0];
                        sb.append("network ").append(te.getCount()).append(" trainingMs ").append(te.getDurationSoFar())
                                .append(" evalMS ").append(endEval - startEval)
                                .append(" accuracy ").append(e.accuracy()).append(" f1 ").append(e.f1()).append("\n");

                        FileUtils.writeStringToFile(new File(resultsFile), sb.toString(), Charset.forName("UTF-8"), true);    //Append new output to file
                        saveEvaluation(false, evals, sc);
                        log.info("Evaluation: {}", sb.toString());

                    }
                }

                if(maxRuntimeSec > 0 && (System.currentTimeMillis() - startTrain)/MILLISEC_PER_SEC > maxRuntimeSec){
                    log.info("Terminating due to exceeding max runtime");
                    epoch = numEpochs;
                    break;
                }
            }
            firstSubsetIdx = 0;
        }

        log.info("----- Example Complete -----");
        sc.stop();
        System.exit(0);
    }

    private void writeConfig(JavaSparkContext sc) throws Exception {
        long time = System.currentTimeMillis();

        StringBuilder sb = new StringBuilder();
        sb.append("Output Path: ").append(outputPath).append("\n")
                .append("Time: ").append(time).append("\n")
                .append("numEpoch: ").append(numEpochs).append("\n")
                .append("minibatch: ").append(minibatch).append("\n")
                .append("numNodes: ").append(numNodes).append("\n")
                .append("numWorkpersPerNode: ").append(numWorkersPerNode).append("\n")
                .append("Listener Frequency: ").append(listenerFrequency).append("\n")
                .append("Azure Storage Account: ").append(azureStorageAcct).append("\n")
                .append("Gradient threshold: ").append(gradientThreshold).append("\n")
                .append("Controller: ").append(masterIP).append("\n")
                .append("Port: ").append(port).append("\n")
                .append("Network Mask: ").append(networkMask).append("\n")
                .append("Word vectors path: ").append(wordVectorsPath).append("\n")
                .append("Continue training: ").append(continueTraining).append("\n")
                .append("saveFreqSec: ").append(saveFreqSec).append("\n")
                .append("\n");

        sb.append("\n\n")
                .append("Spark Default Parallelism: ").append(sc.defaultParallelism()).append("\n");

        String str = sb.toString();
        log.info(str);

        String path = FilenameUtils.concat(outputPath, "experimentConfig.txt");
        log.info("Writing experiment config and info to file: {}", path);
        SparkUtils.writeStringToFile(path, str, sc);
    }


    private void evaluateOnly(JavaSparkContext sc, ComputationGraph net, JavaRDD<String> testDataPaths) throws IOException {
        log.info("***** Starting Evaluation only for directory {} *****", outputPath);

        File f = new File(outputPath, "paramSnapshots");
        if(!f.exists() || !f.isDirectory()){
            throw new IllegalStateException("paramSnapshots directory does not exist: " + f.getAbsolutePath());
        }
        File[] content = f.listFiles();
        if(content == null || content.length == 0)
            throw new IllegalStateException("No saved network parameters at " + f.getAbsolutePath());
        Arrays.sort(content);
        testDataPaths.cache();

        log.info("Found {} parameter instances to evaluate", content.length);
        SparkComputationGraph scgForEval = new SparkComputationGraph(sc, net, null);

        DataSetLoader dsl = new LoadDataSetsFunction(
                wordVectorsPath,
                PatentLabelGenerator.classLabelFilteredCounts().size(),
                300);

        File evalResultFile = new File(outputPath, "evaluationOnly_" + System.currentTimeMillis() + ".txt");

        long allStart = System.currentTimeMillis();
        for(int i=0; i<content.length; i++ ){
            if(!content[i].isFile() || content[i].length() == 0) {
                log.error("Skipping file: " + content[i].getAbsolutePath());
                continue;
            }
            log.info("Starting evaluation: {} of {} - {}", (i+1), content.length, content[i].getAbsolutePath());

            INDArray params;
            try{
                params = Nd4j.readBinary(content[i]);
            } catch (Throwable t){
                log.error("Error loading file: {}", content[i].getAbsolutePath(), t);
                continue;
            }
            scgForEval.getNetwork().params().assign(params);

            long startEval = System.currentTimeMillis();
            IEvaluation[] evals = scgForEval.doEvaluation(testDataPaths, 4, minibatch, dsl, new Evaluation());
            long endEval = System.currentTimeMillis();
            StringBuilder sb = new StringBuilder();
            Evaluation e = (Evaluation) evals[0];
            sb.append(content[i].getAbsolutePath()).append(" evalMS ").append(endEval - startEval)
                    .append(" accuracy ").append(e.accuracy()).append(" f1 ").append(e.f1()).append("\n");

            String s = sb.toString();
            FileUtils.writeStringToFile(evalResultFile, s, Charset.forName("UTF-8"), true);    //Append new output to file
            saveEvaluation(false, evals, sc);
            log.info("Evaluation: {}", s);
        }
        long allEnd = System.currentTimeMillis();

        log.info("----- Completed evaluation in {} sec -----", (allEnd-allStart)/1000);
        return;
    }

    private Pair<Integer,ComputationGraph> loadCheckpoint() throws IOException {
        File f = new File(outputPath, "nets");
        File[] list = f.listFiles();
        boolean continued = false;
        ComputationGraph net = null;
        int firstSubsetIdx = 0;
        if (list != null && list.length > 0) {
            //find most recent (largest timestamp)
            //Format: net_1531848680353_epoch0_subset217.zip
            long maxTimestamp = 0;
            File maxFile = null;
            for (File checkpoint : list) {
                if (!checkpoint.isFile())
                    continue;
                String name = checkpoint.getName();
                String[] split = name.split("_");
                long ts = Long.parseLong(split[1]);
                if (ts > maxTimestamp) {
                    maxTimestamp = ts;
                    maxFile = checkpoint;
                    firstSubsetIdx = Integer.parseInt(split[3].substring(6, split[3].indexOf('.'))) + 1;
                }
            }

            if (maxFile == null) {
                log.warn("Could not continue - no checkpoints to load from");
            } else {
                net = ComputationGraph.load(maxFile, true);
                log.info("Continued from checkpoint: {}", maxFile.getAbsolutePath());
                continued = true;
            }
        }

        if (!continued) {
            log.warn("*** Could not continue training (no checkpoint files) even though continueTraining == true ***");
            return null;
        }

        return new Pair<>(firstSubsetIdx, net);
    }


    private void saveEvaluation(boolean train, IEvaluation[] evaluations, JavaSparkContext sc) throws IOException {
        String evalPath = FilenameUtils.concat(outputPath, ("evaluation_" + (train ? "train" : "test")));
        //Write evaluations to disk
        for (int i = 0; i < evaluations.length; i++) {
            String path = FilenameUtils.concat(evalPath, "evaluation_" + System.currentTimeMillis() + "_" + i + ".txt");
            SparkUtils.writeStringToFile(path, evaluations[i].stats(), sc);
        }
    }

    private JavaRDD<String> listPathsSubset(JavaSparkContext sc, String path, int max, int rngSeed) throws IOException {
        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(URI.create(path), config);
        RemoteIterator<LocatedFileStatus> fileIter = hdfs.listFiles(new org.apache.hadoop.fs.Path(path), true);

        List<String> paths = new ArrayList<>();
        while (fileIter.hasNext()) {
            String filePath = fileIter.next().getPath().toString();
            paths.add(filePath);
        }

        //Now, get a consistent random subset - assuming here that file listing isn't consistent
        Collections.sort(paths);
        int[] arr = new int[paths.size()];
        for( int i=0; i<arr.length ; i++){
            arr[i] = i;
        }
        MathUtils.shuffleArray(arr, rngSeed);

        List<String> out = new ArrayList<>();
        for( int i=0; i<arr.length && i < max; i++ ){
            out.add(paths.get(arr[i]));
        }

        return sc.parallelize(out);
    }
}

package org.deeplearning4j.patent;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.patent.preprocessing.PatentLabelGenerator;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.patent.utils.data.LoadDataSetsFunction;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.LocalFileSourceFactory;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.Charset;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * This is a local (single machine, no Spark) implementation of the example.
 * It's main purpose is to provide a single machine performance comparison on the same network.
 * This allows for comparing performance for a single machine vs. cluster, in terms of "time to accuracy X"
 *
 * Before running this script, you will need to run the Spark preprocessing, and the train/test data (or a subset there-of)
 * to the local machine
 *
 * @author Alex Black
 */
public class LocalTraining {
    private static final Logger log = LoggerFactory.getLogger(LocalTraining.class);
    public static final int MILLISEC_PER_SEC = 1000;

    @Parameter(names = {"--outputPath"}, description = "Local output path/directory to write results to", required = true)
    private String outputPath;

    @Parameter(names = {"--dataDir"}, description = "Directory containing the training data - locally", required = true)
    private String dataDir;

    @Parameter(names = {"--w2vPath"}, description = "Path to the Word2Vec vectors - locally", required = true)
    private String w2vPath;

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 1;

    @Parameter(names = {"--rngSeed"}, description = "Random number generator seed (for repeatability)")
    private int rngSeed = 12345;

    @Parameter(names = {"--totalExamplesTrain"}, description = "Total number of examples for training", required = true)
    private int totalExamplesTrain = -1;

    @Parameter(names = {"--totalExamplesTest"}, description = "Total number of examples for testing")
    private int totalExamplesTest = 5000;

    @Parameter(names = {"--saveConvergenceNets"}, description = "If true, save networks at each evaluation point during training")
    private boolean saveConvergenceNets = true;

    @Parameter(names = {"--batchSize"}, description = "Batch size for training")
    private int batchSize = 32;

    @Parameter(names = {"--listenerFrequency"}, description = "Listener Frequency")
    private int listenerFrequency = 10;

    @Parameter(names = {"--convergenceEvalFrequencyBatches"}, description = "Perform convergence evaluation every N minibatches total")
    private int convergenceEvalFrequencyBatches = 400;

    @Parameter(names = {"--maxTrainingTimeMin"}, description = "Maximum time for training, in minutes (or < 0 to use no limit)")
    private int maxTrainingTimeMin = -1;

    /**
     * Main function
     *
     * @param args
     * @throws Exception
     */
    public static void main(String... args) throws Exception {
        new LocalTraining().entryPoint(args);
    }

    /**
     * JCommander entry point
     *
     * @param args
     * @throws Exception
     */
    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);
        File resultFile = new File(outputPath, "results.txt");   //Output will be written here
        Preconditions.checkArgument(convergenceEvalFrequencyBatches > 0, "convergenceEvalFrequencyBatches must be positive: got %s", convergenceEvalFrequencyBatches);

        Nd4j.getMemoryManager().setAutoGcWindow(15000);

        // Prepare neural net
        ComputationGraph net = new ComputationGraph(NetworkConfiguration.getConf());
        net.init();
        net.setListeners(new PerformanceListener(listenerFrequency, true));
        log.info("Parameters: {}", net.params().length());

        //Write configuration
        writeConfig();

        // Train neural net
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        int subsetCount = 0;
        boolean firstSave = true;
        long overallStart = System.currentTimeMillis();
        boolean exit = false;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Training
            log.info("epoch {} training begin: {}", epoch + 1, dtf.format(LocalDateTime.now()));
            // Prepare training data. Note we'll get this again for each epoch in case we are using early termination iterator
            // plus randomization. This is to ensure consistency between epochs
            DataSetIterator trainData = getDataIterator(dataDir, true, totalExamplesTrain, batchSize, rngSeed);

            //For convergence purposes: want to split into subsets, time training on each subset, an evaluate
            while(trainData.hasNext()){
                subsetCount++;
                log.info("Starting training: epoch {} of {}, subset {} ({} minibatches)", (epoch + 1), numEpochs, subsetCount, convergenceEvalFrequencyBatches);
                DataSetIterator subset = new EarlyTerminationDataSetIterator(trainData, convergenceEvalFrequencyBatches);
                int itersBefore = net.getIterationCount();
                long start = System.currentTimeMillis();
                net.fit(subset);
                long end = System.currentTimeMillis();
                int iterAfter = net.getIterationCount();

                //Save model
                if(saveConvergenceNets){
                    String fileName = "net_" + System.currentTimeMillis() + "_epoch" + epoch + "_subset" + subsetCount + ".zip";
                    String outpath = FilenameUtils.concat(outputPath, "nets/" + fileName);
                    File f = new File(outpath);
                    if(firstSave){
                        firstSave = false;
                        f.getParentFile().mkdirs();
                    }
                    ModelSerializer.writeModel(net, f, true);
                    log.info("Saved network to {}", outpath);
                }


                DataSetIterator test = getDataIterator(dataDir, false, totalExamplesTrain, batchSize, rngSeed);
                long startEval = System.currentTimeMillis();
                IEvaluation[] evals = net.doEvaluation(test, new Evaluation(), new ROCMultiClass());
                long endEval = System.currentTimeMillis();

                StringBuilder sb = new StringBuilder();
                Evaluation e = (Evaluation) evals[0];
                ROCMultiClass r = (ROCMultiClass) evals[1];
                sb.append("epoch ").append(epoch + 1).append(" of ").append(numEpochs).append(" subset ").append(subsetCount)
                        .append(" subsetMiniBatches ").append(iterAfter - itersBefore)      //Note: "end of epoch" effect - may be smaller than subset size
                        .append(" trainMS ").append(end - start).append(" evalMS ").append(endEval - startEval)
                        .append(" accuracy ").append(e.accuracy()).append(" f1 ").append(e.f1())
                        .append(" AvgAUC ").append(r.calculateAverageAUC()).append(" AvgAUPRC ").append(r.calculateAverageAUCPR()).append("\n");

                FileUtils.writeStringToFile(resultFile, sb.toString(), Charset.forName("UTF-8"), true);    //Append new output to file
                saveEvaluation(false, evals);
                log.info("Evaluation: {}", sb.toString());

                if(maxTrainingTimeMin > 0 && (System.currentTimeMillis() - overallStart) / 60000 > maxTrainingTimeMin){
                    log.info("Exceeded max training time of {} minutes - terminating", maxTrainingTimeMin);
                    exit = true;
                    break;
                }
            }
            if(exit)
                break;
        }

        File dir = new File(outputPath, "trainedModel.bin");
        net.save(dir, true);
    }



    private void writeConfig() throws Exception {
        long time = System.currentTimeMillis();

        StringBuilder sb = new StringBuilder();
        sb.append("Output Path: ").append(outputPath).append("\n")
                .append("Time: ").append(time).append("\n")
                .append("RNG Seed: ").append(rngSeed).append("\n")
                .append("Total Examples (Train): ").append(totalExamplesTrain).append("\n")
                .append("Total Examples (Test): ").append(totalExamplesTest).append("\n")
                .append("numEpoch: ").append(numEpochs).append("\n")
                .append("BatchSize: ").append(batchSize).append("\n")
                .append("Listener Frequency: ").append(listenerFrequency).append("\n")
                .append("\n");

        String str = sb.toString();
        log.info(str);

        //Write to file:
        String toWrite = sb.toString();

        String path = FilenameUtils.concat(outputPath, "experimentConfig.txt");
        log.info("Writing experiment config and info to file: {}", path);
        FileUtils.writeStringToFile(new File(outputPath, "experimentConfig.txt"), toWrite, Charset.forName("UTF-8"));
    }

    private void saveEvaluation(boolean train, IEvaluation[] evaluations) throws Exception {
        String evalPath = FilenameUtils.concat(outputPath, ("evaluation_" + (train ? "train" : "test")));
        //Write evaluations to disk
        for( int i=0; i<evaluations.length; i++ ){
            String path = FilenameUtils.concat(evalPath, "evaluation_" + i + ".txt");
            FileUtils.writeStringToFile(new File(path), evaluations[i].stats(), Charset.forName("UTF-8"));
        }
    }


    public DataSetIterator getDataIterator(String dataRootDir, boolean train, int totalExamples, int batchSize, int seed) {
        File root = new File(dataRootDir, train ? "train" : "test");
        List<String> all = new ArrayList<>();
        File[] files = root.listFiles();
        if(files == null || files.length == 0){
            throw new IllegalStateException("Did not find files in directory " + root.getAbsolutePath());
        }
        for(File f : files){
            all.add(f.getAbsolutePath());
        }
        Collections.sort(all);
        int totalBatches = (totalExamples < 0 ? -1 : totalExamples / batchSize);
        if(totalBatches > 0 && totalBatches < all.size()){
            Random r = new Random(seed);
            int[] order = new int[all.size()];
            for( int i=0; i<order.length; i++ ){
                order[i] = i;
            }
            MathUtils.shuffleArray(order, r);
            List<String> from = all;
            all = new ArrayList<>();
            for( int i=0; i<totalBatches; i++ ){
                all.add(from.get(order[i]));
            }
        }

        Loader<DataSet> loader = new LoadDataSetsFunction(w2vPath, PatentLabelGenerator.classLabelToIndex().size(), 300);
        return new DataSetLoaderIterator(all, loader, new LocalFileSourceFactory());
    }
}

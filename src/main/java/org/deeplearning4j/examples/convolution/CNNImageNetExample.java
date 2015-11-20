package org.deeplearning4j.examples.convolution;

import com.fasterxml.jackson.databind.ObjectMapper;
import static org.junit.Assert.*;
import net.didion.jwnl.data.Exc;
import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.recordreader.ImageNetRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.convolution.sampleNetStructure.AlexNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.LeNet;
import org.deeplearning4j.examples.convolution.sampleNetStructure.VGGNetA;
import org.deeplearning4j.examples.convolution.sampleNetStructure.VGGNetD;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import static org.junit.Assert.assertTrue;

/**
 *
 * Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
 * Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei.
 * (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
 *

 * Created by nyghtowl on 9/24/15.
 */
public class CNNImageNetExample {

    private static final Logger log = LoggerFactory.getLogger(CNNImageNetExample.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--modelType",usage="Type of model (AlexNet, VGGNetA, VGGNetB)",aliases = "-mT")
    private String modelType = "AlexNet";
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    private int batchSize = 8;
    @Option(name="--numBatches",usage="Number of batches",aliases="-nB")
    private int numBatches = 1;
    @Option(name="--numTestBatches",usage="Number of test batches",aliases="-nTB")
    private int numTestBatches = 1;
    @Option(name="--numEpochs",usage="Number of epochs",aliases="-nE")
    private int numEpochs = 1;
    @Option(name="--iterations",usage="Number of iterations",aliases="-i")
    private int iterations = 1;
    @Option(name="--numCategories",usage="Number of categories",aliases="-nC")
    private int numCategories = 4;
    @Option(name="--trainFolder",usage="Train folder",aliases="-taF")
    private String trainFolder = "train";
    @Option(name="--testFolder",usage="Test folder",aliases="-teF")
    private String testFolder = "val/val-sample";
    @Option(name="--saveParams",usage="Save parameters",aliases="-sP")
    private boolean saveParams = true;

    public CNNImageNetExample() {
    }

    public void doMain(String[] args) throws Exception {
        File outputPath = defineOutputDir();

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);

        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        boolean train = true;
        boolean splitTrainData = false;
        boolean gradientCheck = false;
        boolean loadParams = false;

        MultiLayerNetwork model = null;
        DataSetIterator dataIter, testIter;
        long startTimeTrain = 0;
        long endTimeTrain = 0;
        long startTimeEval = 0;
        long endTimeEval = 0;

        // libraries like Caffe scale to 256?
        final int numRows = 224;
        final int numColumns = 224;
        int nChannels = 3;
        int outputNum = 1860;
        int seed = 123;
        int listenerFreq = 1;

        int totalCSLExamples2013 = 1281167;
        int totalCSLValExamples2013 = 50000;
        int totalTrainNumExamples = batchSize * numBatches;
        int totalTestNumExamples = batchSize * numTestBatches;

        String basePath = System.getProperty("user.home") + File.separator + "Documents" + File.separator + "skymind" + File.separator + "imagenet" + File.separator;
        String trainData = basePath + trainFolder + File.separator;
        String testData = basePath + testFolder + File.separator;
        String labelPath = basePath + "cls-loc-labels.csv";
        String valLabelMap = basePath + "cls-loc-val-map.csv";
        String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};

        System.out.println(modelType.toString());
        log.info("Load data....");
        RecordReader recordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath);
        recordReader.initialize(new LimitFileSplit(new File(trainData), allForms, totalTrainNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);


        log.info("Build model....");
        switch (modelType) {
            case "LeNet":
                model = new LeNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "AlexNet":
                model = new AlexNet(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "VGGNetA":
                model = new VGGNetA(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
            case "VGGNetD":
                model = new VGGNetD(numRows, numColumns, nChannels, outputNum, seed, iterations).init();
                break;
        }

        if(loadParams) {
            Layer layer;
//            String[] specificLayers ={ "conv1", "conv2", "conv3", "conv4", "ffn1", "ff2n", "output"};
//            for(String name: specificLayers) {
//                layer = model.getLayer(name);
//                loadParameters(layer, new File(outputPath + File.separator + layer.conf().getLayer().getLayerName() + ".json"));
//            }

            // Load specific layers for VGGD from VGGA
            int[] layerIds = {0,1,3,4,18,19,20};
            for(int layerId: layerIds) {
                layer = model.getLayer(layerId);
                loadParameters(layer, new File(outputPath + File.separator + layer.conf().getLayer().getLayerName() + ".json"));
            }

        }

        // Listeners
        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(1).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
//        model.setListeners(Arrays.asList((IterationListener) new HistogramIterationListener(listenerFreq)));
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));

        if (gradientCheck) gradientCheck(dataIter, model);
        
        if (train) {
            log.info("Train model....");

            //TODO need dataIter that loops through set number of examples like SamplingIter but takes iter vs dataset
            MultipleEpochsIterator epochIter = new MultipleEpochsIterator(numEpochs, dataIter);
////                asyncIter = new AsyncDataSetIterator(dataIter, 1); TODO doesn't have next(num)
            Evaluation eval = new Evaluation(recordReader.getLabels());


            // split training and evaluatioin out of same DataSetIterator
            if (splitTrainData) {
                int splitTrainNum = (int) (batchSize * .8);
                int numTestExamples = totalTrainNumExamples / (numBatches) - splitTrainNum;

                for (int i = 0; i < numEpochs; i++) {
                    for (int j = 0; j < numBatches; j++)
                        model.fit(epochIter.next(splitTrainNum)); // if spliting test train in same dataset - put size of train in next
                    eval = evaluatePerformance(model, epochIter, numTestExamples, eval);
                }
            } else{
                // track training time
                startTimeTrain = System.currentTimeMillis();
                for (int i = 0; i < numEpochs; i++) {
                    for (int j = 0; j < numBatches; j++)
                        model.fit(epochIter.next());
                }
                endTimeTrain = System.currentTimeMillis();

                // use different data sets for train and test
                // TODO uncomment code when using full validation set
//                RecordReader testRecordReader = new ImageNetRecordReader(numColumns, numRows, nChannels, true, labelPath, valLabelMap); // use when pulling from main val for all labels
//                testRecordReader.initialize(new LimitFileSplit(new File(testData), allForms, totalNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));
//                testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, numRows * numColumns * nChannels, 1860);

                recordReader.initialize(new LimitFileSplit(new File(testData), allForms, totalTestNumExamples, numCategories, Pattern.quote("_"), 0, new Random(123)));
                testIter = new RecordReaderDataSetIterator(recordReader, batchSize, numRows * numColumns * nChannels, 1860);

                MultipleEpochsIterator testEpochIter = new MultipleEpochsIterator(numEpochs, testIter);

                // track evaluating time
                startTimeEval = System.currentTimeMillis();
                eval = evaluatePerformance(model, testEpochIter, batchSize, eval);

                endTimeEval = System.currentTimeMillis();

            }

            log.info(eval.stats());
            System.out.println("Total training runtime: " + ((endTimeTrain-startTimeTrain)/60000) + " minutes");
            System.out.println("Total evaluation runtime: " + ((endTimeEval - startTimeEval) / 60000) + " minutes");
            log.info("****************************************************");

            if (saveParams) saveModelAndParameters(model, outputPath);

            MultiLayerNetwork reloadedNet = loadModel(new File(outputPath + File.separator + modelType.toString() + "-conf.json"), new File(outputPath + File.separator + modelType.toString() + ".bin"));
            assertEquals("Generated model and loaded model parameters are not equal", model.params(), reloadedNet.params());

            log.info("****************Example finished********************");
        }
    }


    private Evaluation evaluatePerformance(MultiLayerNetwork model, MultipleEpochsIterator iter, int testBatchSize, Evaluation eval){
        log.info("Evaluate model....");
        DataSet imgNet;
        INDArray output;

        //TODO setup iterator to randomize
        for(int i=0; i < numTestBatches; i++){
            imgNet = iter.next(testBatchSize);
            output = model.output(imgNet.getFeatureMatrix());
            eval.eval(imgNet.getLabels(), output);
        }
        return eval;
    }

    private File defineOutputDir(){
        String tmpDir = System.getProperty("java.io.tmpdir");
        String outputPath = File.separator + modelType.toString() + File.separator + "output";
        File dataDir = new File(tmpDir,outputPath);
        if (!dataDir.getParentFile().exists())
            dataDir.mkdirs();
        return dataDir;

    }

    private void saveModelAndParameters(MultiLayerNetwork net, File dataDir) throws IOException {
        System.out.println("Saving model and parameters to " + dataDir.toString() + "...");

        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dataDir + File.separator + modelType.toString() + ".bin"));
        Nd4j.write(bos, net.params());
        bos.flush();
        bos.close();

        // save model configuration
        FileUtils.write(new File(dataDir + File.separator + modelType.toString() + "-conf.json"), net.conf().toJson());

        // save parameter table for each layer
        for (Layer layer: net.getLayers()) {
            if (layer.params() != null)
                FileUtils.write(new File(dataDir + File.separator + layer.conf().getLayer().getLayerName() + ".json"), layer.paramTable().toString());
        }
    }

    private MultiLayerNetwork loadModel(File confPath, File paramsBinPath) throws IOException {
        System.out.println("Loading saved model...");
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(confPath));
        DataInputStream dis = new DataInputStream(new FileInputStream(paramsBinPath.toString()));
        INDArray newParams = Nd4j.read(dis);
        dis.close();

        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);

        return savedNetwork;

    }

    private void loadParameters(Layer layer, File paramPath) throws IOException{
        System.out.println("Loading parameters...");
        String params = FileUtils.readFileToString(paramPath);
        Map<String, INDArray> result = new ObjectMapper().readValue(params, HashMap.class);
        layer.setParamTable(result);
    }

    private void gradientCheck(DataSetIterator dataIter, MultiLayerNetwork model){
        DataSet imgNet;
        log.info("Gradient Check....");

        imgNet = dataIter.next();
        String name = new Object() {
        }.getClass().getEnclosingMethod().getName();

        model.setInput(imgNet.getFeatures());
        model.setLabels(imgNet.getLabels());
        model.computeGradientAndScore();
        double scoreBefore = model.score();
        for (int j = 0; j < 1; j++)
            model.fit(imgNet);
        model.computeGradientAndScore();
        double scoreAfter = model.score();
//            String msg = name + " - score did not (sufficiently) decrease during learning (before=" + scoreBefore + ", scoreAfter=" + scoreAfter + ")";
//            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
        for (int j = 0; j < model.getnLayers(); j++)
            System.out.println("Layer " + j + " # params: " + model.getLayer(j).numParams());

        double default_eps = 1e-6;
        double default_max_rel_error = 0.25;
        boolean print_results = true;
        boolean return_on_first_failure = false;
        boolean useUpdater = true;

        boolean gradOK = GradientCheckUtil.checkGradients(model, default_eps, default_max_rel_error,
                print_results, return_on_first_failure, imgNet.getFeatures(), imgNet.getLabels(), useUpdater);

        assertTrue(gradOK);

    }
    public static void main(String[] args) throws Exception {
        new CNNImageNetExample().doMain(args);
    }


}

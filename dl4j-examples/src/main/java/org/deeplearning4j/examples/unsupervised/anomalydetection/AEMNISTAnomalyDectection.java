package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


/**this's unsupervised examples for anormaly detection using autoencoder on MNIST,the example don't use the label of the MNIST
 * when doing the example,the dataset contains  some digits written by me,especially those images don't look like digit
 * of course you can use ImageWithBytesUtil  to generate images from ubyte format files of MNIST, and then add your pictures to training or testing directory .
 * The goal is to identify outliers digits, i.e., those digits that are unusual or not like the typical digits.
 * in fact, the normal images should have low reconfiguration error,whereas those weird pictures have high reconstruction error
 *
 * @author WANG FENG
 */
public class AEMNISTAnomalyDectection {
    private static final Logger log = LoggerFactory.getLogger(AEMNISTAnomalyDectection.class);


    private static String rootPath = System.getProperty("user.dir");
    private static String modelPath = rootPath.substring(0, rootPath.lastIndexOf(File.separatorChar)) + File.separatorChar + "out" + File.separatorChar + "models" + File.separatorChar + "mnistAbnormalDetectedModel.json";

    private static List<INDArray> featuresTrain = new ArrayList<>();
    private static List<INDArray> featuresTest = new ArrayList<>();
    private static double threshold = 0.01;//setting the value based on the situation when traning model shows loss value,


    public static void main(String[] args) throws Exception {

        MultiLayerNetwork net = createModel();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File("java.io.tmpdir"));
        uiServer.attach(statsStorage);
        net.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(10));

        DataSetIterator iter = new MnistIterator(100,true);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            featuresTrain.add(ds.getFeatures());
        }
        iter = new MnistIterator(100,false);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            featuresTest.add(ds.getFeatures());
        }
        //Train model
        int nEpochs = 35;
        for( int i = 0; i < nEpochs; i ++ ){
            for(INDArray data : featuresTrain){
                net.fit(data,data);
            }
            System.out.println("Epoch " + i + " complete");
        }
        List<Pair<Double,INDArray>> evalList = new ArrayList<>();
        double totalScore = 0;
        for( int i = 0; i < featuresTest.size(); i ++ ){
            INDArray testData = featuresTest.get(i);
            int nRows = testData.rows();
            for( int j = 0; j < nRows; j ++){
                INDArray example = testData.getRow(j);
                double score = net.score(new DataSet(example,example));
                totalScore += score;
                evalList.add(new ImmutablePair<>(score, example));
            }
            System.out.println("featuresTest " + i + " complete");
        }
        //Sort each list in the list by score
        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                return Double.compare(o1.getLeft(),o2.getLeft());
            }
        };

        Collections.sort(evalList, c);
        List<INDArray> normalList = new ArrayList<>();
        List<INDArray> anomalyList = new ArrayList<>();

        int listsize = evalList.size();
        for( int j = 0; j < listsize && j < 60; j ++ ){
            normalList.add(evalList.get(j).getRight());
            anomalyList.add(evalList.get(listsize -j -1).getRight());
        }
        VisualizerUtil bestVisualizer = new VisualizerUtil(2.0,normalList,"best (High Rec. Error)");
        bestVisualizer.visualize();
        VisualizerUtil worstVisualizer = new VisualizerUtil(2.0,anomalyList,"Worst (High Rec. Error)");
        worstVisualizer.visualize();

        //save model
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            modelFile.createNewFile();
        }
        log.info( modelFile.getPath() );

        net.save(modelFile,true);
    }

    public static MultiLayerNetwork createModel() {
        //Set up network. 784 in/out (as MNIST images are 28x28).
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123456)
            .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nadam())
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .list()
            .layer(new LSTM.Builder().name("encoder0").nIn(784).nOut(800).build())
            .layer(new LSTM.Builder().name("encoder1").nOut(250).build())
            .layer(new LSTM.Builder().name("encoder2").nOut(10).build())
            .layer(new LSTM.Builder().name("decoder1").nOut(250).build())
            .layer(new LSTM.Builder().name("decoder2").nOut(800).build())
            .layer(new RnnOutputLayer.Builder().name("output").nOut(784)
                .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
            .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

}

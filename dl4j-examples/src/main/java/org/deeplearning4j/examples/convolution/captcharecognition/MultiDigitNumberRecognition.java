package org.deeplearning4j.examples.convolution.captcharecognition;


import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * @Description This is a demo that multi-digit number recognition. The maximum length is 6 digits.
 * If it is less than 6 digits, then zero is added to last
 * Training set: There were 14108 images, and they were used to train a model.
 * Testing set: in total 108 images,they copied from the training set,mainly to determine whether it's good that the model fited training data
 * Verification set: The total quantity of the images has 248 that's the unknown image,the main judgment that the model is good or bad
 * Other: Using the current architecture and hyperparameters, the accuracy of the best model prediction validation set is (215-227) / 248 with different epochs
 * of course, if you're interested, you can continue to optimize.
 * @author WangFeng
 */
public class MultiDigitNumberRecognition {


    private static final Logger log = LoggerFactory.getLogger(MultiDigitNumberRecognition.class);

    private static long seed = 123;
    private static int epochs = 50;
    private static int batchSize = 15;
    private static String rootPath = System.getProperty("user.dir");
    private static String modelPath = rootPath.substring(0, rootPath.lastIndexOf(File.separatorChar)) + File.separatorChar + "out" + File.separatorChar + "models" + File.separatorChar + "validateCodeCheckModel.json";


    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        System.out.println(startTime);
        File modelFile = new File(modelPath);
        boolean hasFile = modelFile.exists()?true:modelFile.createNewFile();
        log.info( modelFile.getPath() );
       // ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFile);
        ComputationGraph model =  createModel();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener( statsStorage));


        //ImageTransform showTransform = new ShowImageTransform("Display");
        //ImageTransform multiTransform = new MultiImageTransform(showTransform);
        //MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, multiTransform, "train");

        MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, "train");

        MultiDataSetIterator testMulIterator = new MultiRecordDataSetIterator(batchSize,"test");
        MultiDataSetIterator validateMulIterator = new MultiRecordDataSetIterator(batchSize,"validate");
        for ( int i = 0; i < epochs; i ++ ) {
            System.out.println("Epoch=====================" + i);
            model.fit(trainMulIterator);
        }
        ModelSerializer.writeModel(model, modelFile,true);
        long endTime = System.currentTimeMillis();
        System.out.println("=============run time=====================" + (endTime - startTime));


        int tc = 0;
        int tt = 0;
        System.out.println("=====eval model===========test============");
        while(testMulIterator.hasNext()) {
            MultiDataSet mds = testMulIterator.next();
            INDArray[]  output = model.output(mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            int dataNum = batchSize > output[0].rows() ? output[0].rows() : batchSize;
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                String reLabel = "";
                String peLabel = "";
                INDArray preoutput = null;
                INDArray realLabel = null;
                for (int digit = 0; digit < 6; digit ++) {
                    preoutput = output[digit].getRow(dataIndex);
                    peLabel += Nd4j.argMax(preoutput, 1).getInt(0);
                    realLabel = labels[digit].getRow(dataIndex);
                    reLabel += Nd4j.argMax(realLabel, 1).getInt(0);

                }
                if (peLabel.equals(reLabel)) {
                    tt ++;
                }
                tc ++;
                log.info("real image {}  prediction {} status {}",  reLabel,peLabel, peLabel.equals(reLabel));
            }
        }
        System.out.println("test result : sum count =" + tc + " correct count=" + tt );

        System.out.println("=====eval model=====validate==================");
        int vc = 0;
        int vt = 0;
        while(validateMulIterator.hasNext()) {
            MultiDataSet mds = validateMulIterator.next();
            INDArray[]  output = model.output(mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            int dataNum = batchSize > output[0].rows() ? output[0].rows() : batchSize;
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                String reLabel = "";
                String peLabel = "";
                INDArray preoutput = null;
                INDArray realLabel = null;
                for (int digit = 0; digit < 6; digit ++) {
                    preoutput = output[digit].getRow(dataIndex);
                    peLabel += Nd4j.argMax(preoutput, 1).getInt(0);
                    realLabel = labels[digit].getRow(dataIndex);
                    reLabel += Nd4j.argMax(realLabel, 1).getInt(0);
                }
                if (peLabel.equals(reLabel)) {
                    vt ++;
                }
                vc ++;
                log.info("real image {}  prediction {} status {}",  reLabel,peLabel, peLabel.equals(reLabel));
            }
        }
        System.out.println("validate result : sum count =" + vc + " correct count=" + vt );

    }

    public static ComputationGraph createModel() {

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .regularization(true)
            .l2(1e-3)
            .learningRate(1e-3)
            .learningRateDecayPolicy(LearningRatePolicy.Score)
            .lrPolicyDecayRate(1e-7)
            .updater(Updater.ADAM)
            .weightInit( WeightInit.XAVIER_UNIFORM)
            .graphBuilder()
            .addInputs("trainFeatures")
            .setInputTypes(InputType.convolutional(60, 160, 1))
            .setOutputs("out1", "out2", "out3", "out4", "out5", "out6")
            .addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nIn(1).nOut(48).activation( Activation.RELU).build(), "trainFeatures")
            .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn1")
            .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(64).activation( Activation.RELU).build(), "maxpool1")
            .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
                .build(), "cnn2")
            .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(128).activation( Activation.RELU).build(), "maxpool2")
            .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn3")
            .addLayer("cnn4",  new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(256).activation( Activation.RELU).build(), "maxpool3")
            .addLayer("maxpool4",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn4")
            .addLayer("ffn0",  new DenseLayer.Builder().nOut(3072)
                .build(), "maxpool4")
            .addLayer("ffn1",  new DenseLayer.Builder().nOut(3072)
                .build(), "ffn0")
            .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out2", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out3", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out4", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out5", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out6", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .pretrain(false).backprop(true)
            .build();

        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();

        return model;
    }

}

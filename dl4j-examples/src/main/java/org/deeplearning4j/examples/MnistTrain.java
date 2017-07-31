package org.deeplearning4j.examples;

import com.google.common.base.Stopwatch;
import org.deeplearning4j.datasets.iterator.AsyncShieldDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.concurrent.TimeUnit;

public class MnistTrain {
    private static MultiLayerConfiguration mlp() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .trainingWorkspaceMode(WorkspaceMode.SINGLE)
            .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
            .cacheMode(CacheMode.DEVICE)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            //.updater(Nesterovs.builder().momentum(0.95).build()) // <- Gets to 0.01 after 12 epochs
            .updater(RmsProp.builder().learningRate(0.001).rmsDecay(0.0).epsilon(1e-8).build())
            //.updater(Adam.builder().build()) // <- This gets to 0.95
            //.updater(AdaDelta.builder().rho(0.95).epsilon(1e-8).build())
            .learningRate(1.0)

            .weightInit(WeightInit.XAVIER_UNIFORM)
            .activation(Activation.RELU)
            .list(
                new DenseLayer.Builder().nOut(512).build(),
                new DropoutLayer.Builder(0.8).build(),
                new DenseLayer.Builder().nOut(512).build(),
                new DropoutLayer.Builder(0.8).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(10).dropOut(0.5).build()
            )
            .setInputType(new InputType.InputTypeConvolutionalFlat(28, 28, 1))
            .build();

        return conf;
    }

    private static MultiLayerConfiguration cnn() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .trainingWorkspaceMode(WorkspaceMode.NONE)
            .inferenceWorkspaceMode(WorkspaceMode.NONE)
            .cacheMode(CacheMode.NONE)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            //.updater(Nesterovs.builder().momentum(0.95).build()) // <- Gets to 0.01 after 12 epochs
            //.updater(RmsProp.builder().learningRate(0.001).rmsDecay(0.0).epsilon(1e-8).build())
            .updater(Adam.builder().build()) // <- This gets to 0.95
            //.updater(Updater.SGD)
            //.updater(AdaDelta.builder().rho(0.95).epsilon(1e-8).build())
            .learningRate(0.01)

            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .list(
                new ConvolutionLayer.Builder(3, 3)/*.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)*/.stride(1, 1).nIn(1).nOut(32).build(),
                new ConvolutionLayer.Builder(3, 3)/*.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)*/.stride(1, 1).nIn(32).nOut(64).build(),
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                new DropoutLayer.Builder(0.75).build(),
                new DenseLayer.Builder().nOut(128).dropOut(5).build(),
                new DropoutLayer.Builder(0.5).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(10).dropOut(0.5).build()
            )
            .setInputType(new InputType.InputTypeConvolutionalFlat(28, 28, 1))
            .build();

        return conf;
    }

    public static void main(String... args) throws Exception {
        Nd4j.getMemoryManager().setAutoGcWindow(15000);

        MnistDataSetIterator trainData = new MnistDataSetIterator(32, true, 42);
        MnistDataSetIterator testData = new MnistDataSetIterator(32, false, 42);

        boolean mlp = false;

        MultiLayerNetwork model = new MultiLayerNetwork((mlp) ? mlp() : cnn());
        model.setListeners(new PerformanceListener(50, true));

        Stopwatch stopwatch = Stopwatch.createStarted();
        //model.fit(new MultipleEpochsIterator(12, trainData));

        int epochs = (mlp) ? 20 : 12;
        for (int i = 0; i < epochs; i++) {
            model.fit(new AsyncShieldDataSetIterator(trainData));
            double trainAcc = model.evaluate(trainData).accuracy();
            double testAcc = model.evaluate(testData).accuracy();

            System.out.println("Epoch " + (i + 1) + " finished. Train ACC: " + trainAcc + " Test ACC: " + testAcc);
            if (testAcc > 0.99) {
                epochs = i;
                break;
            }
        }

        stopwatch.stop();

        System.out.println(model.evaluate(testData).stats());
        long elapsed = stopwatch.elapsed(TimeUnit.MILLISECONDS);
        long perEpoch = elapsed / epochs;
        System.out.println("Total Epochs: " + epochs);
        System.out.println("Elapsed Total: " + elapsed + " ms");
        System.out.println("Per Epoch Avg: " + perEpoch + " ms");
    }
}

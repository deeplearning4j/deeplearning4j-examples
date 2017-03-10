package org.deeplearning4j.examples.multigpu.crash;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;

/**
 * @author raver119@gmail.com
 */
public class CrashGraph {
    public static ComputationGraph getModel() {
        INDArray weights = Nd4j.ones(2);
        weights.putScalar(0, 1.0);
        weights.putScalar(1, 0);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.25)
            .updater(Updater.SGD)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .seed(119)
            .graphBuilder()
            .addInputs("input")
            .setOutputs("output1", "output2","output3", "output4")
            .addLayer("dense1", new DenseLayer.Builder().nIn(814).nOut(1024).activation(Activation.RELU).build(), "input")
            .addLayer("dense2", new DenseLayer.Builder().nIn(814).nOut(1024).activation(Activation.RELU).build(), "input")
            .addLayer("dense3", new DenseLayer.Builder().nIn(814).nOut(1024).activation(Activation.RELU).build(), "input")
            .addLayer("dense4", new DenseLayer.Builder().nIn(814).nOut(1024).activation(Activation.RELU).build(), "input")

            .addLayer("output1", new OutputLayer.Builder(new LossBinaryXENT(weights)).nIn(1024).nOut(2).activation(Activation.SOFTMAX).build(), "dense1")
            .addLayer("output2", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(1024).nOut(2).activation(Activation.SOFTMAX).build(), "dense2")
            .addLayer("output3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(1024).nOut(3).activation(Activation.SOFTMAX).build(), "dense3")
            .addLayer("output4", new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(1024).nOut(13).activation(Activation.SOFTMAX).build(), "dense4")
            .backprop(true)
            .build();

        return new ComputationGraph(conf);
    }

    public static void main(String[] args) throws Exception {
        ComputationGraph graph = getModel();
        graph.init();
        graph.setListeners(new PerformanceListener(100, true));

        ParallelWrapper wrapper = new ParallelWrapper.Builder<>(graph)
            .prefetchBuffer(6)
            .averagingFrequency(100)
            .useLegacyAveraging(false)
            .workers(2)
            .build();

        wrapper.fit(new CrashIterator());
    }
}

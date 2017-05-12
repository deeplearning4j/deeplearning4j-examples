package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SusansExample {
    public static final int RNNSIZE = 128;
    private static final int TBTTLEN = 168;
    public static final INDArray KPI_WEIGHTS = Nd4j.create(new double[]{1, 2});
    public static final INDArray KPI_WEIGHTS_ZERO = Nd4j.create(new double[]{0, 0});

    public void run(String[] args) throws Exception {
        //Nd4j.setDataType(DataBuffer.Type.DOUBLE);
/*
        ComputationGraphConfiguration myConf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.RMSPROP)
            .learningRate(0.005)
            //.rmsDecay(0.95)
            .seed(119)
            //.regularization(true)
            //.l2(0.001)
            .graphBuilder()
            .addInputs("input")
            .addLayer("first", new GravesLSTM.Builder().nIn(20).nOut(RNNSIZE)
                .activation(Activation.TANH).build(), "input")
            .addLayer("second", new GravesLSTM.Builder().nIn(RNNSIZE).nOut(RNNSIZE)
                .activation(Activation.TANH).build(), "first")
            .addLayer("outputKPI1", new RnnOutputLayer.Builder(new LossMCXENT(KPI_WEIGHTS_ZERO))
                .activation(Activation.SOFTMAX).nIn(RNNSIZE).nOut(2).build(), "second")
            .addLayer("outputKPI2", new RnnOutputLayer.Builder(new LossMCXENT(KPI_WEIGHTS_ZERO))
                .activation(Activation.SOFTMAX).nIn(RNNSIZE).nOut(2).build(), "second")
            .addLayer("outputKPI3", new RnnOutputLayer.Builder(new LossMCXENT(KPI_WEIGHTS))
                .activation(Activation.SOFTMAX).nIn(RNNSIZE).nOut(2).build(), "second")
            .addLayer("outputKPI4", new RnnOutputLayer.Builder(new LossMCXENT(KPI_WEIGHTS))
                .activation(Activation.SOFTMAX).nIn(RNNSIZE).nOut(2).build(), "second")
            .setOutputs("outputKPI1", "outputKPI2", "outputKPI3", "outputKPI4")
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(TBTTLEN).tBPTTBackwardLength(TBTTLEN)
            .pretrain(false).backprop(true)
            .build();
        myConf.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        myConf.setInferenceWorkspaceMode(WorkspaceMode.SEPARATE);

        ComputationGraph graph = new ComputationGraph(myConf);
        graph.init();


        graph.setListeners(new PerformanceListener(10, true));
*/
        AsyncMultiDataSetIterator amdsi = new AsyncMultiDataSetIterator(new SusansIterator(1000, 32), 2, true);
        while (amdsi.hasNext()) {
            MultiDataSet mds = amdsi.next();

            log.info("Feeding next dataset...");
            Thread.sleep(1000);
            //graph.fit(mds);
        }
    }

    public static void main(String[] args) throws Exception {
        new SusansExample().run(args);
    }
}

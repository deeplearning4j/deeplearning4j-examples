package org.deeplearning4j.ganexamples;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author zdl
 */
public class SimpleGan {
    static double lr = 0.005;
    private static final Logger log = LoggerFactory.getLogger(SimpleGan.class);
    private static String[] generatorLayerNames = new String[]{"g1", "g2", "g3"};
    private static String[] discriminatorLayerNames = new String[]{"d1", "d2", "d3", "out"};

    public static void main(String[] args) throws Exception {

        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new RmsProp(lr))
            .weightInit(WeightInit.XAVIER);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder().backpropType(BackpropType.Standard)
            .addInputs("input1", "input2")
            .addLayer("g1",
                new DenseLayer.Builder().nIn(10).nOut(128).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "input1")
            .addLayer("g2",
                new DenseLayer.Builder().nIn(128).nOut(512).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "g1")
            .addLayer("g3",
                new DenseLayer.Builder().nIn(512).nOut(28 * 28).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "g2")
            .addVertex("stack", new StackVertex(), "input2", "g3")
            .addLayer("d1",
                new DenseLayer.Builder().nIn(28 * 28).nOut(256).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "stack")
            .addLayer("d2",
                new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "d1")
            .addLayer("d3",
                new DenseLayer.Builder().nIn(128).nOut(128).activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER).build(),
                "d2")
            .addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(128).nOut(1)
                .activation(Activation.SIGMOID).build(), "d3")
            .setOutputs("out");

        ComputationGraph net = new ComputationGraph(graphBuilder.build());
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        System.out.println(net.summary());
        DataSetIterator train = new MnistDataSetIterator(30, true, 12345);
        INDArray discriminatorLabel = Nd4j.vstack(Nd4j.ones(30, 1), Nd4j.zeros(30, 1));
        INDArray generatorLabel = Nd4j.ones(60, 1);
        MNISTVisualizer bestVisualizer = new MNISTVisualizer(1, "Gan");
        for (int i = 1; i <= 100000; i++) {
            if (!train.hasNext()) {
                train.reset();
            }
            INDArray trueData = train.next().getFeatures();
            INDArray z = Nd4j.rand(new NormalDistribution(), new long[]{30, 10});
            MultiDataSet dataSetD = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{z, trueData},
                new INDArray[]{discriminatorLabel});
            trainDiscriminator(net, dataSetD);
            z = Nd4j.rand(new NormalDistribution(), new long[]{30, 10});
            MultiDataSet dataSetG = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{z, trueData},
                new INDArray[]{generatorLabel});
            trainGenerator(net, dataSetG);
            if (i % 100 == 0) {
                DataSetIterator dataSetIterator = new MnistDataSetIterator(30, true, 12345);
                INDArray data = dataSetIterator.next().getFeatures();
                Map<String, INDArray> map = net.feedForward(
                    new INDArray[]{Nd4j.rand(new NormalDistribution(), new long[]{50, 10}), data}, false);
                INDArray indArray = map.get("g3");

                List<INDArray> list = new ArrayList<>();
                for (int j = 0; j < indArray.size(0); j++) {
                    list.add(indArray.getRow(j));
                }
                bestVisualizer.setDigits(list);
                bestVisualizer.visualize();
            }
        }

    }

    public static void trainDiscriminator(ComputationGraph net, MultiDataSet dataSet) {
        net.setTrainable(discriminatorLayerNames, true);
        net.setTrainable(generatorLayerNames, false);
        net.fit(dataSet);
    }

    public static void trainGenerator(ComputationGraph net, MultiDataSet dataSet) {
        net.setTrainable(discriminatorLayerNames, false);
        net.setTrainable(generatorLayerNames, true);
        net.fit(dataSet);
    }
}

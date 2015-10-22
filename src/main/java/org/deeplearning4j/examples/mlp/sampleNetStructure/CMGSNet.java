package org.deeplearning4j.examples.mlp.sampleNetStructure;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * Reference: http://arxiv.org/pdf/1003.0358v1.pdf
 * Created by nyghtowl on 9/21/15.
 */
public class CMGSNet {

    // TODO finish reviewing and pulling in paper details
    private int height;
    private int width;
    private int outputNum;
    private long seed;
    private int iterations;

    public CMGSNet(int height, int width, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {

        // TODO for mnist example expand training data like Simard et al
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-3f) // TODO create learnable lr that shrinks by multiplicative constant after each epoch pg 3
                .momentum(0)
                .list(6)
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(2500)
                        .activation("tanh") // TODO set A = 1.7159 and B = 0.6666
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(2500)
                        .nOut(2000)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(1500)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(1500)
                        .nOut(1000)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(500)
                        .activation("tanh")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(500)
                        .nOut(outputNum)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.5, 0.5))
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

}

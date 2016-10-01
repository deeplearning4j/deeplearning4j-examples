package org.deeplearning4j.examples.convolution.debug;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author raver119@gmail.com
 */
public class Cuda_CNN {

    public static void main(String[] args) throws Exception {
   //     CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);
        //DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        int height = 360;
        int width = 640;
        int featureCount = 100;
        int batchSize = 64;

        DataSetIterator iterator = new MnistDataSetIterator(batchSize,true,12345);

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(98)
            .iterations(1)
            .regularization(true)
            .l2(0.0005)
            .learningRate(0.008)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder()
                .name("layer0")
                .kernelSize(4, 4)
                .nIn(3)
                .stride(2, 2)
                .nOut(20)
                .padding(1, 1)
                .dropOut(0.5)
                .activation("relu")
                .build())
            .layer(1,
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .name("layer1")
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(2, new ConvolutionLayer.Builder()
                .name("layer2")
                .kernelSize(3, 3)
                .nIn(20)
                .stride(1, 1)
                .nOut(20)
                .padding(1,1)
                .dropOut(0.5)
                .activation("relu")
                .build())
            .layer(3,
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .name("layer3")
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(4, new DenseLayer.Builder()
                .name("layer4")
                .activation("relu")
                .nOut(200)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                .name("layer5")
                .nOut(featureCount)
                .activation("identity")
                .build())
            .backprop(true).pretrain(false)
            .setInputType(InputType.convolutionalFlat(height, width, 3));


        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new PerformanceListener(1, true));

        for(int i = 0; i < 10; i++){
            model.fit(new DummyIterator(new int[]{64, 360*640*3}, new int[]{64, featureCount}));
        }
    }
}

package org.deeplearning4j.examples.samediff.training;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.List;

/**
 * This example shows the creation and training of a MNIST CNN network.
 */
public class SameDiffMNISTTrainingExample {

    static SameDiff makeMNISTNet(){
        SameDiff sd = SameDiff.create();

        //Properties for MNIST dataset:
        int nIn = 28*28;
        int nOut = 10;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

        SDVariable reshaped = in.reshape(-1, 1, 28, 28);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        // layer 1: Conv2D with a 3x3 kernel and 4 output channels
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
        SDVariable b0 = sd.zero("b0", 4);

        SDVariable conv1 = sd.cnn().conv2d(reshaped, w0, b0, convConfig);

        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);

        // layer 3: Conv2D with a 3x3 kernel and 8 output channels
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
        SDVariable b1 = sd.zero("b1", 8);

        SDVariable conv2 = sd.cnn().conv2d(relu1, w1, b1, convConfig);

        // layer 4: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        SDVariable relu2 = sd.nn().relu(pool2, 0);

        SDVariable flat = relu2.reshape(-1, 5 * 5 * 8);

        // layer 5: Output layer on flattened input
        SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
        SDVariable bOut = sd.zero("bOut", 10);

        SDVariable z = sd.nn().linear("z", flat, wOut, bOut);

        // softmax crossentropy loss function
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, z);

        //noinspection unused
        SDVariable out = sd.nn().softmax("out", z, 1);

        sd.setLossVariables(loss);

        return sd;
    }

    public static void main(String[] args) throws Exception {
        SameDiff sd = makeMNISTNet();

        //Create and set the training configuration

        Evaluation evaluation = new Evaluation();

        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
            .l2(1e-4)                               //L2 regularization
            .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
            .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
            .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
            .trainEvaluation("out", 0, evaluation)  // add a training evaluation
            .build();

        // You can add validation evaluations as well, but they have some issues in beta5 and most likely won't work.
        // If you want to use them, use the SNAPSHOT build.

        sd.setTrainingConfig(config);

        // Adding a listener to the SameDiff instance is necessary because of a beta5 bug, and is not necessary in snapshots
        sd.addListeners(new ScoreListener(20));

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        //Perform training for 4 epochs
        int numEpochs = 4;
        History hist = sd.fit()
            .train(trainData, numEpochs)
            .exec();
        List<Double> acc = hist.trainingEval(Metric.ACCURACY);

        System.out.println("Accuracy: " + acc);
    }
}

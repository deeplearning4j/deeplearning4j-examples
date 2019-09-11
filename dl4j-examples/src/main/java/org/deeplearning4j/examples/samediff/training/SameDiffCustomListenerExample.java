package org.deeplearning4j.examples.samediff.training;

import static org.deeplearning4j.examples.samediff.training.SameDiffMNISTTrainingExample.makeMNISTNet;

import java.util.Arrays;
import java.util.List;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerVariables;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 * This example shows how to use a custom listener, and is based on the {@link SameDiffMNISTTrainingExample}.
 */
public class SameDiffCustomListenerExample {

    /**
     * A basic custom listener that records the values of z and out, for comparison later
     */
    public static class CustomListener extends BaseListener {

        public INDArray z;
        public INDArray out;

        // Specify that this listener is active during inference operations
        @Override
        public boolean isActive(Operation operation) {
            return operation == Operation.INFERENCE;
        }

        // Specify that this listener requires the activations of "z" and "out"
        @Override
        public ListenerVariables requiredVariables(SameDiff sd) {
            return new ListenerVariables.Builder().inferenceVariables("z", "out").build();
        }

        // Called when the activation of a variable becomes available
        @Override
        public void activationAvailable(SameDiff sd, At at,
            MultiDataSet batch, SameDiffOp op,
            String varName, INDArray activation) {

            // if the variable is z or out, store its activation
            if(varName.equals("out")){
                out = activation.detach().dup();
            } else if(varName.equals("z")){
                z = activation.detach().dup();
            }
        }
    }


    public static void main(String[] args) throws Exception {
        SameDiff sd = makeMNISTNet();

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
            .l2(1e-4)                               //L2 regularization
            .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
            .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
            .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
            .build();

        sd.setTrainingConfig(config);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        //Perform training
        History hist = sd.fit()
            .train(trainData, 4)
            .exec();
        List<Double> acc = hist.trainingEval(Metric.ACCURACY);

        System.out.println("Accuracy: " + acc);

        CustomListener listener = new CustomListener();

        sd.output()
            .data(new MnistDataSetIterator(10, 10, false, false, true, 12345))
            .output("out")
            .listeners(listener)
            .exec();

        System.out.println("Z: " + listener.z);
        System.out.println("Out (softmax(z)): " + listener.out);
    }
}

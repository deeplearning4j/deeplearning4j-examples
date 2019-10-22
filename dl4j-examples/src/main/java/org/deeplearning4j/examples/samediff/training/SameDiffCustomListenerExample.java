package org.deeplearning4j.examples.samediff.training;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.samediff.tfimport.SameDiffTransferLearningExample;
import org.nd4j.autodiff.listeners.*;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;

import java.util.List;

import static org.deeplearning4j.examples.samediff.training.SameDiffMNISTTrainingExample.makeMNISTNet;

/**
 * This example shows how to use a custom listener, and is based on the {@link SameDiffMNISTTrainingExample}.<br><br>
 *
 * We use a basic custom listener that records the values of 2 variables, for comparison or printing later.
 *
 * Another example is a listener that prints the shapes of activations, used in {@link SameDiffTransferLearningExample}.
 *
 * For more details on what you can do with listeners, and the methods you can implement, look at {@link BaseListener} and {@link Listener}.<br>
 * If you want to use evaluations in your listener, look at {@link BaseEvaluationListener}.
 *
 */
@SuppressWarnings("DuplicatedCode")
public class SameDiffCustomListenerExample {

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

    /**
     * A basic custom listener that records the values of z and out, for comparison or printing later.
     */
    public static class CustomListener extends BaseListener {

        INDArray z;
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
}

package org.deeplearning4j.examples.recurrent.seq2seq;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by susaneraly on 1/11/17.
 */
public class Seq2SeqPredicter {

    private ComputationGraph net;
    private INDArray decoderInputTemplate = null;

    public Seq2SeqPredicter(ComputationGraph net) {
        this.net = net;
    }

    /*
        Given an input to the computation graph (which is expected to a be a seq2seq model)
        Predict the output given the encoder input (which is fixed) + the first time step from the decoder input
        All other time steps in the decoder input will be ignored
        //FIX ME, comments
     */

    public INDArray output(MultiDataSet testSet) {
        if (testSet.getFeatures()[0].size(0) > 2) {
            return output(testSet, false);
        } else {
            return output(testSet, true);
        }

    }

    public INDArray output(MultiDataSet testSet, boolean print) {

        INDArray correctOutput = testSet.getLabels()[0];
        INDArray ret = Nd4j.zeros(correctOutput.shape());
        copyTimeSteps(0,testSet.getFeatures()[1],decoderInputTemplate);

        int currentStepThrough = 0;
        int stepThroughs = correctOutput.size(2);

        while (currentStepThrough < stepThroughs) {
            ret = stepOnce(testSet, currentStepThrough);
            if (print) {
                System.out.println("In time step "+currentStepThrough);
                System.out.println("\tEncoder input and Decoder input:");
                CustomSequenceIterator.mapToString(testSet.getFeatures()[0],ret, " +  ");
                System.out.println("\tEncoder input and Decoder output:");
                CustomSequenceIterator.mapToString(testSet.getFeatures()[0],ret);
            }
            currentStepThrough++;
        }

        return ret;
    }

    /*
        Will do a forward pass through encoder + decoder with the given input
        Updates the decoder input template from time = 1 to time t=n+1;
        Returns the output from this forward pass
     */
    private INDArray stepOnce(MultiDataSet testSet, int n) {

        INDArray currentOutput = net.output(false, testSet.getFeatures()[0], decoderInputTemplate)[0];
        copyTimeSteps(n+1,currentOutput,decoderInputTemplate);
        return currentOutput;

    }

    /*
        Copies timesteps
        time = 0 to time = t in "fromArr" to time = 1 to time = t+1 in "toArr"
     */
    private void copyTimeSteps(int t, INDArray fromArr, INDArray toArr) {
        if (toArr == null) {
            toArr = fromArr.dup();
            return;
        }
        INDArray fromView = fromArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,t,true));
        INDArray toView = toArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,t+1,true));
        toView.assign(fromView);
    }

}

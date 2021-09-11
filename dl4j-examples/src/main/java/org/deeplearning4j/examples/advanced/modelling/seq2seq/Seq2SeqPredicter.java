/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.seq2seq;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by susaneraly on 1/11/17.
 */
/*
    Note this is a helper class with methods to step through the decoder, one time step at a time.
    This process is common to all seq2seq models and will eventually be wrapped in a class in dl4j (along with an easier API).
    Track issue:
        https://github.com/eclipse/deeplearning4j/issues/2635
 */
public class Seq2SeqPredicter {

    private ComputationGraph net;
    private INDArray decoderInputTemplate = null;

    Seq2SeqPredicter(ComputationGraph net) {
        this.net = net;
    }

    /*
        Given an input to the computation graph (which is expected to a be a seq2seq model)
        Predict the output given the encoder input (which is fixed) + the first time step from the decoder input
        All other time steps in the decoder input will be ignored
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
        INDArray ret;
        decoderInputTemplate = testSet.getFeatures()[1].dup();

        int currentStepThrough = 0;
        int stepThroughs = (int)correctOutput.size(2)-1;

        while (currentStepThrough < stepThroughs) {
            if (print) {
                System.out.println("In time step "+currentStepThrough);
                System.out.println("\tEncoder input and Decoder input:");
                System.out.println(CustomSequenceIterator.mapToString(testSet.getFeatures()[0],decoderInputTemplate));

            }
            ret = stepOnce(testSet, currentStepThrough);
            if (print) {
                System.out.println("\tDecoder output:");
                System.out.println("\t"+String.join("\n\t",CustomSequenceIterator.oneHotDecode(ret)));
            }
            currentStepThrough++;
        }

        ret = net.output(false,testSet.getFeatures()[0],decoderInputTemplate)[0];
        if (print) {
            System.out.println("Final time step "+currentStepThrough);
            System.out.println("\tEncoder input and Decoder input:");
            System.out.println(CustomSequenceIterator.mapToString(testSet.getFeatures()[0],decoderInputTemplate));
            System.out.println("\tDecoder output:");
            System.out.println("\t"+String.join("\n\t",CustomSequenceIterator.oneHotDecode(ret)));
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
        copyTimeSteps(n,currentOutput,decoderInputTemplate);
        return currentOutput;

    }

    /*
        Copies timesteps
        time = 0 to time = t in "fromArr"
        to time = 1 to time = t+1 in "toArr"
     */
    private void copyTimeSteps(int t, INDArray fromArr, INDArray toArr) {
        INDArray fromView = fromArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,t,true));
        INDArray toView = toArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,t+1,true));
        toView.assign(fromView.dup());
    }

}

package org.deeplearning4j.examples.samediff.dl4j.layers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class MergeLambdaVertex extends SameDiffLambdaVertex {
    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        //2 inputs to the vertex. The VertexInputs class will dynamically add as many variables as we request from it!
        SDVariable input1 = inputs.getInput(0);
        SDVariable input2 = inputs.getInput(1);
        SDVariable average = sameDiff.math().mergeAvg(input1, input2);
        return average;
    }
}

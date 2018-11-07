package org.deeplearning4j.examples.samediff.dl4j.layers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Lambda layers don't have any parameters. They are useful for simple operations
 *
 * This particular lambda layer implements the following:
 * out = in / l2Norm(in), on a per-example basis
 *
 * @author Alex Black
 */
public class L2NormalizeLambdaLayer extends SameDiffLambdaLayer {

    private int[] dimensions;

    /**
     *
     * @param dimensions Dimensions to calculate L2 norm over.
     *                   For DenseLayer/FeedForward input, this would be dimension 1
     *                   For RNNs, this would also be dimension 1 (to normalize each time step separately)
     *                   For CNNs, this would be dimensions 1, 2 and 3
     */
    public L2NormalizeLambdaLayer(int... dimensions){
        this.dimensions = dimensions;
    }

    private L2NormalizeLambdaLayer(){
        //Add a private no-arg constructor for use in JSON deserialization
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
        //Note the 'keepdims' arg: this will keep the dimensions, so we can auto-broadcast the later division
        //For example, if input is shape [3,4,5,6] and dimensions are [1,2,3] (i.e., for CNN activations)
        //then Norm2 has shape [3,1,1,1] - or if keepDims=false was used, it would have shape [3]
        SDVariable norm2 = layerInput.norm2(true, dimensions);
        return layerInput.div(norm2);
    }

    //Getters and setters for JSON serialization
    public int[] getDimensions(){
        return dimensions;
    }

    public void setDimensions(int[] dimensions){
        this.dimensions = dimensions;
    }
}

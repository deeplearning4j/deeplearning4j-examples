# SameDiff - Custom Layers

The examples in this directory show how to implement Deeplearning4j layers using SameDiff.

SameDiff is an automatic differentiation package build on top of ND4J. It can be considered analogous to 
libraries such as TensorFlow and PyTorch, in that a user can define a set of tensor operations, and   

DL4J has supported custom layers for a long time - however, using SameDiff layers has some advantages:
1. Only the forward pass has to be defined: i.e., you don't need to manually work out gradient calculation
2. The layer definition is simpler: fewer methods to implement (and only one class)

## Available Layer Types

5 types of SameDiff layers are available: 
1. **Layers**: standard single input, single output layers defined using SameDiff. To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer```
2. **Lambda layers**: as above, but without any parameters. You only need to implement a single method for these! To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer```
3. **Graph vertices**: multiple inputs, single output layers usable only in ComputationGraph. To implement: extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex```
4. **Lambda vertices**: as above, but without any parameters. Again, you only need to implement a single method for these! To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex```
5. **Output layers**: An output layer, for calculating scores/losses. Used as the final layer in a network. To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer```

## Restrictions

Currently, most ops in SameDiff execute on CPU only - GPU support for all ops is in the process of being implemented
and will be available in a future release.

Given DL4J API restrictions, currently no multi-output layers/vertices are supported. However, some workarounds are available:
1. Implement one vertex for each output required
2. Concatenate the activations, then split them later (using SubsetVertex, or as part of another DL4J SameDiff layer)

## The Examples

1. **Example 1**: A minimal feed forward layer example - equivalent to DL4J's DenseLayer
2. **Example 2**: A example of a SameDiff Lambda layer (layer without any parameters) - implements ```out = in / l2Norm(in)```
5. **Example 3**: An example of implementing a Lambda vertex (no parameters, only 1 method to implement). Implements ```out=avg(in1, in2)```

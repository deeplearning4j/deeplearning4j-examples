# SameDiff - Custom Layers

The examples in this directory/package show how to implement layers using SameDiff.

What is SameDiff? An automatic differentiation package build on top of ND4J. It can be considered analogous to tools
such as TensorFlow and PyTorch.  

DL4J has supported custom layers for a long time - however, using SameDiff layers has some advantages:
1. Only the forward pass has to be defined: i.e., you don't need to manually work out gradient calculation
2. The layer definition is simpler: fewer methods to implement

## Available Layer Types

4 types of SameDiff layers are available: 
1. Layers: standard single input, single output layers defined using SameDiff. To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer```
2. Lambda layers: as above, but without any parameters. You only need to implement a single method for these! To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer```
3. Graph vertices: multiple inputs, single output layers usable only in ComputationGraph. To implement: extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex```
4. Lambda vertices: as above, but without any parameters. Again, you only need to implement a single method for these! To implement, extend ```org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex```

## Restrictions

Currently, most ops in SameDiff execute on CPU only - GPU support for all ops is currently in-progress.

Given DL4J API restrictions, currently no multi-output layers/vertices are supported. However, some workarounds are available:
1. Implement one vertex for each output required
2. Concatenate the activations, then split them later (using SubsetVertex, or as part of another DL4J SameDiff layer)

## The Examples

1. **Example 1**: A minimal feed forward layer example - equivalent to DL4J's DenseLayer
2. **Example 2**: A example of a SameDiff Lambda layer (layer without any parameters) - implements ```out = in / l2Norm(in)```
3. **Example 3**: An example of a convolutional layer, with extra features (builder, DL4J global configuration inheritance, etc)
4. **Example 4**: An example of implementing a Graph vertex with trainable parameters
5. **Example 5**: An example of implementing a Lambda vertex (no parameters, only 1 method to implement). Implements ```out=avg(in1, in2)```

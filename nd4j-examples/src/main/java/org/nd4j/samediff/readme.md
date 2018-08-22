# SameDiff Examples

SameDiff is an automatic differentiation library built on top of ND4J.
It can be considered comparable to TensorFlow or PyTorch, in that users can define a set of tensor operations
(a graph of operations, defining the "forward pass") and SameDiff will automatically differentiate the graph.

Note that as of 1.0.0-beta2, SameDiff has some limitations that will be fixed in the next few releases:
1. Many operations in SameDiff are limited to CPU-only exeuction. GPU (CUDA) execution will be added in the future
2. Standalone graph training (outside of the existing DL4J SameDiff layer training) not yet been added

Note also that a number of Deeplearning4j examples demonstrate how to use SameDiff to create DL4J layers and
vertices as part of a Deeplearning4j MultiLayerNetwork or ComputationGraph.
These examples can be found at:
[https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff) 

SameDiff also supports importing TensorFlow models to a SameDiff graph.
Note that this functionality is still being built out - some operations are not yet available.
The TensorFlow import examples can be found here:
[https://github.com/deeplearning4j/dl4j-examples/tree/master/tf-import-examples/src/main/java/org/nd4j/examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/tf-import-examples/src/main/java/org/nd4j/examples)

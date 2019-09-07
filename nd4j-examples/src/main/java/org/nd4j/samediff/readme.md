# SameDiff Examples

SameDiff is an automatic differentiation library built on top of ND4J.
It can be considered comparable to TensorFlow or PyTorch, in that users can define a set of tensor operations
(a graph of operations, defining the "forward pass") and SameDiff will automatically differentiate the graph.

Note that as of 1.0.0-beta4, SameDiff has a limitation that will be fixed in the next few releases:
Many operations in SameDiff are limited to CPU-only exeuction. GPU (CUDA) execution will be added in the future

Note also that a number of Deeplearning4j examples demonstrate how to use SameDiff to create DL4J layers and
vertices as part of a Deeplearning4j MultiLayerNetwork or ComputationGraph.
These examples can be found under [samediff](../../../../../../../dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff)

SameDiff also supports importing TensorFlow models to a SameDiff graph.
Note that this functionality is still being built out - some operations are not yet available.
The TensorFlow import examples can be found under [tf-import-examples](../../../../../../../tf-import-examples)

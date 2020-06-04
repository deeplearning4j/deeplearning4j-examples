## Eclipse Deeplearning4j: ND4J NDArray Examples

This project contains a set of examples that demonstrate how to manipulate NDArrays. The functionality of ND4J demonstrated here can be likened to NumPy.

The examples in this project along with a short summary are listed below. This is also the recommended order to explore them in.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

## Quickstart
* [Nd4jEx0_INDArrayBasics.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx0_INDArrayBasics.java)
Creating, concatenating, stacking, padding NDArrays
* [Nd4jEx1_INDArrayBasics.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx1_INDArrayBasics.java)
Printing, querying shape properties, modifying elements, doing operations
* [Nd4jEx2_CreatingINDArrays.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx2_CreatingINDArrays.java)
Different ways to create NDArrays
* [Nd4jEx3_GettingAndSettingSubsets.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx3_GettingAndSettingSubsets.java)
Querying and modifying subset of an NDArray
* [Nd4jEx4_Ops.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx4_Ops.java)
More ops on NDArrays
* [Nd4jEx5_Accumulations.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx5_Accumulations.java)
Reduction Operations
* [Nd4jEx6_BooleanIndexing.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx6_BooleanIndexing.java)
Simple conditional element wise operations
* [Nd4jEx7_MatrixOperation.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx7_MatrixOperation.java)
Matrix multiplication
* [Nd4jEx8_ReshapeOperation.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx8_ReshapeOperation.java)
Reshaping
* [Nd4jEx9_Functions.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx9_Functions.java)
Transforming NDArrays
* [Nd4jEx10_ElementWiseOperation.java](./src/main/java/org/nd4j/examples/quickstart/Nd4jEx10_ElementWiseOperation.java)
Even more operations like add row/col etc
* [NumpyCheatSheet.java](NumpyCheatSheet.java) (FIXME???)
Examples to help NumPy users get acquainted with ND4J

## Advanced
* [MultiClassLogitExample.java](./src/main/java/org/nd4j/examples/advanced/lowlevelmodeling/MultiClassLogitExample.java)
Multiclass logistic regression from scratch with ND4J
* [WorkspacesExample.java](./src/main/java/org/nd4j/examples/advanced/memoryoptimization/WorkspacesExample.java)
For cyclic workloads like training a neural net the DL4J ecosystem does not rely on garbage collection. Instead a chunk of memory is resued avoiding the performance hits from expensive pauses for GC. Workspaces are used by default when calling .fit on a neural network etc. This example demonstrates the concepts behind it for advanced users if they need to go beyond what is available by default in the library in their particular use case.
* [Nd4jEx11_Axpy.java](./src/main/java/org/nd4j/examples/advanced/operations/Nd4jEx11_Axpy.java)
Use the ND4J blas wrapper to call the AXPY operation
* [Nd4jEx12_LargeMatrices.java](./src/main/java/org/nd4j/examples/advanced/operations/Nd4jEx12_LargeMatrices.java)
Operations with a 10000000 element NDarray and its transpose
* [Nd4jEx13_Serialization.java](./src/main/java/org/nd4j/examples/advanced/operations/Nd4jEx13_Serialization.java)
Examples for binary and text serialization.
* [Nd4jEx14_Normalizers.java](./src/main/java/org/nd4j/examples/advanced/operations/Nd4jEx14_Normalizers.java)
Create and fit a normalizer, and save and restore it.
* [CustomOpsExamples.java](./src/main/java/org/nd4j/examples/advanced/operations/CustomOpsExamples.java)
**Only relevant to the 1.0.0-beta6 release**. There are some operations that were implemented in C++ that had not been mapped to Java. This example demonstrates how to access them using ND4J's DynamicCustomOp. As of the beta7 release all maps have corresponding Java mappings.


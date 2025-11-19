

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

# ND4J NDArray Examples

This module contains simple examples that demonstrate how to use **ND4J**  
(N-dimensional arrays for Java) — the core numerical computing library behind Deeplearning4j.

ND4J provides fast tensor operations similar to NumPy, backed by optimized BLAS  
implementations.

These examples help new users understand:

- What an `INDArray` is  
- How to create arrays  
- How to perform basic math  
- How to reshape, slice, and combine arrays  
- How to run the examples using Maven

---

## 📌 What is an NDArray?

An **NDArray** (N-dimensional array) is ND4J’s fundamental data structure.  
It represents:

- Scalars (0D)
- Vectors (1D)
- Matrices (2D)
- Higher dimensional tensors (3D, 4D, …)

ND4J operations are *vectorized* and run efficiently on CPU or GPU (via CUDA backend).

Example:

```java
INDArray arr = Nd4j.create(new float[]{1, 2, 3});
System.out.println(arr);   // [1.00, 2.00, 3.00]
📘 Basic NDArray Operations
Create Arrays
java

INDArray zeros = Nd4j.zeros(3, 3);
INDArray ones = Nd4j.ones(2, 2);
INDArray random = Nd4j.rand(1, 5);
INDArray arange = Nd4j.arange(1, 10);
Math Operations
java

INDArray a = Nd4j.create(new float[]{1, 2, 3});
INDArray b = Nd4j.create(new float[]{4, 5, 6});

System.out.println(a.add(b));       // elementwise add  
System.out.println(a.mul(b));       // elementwise multiply
System.out.println(a.mmul(b.T()));  // matrix multiplication
Reshape and Shape Ops

INDArray m = Nd4j.linspace(1, 9, 9).reshape(3, 3);
System.out.println(m);

System.out.println(m.transpose());
System.out.println(m.permute(1, 0));
▶️ How to Run These Examples
Make sure you have:

Java 8 or later

Maven installed

Then run:


mvn -q exec:java -Dexec.mainClass="org.nd4j.examples.quickstart.BasicNDArrayExample"
Or run any other example under:



src/main/java/org/nd4j/examples/
Example:


mvn -q exec:java -Dexec.mainClass="org.nd4j.examples.advanced.SlicingExample"
📤 Expected Output (Sample)
Running a basic NDArray creation example:


Copy code
Zeros array:
[[0.00, 0.00, 0.00],
 [0.00, 0.00, 0.00],
 [0.00, 0.00, 0.00]]

Random array:
[0.42, 0.91, 0.12, 0.55, 0.33]

Vector add:
[5.00, 7.00, 9.00]
📁 Project Structure

nd4j-ndarray-examples/
 ├── src/
 │   └── main/java/org/nd4j/examples/
 │        ├── advanced/
 │        ├── quickstart/
 │        ├── utils/
 │        └── resources/
 ├── pom.xml
 └── README.md
📄 License
This project is part of the Eclipse Deeplearning4j examples collection
and is licensed under the Apache License 2.0.

Happy coding with ND4J tensors! 🚀


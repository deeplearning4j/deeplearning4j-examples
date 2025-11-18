ONNX Import Examples (DeepLearning4J)

This module contains example programs demonstrating how to import ONNX models into DeepLearning4J (DL4J) and run inference using ND4J.

These examples help users understand the workflow required for loading pretrained ONNX models, preprocessing inputs, performing forward passes, and reading predictions.

🚀 What You Will Learn

How to load ONNX models using DL4J’s OnnxGraphImporter

How to inspect ONNX graph metadata

How to prepare NDArray input tensors for inference

Running inference on imported ONNX models

Reading and interpreting output layers

📦 How to Run

Use Maven to compile and run any example:

mvn clean compile exec:java -Dexec.mainClass="org.deeplearning4j.examples.onnx.<ExampleClassName>"


Example:

mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.onnx.ImportBasicOnnxModel"


Replace <ExampleClassName> with any class inside the onnx-import-examples folder.

📁 Model Requirements

To run inference, you need an .onnx model file.

If the example references a model that is not included in the repository, download it from:

https://github.com/onnx/models

Or any ONNX-compatible export from PyTorch/Keras/TF

Place it in the example’s resources/ directory or update the file path in the code.

🧩 Folder Structure
onnx-import-examples/
 ├── src/main/java/org/deeplearning4j/examples/onnx/
 │     ├── ImportBasicOnnxModel.java
 │     ├── InspectOnnxGraph.java
 │     └── ...
 ├── src/main/resources/
 ├── pom.xml
 └── README.md   ← (This file)

🧪 Expected Output

Typical output may include:

Loading ONNX model...
Model imported successfully.
Running inference...
Output shape: [1, 1000]
Predicted class: 281 (tabby cat)

❤️ Contribution

Feel free to add additional examples demonstrating:

ONNX opset compatibility

Image preprocessing pipelines

Importing models trained in TensorFlow / PyTorch
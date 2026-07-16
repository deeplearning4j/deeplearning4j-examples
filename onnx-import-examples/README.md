## Eclipse Deeplearning4j: ONNX Import & OmniHub Examples

This project contains examples that demonstrate ONNX model import via SameDiff, and the OmniHub model hub for loading pretrained models from the DL4J zoo and HuggingFace Hub in multiple formats (GGUF, SafeTensors, TorchScript).

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

---

### ONNX Import

* [OnnxImportLoad.java](./src/main/java/org/deeplearning4j/modelimportexamples/onnx/OnnxImportLoad.java)
Import an ONNX model into a SameDiff computation graph
* [OnnxImportSave.java](./src/main/java/org/deeplearning4j/modelimportexamples/onnx/OnnxImportSave.java)
Import an ONNX model and save it in DL4J's native format
* [ImageProcessUtils.java](./src/main/java/org/deeplearning4j/modelimportexamples/onnx/ImageProcessUtils.java)
Image preprocessing utilities for running inference on imported models

### OmniHub & Multi-Format Model Import (NEW)

OmniHub is the DL4J model zoo system that provides easy access to pretrained models. It supports loading models from the DL4J model zoo (hosted pretrained models in DL4J and SameDiff formats) and HuggingFace Hub (GGUF, SafeTensors formats auto-detected). Models are cached locally in the omnihub home directory (configurable via `OMNIHUB_HOME` environment variable, defaults to `~/.omnihub/`).

* [OmniHubPretrainedModels.java](./src/main/java/org/deeplearning4j/modelimportexamples/omnihub/OmniHubPretrainedModels.java)
Load pretrained models from the DL4J zoo (`Pretrained.dl4j()`, `Pretrained.samediff()`) and HuggingFace Hub (`OmniHubUtils.loadFromHuggingFace()`)
* [HuggingFaceGGUFImport.java](./src/main/java/org/deeplearning4j/modelimportexamples/omnihub/HuggingFaceGGUFImport.java)
Download and import GGUF models from HuggingFace Hub using the AutoModel pipeline with auto-format detection
* [GGMLModelImportExample.java](./src/main/java/org/deeplearning4j/modelimportexamples/omnihub/GGMLModelImportExample.java)
Low-level GGUF/GGML import and export API (nd4j-ggml module) -- format detection, direct GGUF I/O, quantization
* [SafeTensorsImportExample.java](./src/main/java/org/deeplearning4j/modelimportexamples/omnihub/SafeTensorsImportExample.java)
SafeTensors format import -- HuggingFace's recommended format for model weights with zero-copy deserialization
* [TorchScriptImportExample.java](./src/main/java/org/deeplearning4j/modelimportexamples/omnihub/TorchScriptImportExample.java)
TorchScript (.pt) model import into SameDiff computation graphs

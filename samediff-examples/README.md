## Eclipse Deeplearning4j: SameDiff Examples

This project contains examples that demonstrate the SameDiff API, LLM text generation, vision-language models, audio processing, and advanced training techniques.

SameDiff is the automatic differentiation / deep learning framework within the ND4J library. It uses a graph-based (define then run) approach, similar to TensorFlow graph mode. SameDiff supports importing TensorFlow frozen .pb models and ONNX models. DL4J also has full SameDiff support for writing custom layers and loss functions.

**New in 1.0.0:** SameDiff now includes LLM, VLM, and audio pipeline modules, GGML/GGUF model support, and advanced training APIs (LoRA, PEFT, knowledge distillation, mixed precision).

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

---

### Basics
* [Ex1_SameDiff_Basics.java](./src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex1_SameDiff_Basics.java)
SameDiff class, variables, functions and forward pass
* [Ex2_LinearRegression.java](./src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex2_LinearRegression.java)
Placeholders, forward pass and gradient calculations on a simple linear regression graph
* [Ex3_Variables.java](./src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex3_Variables.java)
Alternate ways to create variables

### Modeling
* [MNISTFeedforward.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/MNISTFeedforward.java)
Create, train, evaluate, save and load a basic feedforward network using SameDiff.
* [MNISTCNN.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/MNISTCNN.java)
The same as the above but with a CNN network
* [CustomListenerExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/CustomListenerExample.java)
Implementing a basic custom listener that records variable values during training.
* [GraphOptimizerExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/GraphOptimizerExample.java)
SameDiff graph optimization passes

### LLM / Text Generation (NEW)
* [QwenTextGenerationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/QwenTextGenerationExample.java)
Full Qwen LLM pipeline: download GGUF from HuggingFace, import into SameDiff, tokenize with HuggingFace tokenizer, generate text with sampling strategies and chat templates
* [GGMLImportExportExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/GGMLImportExportExample.java)
GGML/GGUF format detection, model import, export, low-level GGUF I/O, quantization/dequantization

### Vision-Language Models (NEW)
* [SmolDoclingVLMExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/SmolDoclingVLMExample.java)
SmolDocling 256M VLM for document understanding -- OCR, table extraction, markdown conversion from scanned pages/PDFs
* [VideoVLMExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/VideoVLMExample.java)
Video VLM preprocessing pipeline -- frame extraction, temporal sampling, video-to-text

### Audio (NEW)
* [WhisperSpeechToTextExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/WhisperSpeechToTextExample.java)
OpenAI Whisper speech-to-text: model download, transcription, mel spectrogram extraction, audio preprocessing, tokenizer inspection
* [TtsTrainingPipelineExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/TtsTrainingPipelineExample.java)
Text-to-speech training pipeline with SameDiff and the DL4J audio processing stack

### SameDiff Operations (NEW)
* [SameDiffOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/SameDiffOpsExample.java)
SameDiff operations namespace overview
* [CNNOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/CNNOpsExample.java)
`sd.cnn()` namespace -- conv2d, pooling, batch norm, separable convolution
* [RNNOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/RNNOpsExample.java)
`sd.rnn()` namespace -- LSTM, GRU, SRU cells
* [TransformerOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/TransformerOpsExample.java)
`sd.nn()` namespace -- Multi-head attention, RoPE, RMS norm, KV cache management
* [LossOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/LossOpsExample.java)
`sd.loss()` namespace -- cross-entropy, MSE, hinge, Huber, cosine distance, etc.
* [LinalgOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/LinalgOpsExample.java)
`sd.linalg()` namespace -- SVD, Cholesky, QR decomposition, eigenvalues
* [ImageOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/ImageOpsExample.java)
`sd.image()` namespace -- resize, crop, pad, color space conversion
* [AudioOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/AudioOpsExample.java)
`sd.audio()` namespace -- STFT, mel spectrogram, MFCC extraction
* [SignalMathBitwiseOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/SignalMathBitwiseOpsExample.java)
Signal processing, math, bitwise, and random operations

### Training & Fine-Tuning (NEW)
* [SFTLoRATrainingConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/SFTLoRATrainingConfigExample.java)
SFT, LoRA, GRPO, DPO, and mixed precision training configurations
* [AdvancedPEFTConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/AdvancedPEFTConfigExample.java)
Advanced PEFT (Parameter-Efficient Fine-Tuning) method configurations
* [SpecializedPEFTConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/SpecializedPEFTConfigExample.java)
Specialized PEFT methods
* [MixedPrecisionTrainingExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/MixedPrecisionTrainingExample.java)
FP16/BF16 mixed precision training API reference
* [KnowledgeDistillationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/KnowledgeDistillationExample.java)
Knowledge distillation (teacher-student training)
* [KnowledgeDistillationConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/KnowledgeDistillationConfigExample.java)
Knowledge distillation configuration options
* [TransferLearningAndFreezingExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/TransferLearningAndFreezingExample.java)
Transfer learning, variable freezing, PeftModel, and training utilities
* [TransferLearningConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/TransferLearningConfigExample.java)
Transfer learning and fine-tuning configurations
* [LRScheduleConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/LRScheduleConfigExample.java)
Learning rate schedule configurations (cosine, linear warmup, etc.)
* [RLAlignmentConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/RLAlignmentConfigExample.java)
RL alignment (RLHF/GRPO/DPO) training configurations
* [DataCurationPipelineExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/DataCurationPipelineExample.java)
Data curation pipeline for LLM training data

### Advanced: Dynamic Shape Plan (DSP) Execution (NEW)
* [DSPExecutionExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPExecutionExample.java)
Dynamic Shape Plan execution basics
* [DSPAdvancedExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPAdvancedExample.java)
Advanced DSP API usage
* [DSPBackendsAndKernelSelectionExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPBackendsAndKernelSelectionExample.java)
DSP backends, kernel selection, and graph execution modes
* [DSPDiskCacheAndTritonExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiskCacheAndTritonExample.java)
Disk cache and Triton compilation cache for execution plans
* [DSPDiagnosticsAndDebuggingExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiagnosticsAndDebuggingExample.java)
Diagnostics, debugging, and plan introspection tools

### Advanced: LLM Generation Pipeline (NEW)
* [LLMGenerationPipelineExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/LLMGenerationPipelineExample.java)
Complete API reference for GenerationPipeline, SamplingConfig, KV cache, streaming generation, vision-language embeddings

### Custom DL4J Layers and Vertices
DL4J has supported custom layers for a long time. Using SameDiff layers has some advantages described [here](src/main/java/org/nd4j/examples/samediff/customizingdl4j/README.md).

* [Ex1BasicSameDiffLayerExample.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex1BasicSameDiffLayerExample.java)
Implement a custom DL4J layer using SameDiff.
* [Ex2LambdaLayer.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex2LambdaLayer.java)
Implement a simple custom DL4J lambda layer using SameDiff.
* [Ex3LambdaVertex.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex3LambdaVertex.java)
Implement a simple custom DL4J lambda vertex using SameDiff.

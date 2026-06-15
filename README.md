# Eclipse Deeplearning4J Examples

## Branch: `1.0.0-rewrite` -- Bridge Release

> **This branch is a bridge release targeting DL4J `1.0.0-SNAPSHOT`.**
>
> The Maven groupId has been fully migrated to **`org.eclipse.deeplearning4j`** for all artifacts.
> Java source packages (`org.deeplearning4j`, `org.nd4j`, `org.datavec`) have **not** been renamespaced yet --
> that is planned for a future release. New modules (LLM, VLM, audio, OmniHub) use the new
> `org.eclipse.deeplearning4j.*` Java packages. This branch exists to let users build and run
> examples against the current `1.0.0-SNAPSHOT` artifacts while the full package renamespacing
> is still in progress.
>
> **What changed from master:**
> - Java 11 minimum (was Java 8)
> - Snapshot repository moved to `https://central.sonatype.com/repository/maven-snapshots/`
> - CUDA backend updated to `nd4j-cuda-12.9-platform` (was `nd4j-cuda-10.2-platform`)
> - All module versions normalized to `1.0.0-SNAPSHOT`
> - New examples: LLM text generation, VLM document understanding, speech-to-text,
>   GGML/GGUF import, OmniHub model hub, LoRA/PEFT fine-tuning, knowledge distillation,
>   mixed precision training, capsule networks, self-attention, and more

---

## Introduction

The **Eclipse Deeplearning4J** (DL4J) ecosystem is a set of projects intended to support all the needs of a JVM-based deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks.

The DL4J stack comprises of:
- **DL4J**: High level API to build MultiLayerNetworks and ComputationGraphs with a variety of layers, including custom ones. Supports importing Keras models from h5, including tf.keras models and distributed training on Apache Spark
- **ND4J**: General purpose linear algebra library with over 500 mathematical, linear algebra and deep learning operations. Based on the highly-optimized C++ codebase LibND4J that provides CPU (AVX2/512) and GPU (CUDA) support via OpenBLAS, OneDNN (MKL-DNN), cuDNN, cuBLAS, etc
- **SameDiff**: Automatic differentiation / deep learning framework using graph-based (define then run) execution. Supports importing TensorFlow and ONNX models. New in 1.0.0: LLM, VLM, and audio pipeline modules for running large language models, vision-language models, and audio models on the JVM
- **DataVec**: ETL for machine learning data in a wide variety of formats and files (HDFS, Spark, Images, Video, Audio, CSV, Excel etc)
- **OmniHub**: Model hub for downloading and managing pretrained models from the DL4J zoo and HuggingFace Hub (GGUF, SafeTensors, TorchScript formats)

## Prerequisites

- **Java 11** or higher (Java 8 is no longer supported as of 1.0.0)
- **Maven** >= 3.3.1

This example repo consists of several separate Maven Java projects, each with their own pom files. There is no shared parent POM at the root -- each top-level directory is a self-contained project that must be built from within its own directory.

Users can also refer to the [simple sample project provided](./mvn-project-template/pom.xml) to get started with a clean project from scratch.

## Maven Coordinates

All DL4J artifacts use the unified groupId:

```xml
<groupId>org.eclipse.deeplearning4j</groupId>
```

Artifact IDs remain the same (e.g., `deeplearning4j-core`, `nd4j-native`, `datavec-api`).

> **Note on Java packages:** The legacy core modules still use `org.deeplearning4j`, `org.nd4j`,
> and `org.datavec` Java packages. New modules (LLM, VLM, audio, OmniHub) use the new
> `org.eclipse.deeplearning4j.*` packages. A complete package renamespacing of all modules is
> planned for a future release.

Snapshot dependencies resolve from:
```
https://central.sonatype.com/repository/maven-snapshots/
```

## Build Commands

There is no root build. Always `cd` into a specific module first.

```bash
# Build a module (e.g., dl4j-examples)
cd dl4j-examples && mvn clean install

# Run tests for a module
cd dl4j-examples && mvn clean test

# Run a specific example class
cd dl4j-examples && mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.quickstart.modeling.feedforward.classification.IrisClassifier"

# Build with CUDA GPU support (any module)
cd dl4j-examples && mvn clean install -Dnd4j.backend=nd4j-cuda-12.9-platform
```

## GPU / CPU Backend Switching

Every module's pom.xml declares `<nd4j.backend>nd4j-native</nd4j.backend>` (CPU). To use CUDA, change this property to `nd4j-cuda-12.9-platform` either in the pom.xml or via `-Dnd4j.backend=nd4j-cuda-12.9-platform` on the command line.

---

## Example Content

Examples are separated into "quickstart" and "advanced" within each project. Below is a complete listing of every example in the repository, organized by module.

---

### [dl4j-examples](dl4j-examples/) -- High-Level DL4J API

#### Quickstart: Feedforward Networks

**Classification:**
- [IrisClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/IrisClassifier.java) -- End-to-end example introducing RecordReaders, MultiLayerConfiguration
- [LinearDataClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/LinearDataClassifier.java) -- Basic classification with plots
- [MNISTDoubleLayer](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MNISTDoubleLayer.java) -- Classify MNIST with multiple layers
- [MoonClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MoonClassifier.java) -- Model "moon"-shaped data with visualization
- [SaturnClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/SaturnClassifier.java) -- Model "saturn"-shaped data with visualization

**Regression:**
- [CSVDataModel](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/CSVDataModel.java) -- Basic regression with plots
- [MathFunctionsModel](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/MathFunctionsModel.java) -- Model various mathematical functions
- [SumModel](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/SumModel.java) -- Model addition on noisy synthetic data
- [ImageDrawer](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/ImageDrawer.java) -- Train a model to draw an image

**Unsupervised:**
- [MNISTAutoencoder](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/unsupervised/MNISTAutoencoder.java) -- Basic autoencoder introduction

#### Quickstart: Convolutional Neural Networks

- [LeNetMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNIST.java) -- Classic LeNet for MNIST digit classification
- [LeNetMNISTReLu](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNISTReLu.java) -- LeNet variant with ReLU
- [CIFARClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/CIFARClassifier.java) -- Classify the CIFAR dataset
- [CenterLossLeNetMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/CenterLossLeNetMNIST.java) -- Train an embedding using center loss
- [Conv1DUCISequenceClassification](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/Conv1DUCISequenceClassification.java) -- 1D convolution for sequence classification
- [DeconvolutionUpsamplingExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/DeconvolutionUpsamplingExample.java) -- **NEW** Deconvolution and upsampling layers for autoencoders/GANs
- [DepthwiseSeparableConvMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/DepthwiseSeparableConvMNIST.java) -- **NEW** MobileNet-style depthwise separable convolutions
- [LocallyConnectedPReLUExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LocallyConnectedPReLUExample.java) -- **NEW** LocallyConnected2D and PReLU layers

#### Quickstart: Recurrent Neural Networks

- [UCISequenceClassification](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/UCISequenceClassification.java) -- Time series classification
- [MemorizeSequence](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/MemorizeSequence.java) -- Train an RNN to memorize a character sequence
- [RNNEmbedding](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/RNNEmbedding.java) -- EmbeddingLayer as first layer in an RNN
- [VideoFrameClassifier](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/VideoFrameClassifier.java) -- Classify shapes in video frames (RNN + CNN + Dense)

#### Quickstart: Variational Auto Encoder

- [VaeMNISTAnomaly](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/variationalautoencoder/VaeMNISTAnomaly.java) -- Unsupervised anomaly detection on MNIST
- [VaeMNIST2dPlots](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/variationalautoencoder/VaeMNIST2dPlots.java) -- VAE latent space visualization

#### Quickstart: New in 1.0.0

- [EvaluationMetricsExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/evaluation/EvaluationMetricsExample.java) -- **NEW** Complete evaluation API reference (accuracy, F1, ROC, regression metrics)
- [WeightInitExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/initialization/WeightInitExample.java) -- **NEW** Weight initialization strategies (Xavier, He, Lecun, etc.)
- [TrainingListenersExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/listeners/TrainingListenersExample.java) -- **NEW** Training listeners and checkpointing
- [LayerNormExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/normalization/LayerNormExample.java) -- **NEW** Layer normalization on MNIST
- [GroupNormExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/normalization/GroupNormExample.java) -- **NEW** Group normalization on MNIST
- [NewOptimizersExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/optimization/NewOptimizersExample.java) -- **NEW** AdaBelief and Adam8bit optimizers
- [ModelSerializationExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/serialization/ModelSerializationExample.java) -- **NEW** Complete model save/load API reference
- [DataPipelineExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/datapipeline/DataPipelineExample.java) -- **NEW** DataVec ETL pipeline API reference

#### Quickstart: Features

- [SaveLoadMultiLayerNetwork](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/modelsavingloading/SaveLoadMultiLayerNetwork.java) -- Save and load a multilayer network
- [SaveLoadComputationGraph](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/modelsavingloading/SaveLoadComputationGraph.java) -- Save and load a computation graph
- [EarlyStoppingMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/earlystopping/EarlyStoppingMNIST.java) -- Early stopping on MNIST
- [PreSaveFirst](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/presavingdatasets/PreSaveFirst.java) & [LoadPreSavedLenetMnistExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/presavingdatasets/LoadPreSavedLenetMnistExample.java) -- Presaving datasets for faster training
- [WeightedLossFunctionExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/classimbalance/WeightedLossFunctionExample.java) -- Weighted loss for imbalanced classes
- [BasicUIExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/BasicUIExample.java) -- DL4J training UI
- [UIStorageExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/UIStorageExample.java) -- Save/reload training data for UI
- [RemoteUIExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/RemoteUIExample.java) -- Remote UI in a separate JVM

#### Advanced: Computer Vision

- [TinyYoloHouseNumberDetection](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/objectdetection/TinyYoloHouseNumberDetection.java) -- Object detection with bounding boxes via transfer learning
- [NeuralStyleTransfer](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/styletransfer/NeuralStyleTransfer.java) -- Neural style transfer
- [MultiDigitNumberRecognition](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/captcharecognition/MultiDigitNumberRecognition.java) -- Captcha recognition
- [DenseNetMain](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/densenet/DenseNetMain.java) -- DenseNet for animal image classification
- [SelfAttentionMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/attention/SelfAttentionMNIST.java) -- **NEW** Self-attention mechanism for MNIST
- [CapsNetMNIST](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/capsulenet/CapsNetMNIST.java) -- **NEW** Capsule network for MNIST

#### Advanced: Natural Language Processing

- [ImdbReviewClassificationRNN](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/pretrainedword2vec/ImdbReviewClassificationRNN.java) -- Sentiment classification with RNN
- [ImdbReviewClassificationCNN](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/pretrainedword2vec/ImdbReviewClassificationCNN.java) -- Sentiment classification with CNN
- [Paragraph Vectors](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/paragraphvectors/) -- Paragraph vector embedding examples
- [Sequence Vectors](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/sequencevectors/) -- Sequence vector examples
- [Word2Vec](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/word2vec/) -- Word2Vec training and uptraining
- [GenerateTxtModel](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/charmodelling/generatetext/GenerateTxtModel.java) -- Character-level text generation ("write Shakespeare")
- [EmbeddingLayerExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingnet/EmbeddingLayerExample.java) -- **NEW** EmbeddingLayer and EmbeddingSequenceLayer

#### Advanced: Sequence Models & Special Architectures

- [SequenceAnomalyDetection](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/sequenceanomalydetection/SequenceAnomalyDetection.java) -- Anomaly detection on sensor data
- [TrainLotteryModelSeqPrediction](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/sequenceprediction/TrainLotteryModelSeqPrediction.java) -- Sequence prediction on synthetic data
- [AlphaGoZeroTrainer](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/alphagozero/AlphaGoZeroTrainer.java) -- AlphaGo Zero model training
- [AdditionModelWithSeq2Seq](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/seq2seq/AdditionModelWithSeq2Seq.java) -- Seq2seq model that learns addition

#### Advanced: Features

- [CustomActivationUsageEx](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/activationfunctions/CustomActivationUsageEx.java) -- Custom activation functions
- [CustomLayerUsageEx](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/layers/CustomLayerUsageEx.java) -- Custom layers
- [CustomLossUsageEx](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/lossfunctions/CustomLossUsageEx.java) -- Custom loss functions
- [ParallelInferenceExample](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/inference/ParallelInferenceExample.java) -- Parallel inference
- [CSVExampleEvaluationMetaData](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/metadata/CSVExampleEvaluationMetaData.java) -- Trace data provenance and prediction errors
- [Transfer Learning](dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/) -- Edit, freeze, and fine-tune pretrained models (VGG16)

---

### [samediff-examples](samediff-examples/) -- SameDiff, LLM, VLM & Audio

#### Quickstart: Basics

- [Ex1_SameDiff_Basics](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex1_SameDiff_Basics.java) -- SameDiff class, variables, functions, forward pass
- [Ex2_LinearRegression](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex2_LinearRegression.java) -- Placeholders, forward pass, gradient calculations
- [Ex3_Variables](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/basics/Ex3_Variables.java) -- Alternate ways to create variables

#### Quickstart: Modeling

- [MNISTFeedforward](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/MNISTFeedforward.java) -- Create, train, evaluate, save and load a feedforward network
- [MNISTCNN](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/MNISTCNN.java) -- CNN network with SameDiff
- [CustomListenerExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/CustomListenerExample.java) -- Custom training listener
- [GraphOptimizerExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/GraphOptimizerExample.java) -- **NEW** SameDiff graph optimization passes

#### Quickstart: LLM / Text Generation (NEW)

- [QwenTextGenerationExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/QwenTextGenerationExample.java) -- **NEW** Full Qwen LLM pipeline: download GGUF, import to SameDiff, tokenize, generate text with sampling strategies and chat templates
- [GGMLImportExportExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/GGMLImportExportExample.java) -- **NEW** GGML/GGUF format detection, import, export, quantization/dequantization

#### Quickstart: Vision-Language Models (NEW)

- [SmolDoclingVLMExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/SmolDoclingVLMExample.java) -- **NEW** SmolDocling 256M VLM for document understanding (OCR, table extraction, markdown conversion)
- [VideoVLMExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/VideoVLMExample.java) -- **NEW** Video VLM preprocessing pipeline (frame extraction, temporal sampling)

#### Quickstart: Audio (NEW)

- [WhisperSpeechToTextExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/WhisperSpeechToTextExample.java) -- **NEW** OpenAI Whisper speech-to-text: model download, transcription, mel spectrogram, audio preprocessing
- [TtsTrainingPipelineExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/TtsTrainingPipelineExample.java) -- **NEW** Text-to-speech training pipeline

#### Quickstart: SameDiff Operations (NEW)

- [SameDiffOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/SameDiffOpsExample.java) -- **NEW** SameDiff operations namespace overview
- [CNNOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/CNNOpsExample.java) -- **NEW** sd.cnn() -- conv2d, pooling, batch norm, etc.
- [RNNOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/RNNOpsExample.java) -- **NEW** sd.rnn() -- LSTM, GRU, SRU cells
- [TransformerOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/TransformerOpsExample.java) -- **NEW** sd.nn() -- Multi-head attention, RoPE, RMS norm, KV cache
- [LossOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/LossOpsExample.java) -- **NEW** sd.loss() -- Cross-entropy, MSE, hinge, Huber, etc.
- [LinalgOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/LinalgOpsExample.java) -- **NEW** sd.linalg() -- SVD, Cholesky, QR, eigenvalues
- [ImageOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/ImageOpsExample.java) -- **NEW** sd.image() -- Resize, crop, pad, color space conversion
- [AudioOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/AudioOpsExample.java) -- **NEW** sd.audio() -- STFT, mel spectrogram, MFCC
- [SignalMathBitwiseOpsExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/operations/SignalMathBitwiseOpsExample.java) -- **NEW** Signal processing, math, bitwise, and random operations

#### Quickstart: Training & Fine-Tuning (NEW)

- [SFTLoRATrainingConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/SFTLoRATrainingConfigExample.java) -- **NEW** SFT, LoRA, GRPO, DPO, and mixed precision training configs
- [AdvancedPEFTConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/AdvancedPEFTConfigExample.java) -- **NEW** Advanced PEFT (Parameter-Efficient Fine-Tuning) configurations
- [SpecializedPEFTConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/SpecializedPEFTConfigExample.java) -- **NEW** Specialized PEFT methods
- [MixedPrecisionTrainingExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/MixedPrecisionTrainingExample.java) -- **NEW** FP16/BF16 mixed precision training
- [KnowledgeDistillationExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/KnowledgeDistillationExample.java) -- **NEW** Knowledge distillation (teacher-student)
- [KnowledgeDistillationConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/KnowledgeDistillationConfigExample.java) -- **NEW** Distillation configuration options
- [TransferLearningAndFreezingExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/TransferLearningAndFreezingExample.java) -- **NEW** Transfer learning, variable freezing, PeftModel
- [TransferLearningConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/TransferLearningConfigExample.java) -- **NEW** Transfer learning configuration
- [LRScheduleConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/LRScheduleConfigExample.java) -- **NEW** Learning rate schedule configurations
- [RLAlignmentConfigExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/RLAlignmentConfigExample.java) -- **NEW** RL alignment (RLHF/GRPO/DPO) configurations
- [DataCurationPipelineExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/training/DataCurationPipelineExample.java) -- **NEW** Data curation pipeline for LLM training

#### Advanced: Dynamic Shape Plan (DSP) Execution (NEW)

- [DSPExecutionExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPExecutionExample.java) -- **NEW** Dynamic Shape Plan execution
- [DSPAdvancedExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPAdvancedExample.java) -- **NEW** Advanced DSP API
- [DSPBackendsAndKernelSelectionExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPBackendsAndKernelSelectionExample.java) -- **NEW** DSP backends, kernel selection, graph execution modes
- [DSPDiskCacheAndTritonExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiskCacheAndTritonExample.java) -- **NEW** Disk cache and Triton compilation cache
- [DSPDiagnosticsAndDebuggingExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiagnosticsAndDebuggingExample.java) -- **NEW** Diagnostics, debugging, plan introspection

#### Advanced: LLM Generation Pipeline (NEW)

- [LLMGenerationPipelineExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/advanced/generation/LLMGenerationPipelineExample.java) -- **NEW** Complete API reference for GenerationPipeline, SamplingConfig, KV cache, streaming generation

#### Custom DL4J Layers with SameDiff

- [Ex1BasicSameDiffLayerExample](samediff-examples/src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex1BasicSameDiffLayerExample.java) -- Custom DL4J layer using SameDiff
- [Ex2LambdaLayer](samediff-examples/src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex2LambdaLayer.java) -- Custom lambda layer
- [Ex3LambdaVertex](samediff-examples/src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex3LambdaVertex.java) -- Custom lambda vertex

---

### [onnx-import-examples](onnx-import-examples/) -- ONNX, OmniHub & Model Import

#### ONNX Import

- [OnnxImportLoad](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/onnx/OnnxImportLoad.java) -- Import an ONNX model into SameDiff
- [OnnxImportSave](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/onnx/OnnxImportSave.java) -- Import and save an ONNX model
- [ImageProcessUtils](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/onnx/ImageProcessUtils.java) -- Image preprocessing utilities for inference

#### OmniHub & Multi-Format Import (NEW)

- [OmniHubPretrainedModels](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/omnihub/OmniHubPretrainedModels.java) -- **NEW** Load pretrained models from the DL4J zoo and HuggingFace Hub
- [HuggingFaceGGUFImport](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/omnihub/HuggingFaceGGUFImport.java) -- **NEW** Download and import GGUF models from HuggingFace
- [GGMLModelImportExample](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/omnihub/GGMLModelImportExample.java) -- **NEW** Low-level GGUF/GGML import and export API
- [SafeTensorsImportExample](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/omnihub/SafeTensorsImportExample.java) -- **NEW** SafeTensors format import (HuggingFace standard)
- [TorchScriptImportExample](onnx-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/omnihub/TorchScriptImportExample.java) -- **NEW** TorchScript (.pt) model import

---

### [nd4j-ndarray-examples](nd4j-ndarray-examples/) -- ND4J NDArray Operations

#### Quickstart

- [Nd4jEx0-10](nd4j-ndarray-examples/src/main/java/org/nd4j/examples/quickstart/) -- NDArray basics: creation, indexing, slicing, operations, accumulations, boolean indexing, matrix multiplication, reshaping, transformations, element-wise operations

#### Advanced

- [MultiClassLogitExample](nd4j-ndarray-examples/src/main/java/org/nd4j/examples/advanced/lowlevelmodeling/MultiClassLogitExample.java) -- Multiclass logistic regression from scratch
- [WorkspacesExample](nd4j-ndarray-examples/src/main/java/org/nd4j/examples/advanced/memoryoptimization/WorkspacesExample.java) -- Memory management with workspaces
- [Nd4jEx11-14](nd4j-ndarray-examples/src/main/java/org/nd4j/examples/advanced/operations/) -- BLAS AXPY, large matrices, serialization, normalizers
- [CustomOpsExamples](nd4j-ndarray-examples/src/main/java/org/nd4j/examples/advanced/operations/CustomOpsExamples.java) -- DynamicCustomOp usage

---

### [tensorflow-keras-import-examples](tensorflow-keras-import-examples/) -- TensorFlow & Keras Import

#### Keras

- [SimpleSequentialMlpImport](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/keras/quickstart/SimpleSequentialMlpImport.java) -- Import Keras Sequential model
- [SimpleFunctionalMlpImport](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/keras/quickstart/SimpleFunctionalMlpImport.java) -- Import Keras Functional model
- [ImportDeepMoji](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/keras/advanced/deepmoji/ImportDeepMoji.java) -- DeepMoji import with custom layer
- [KerasAdvancedLayerImportExample](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/keras/advanced/layertypes/KerasAdvancedLayerImportExample.java) -- **NEW** Comprehensive Keras layer type support reference

#### TensorFlow

- [MNISTMLP](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/MNISTMLP.java) -- Import a frozen TF model
- [BostonHousingPricesModel](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/BostonHousingPricesModel.java) -- Basic TF import
- [ModifyMNISTMLP](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/ModifyMNISTMLP.java) -- Import, modify graph, execute dynamically
- [TFGraphRunnerExample](tensorflow-keras-import-examples/src/main/java/org/deeplearning4j/modelimportexamples/tf/advanced/tfgraphrunnerinjava/TFGraphRunnerExample.java) -- Run a TensorFlow graph from Java

---

### [dl4j-distributed-training-examples](dl4j-distributed-training-examples/) -- Distributed Training on Spark

- [Tiny ImageNet](dl4j-distributed-training-examples/src/main/java/org/deeplearning4j/distributedtrainingexamples/tinyimagenet/) -- Train a CNN on Tiny ImageNet, local and Spark versions
- [Patent Classification](dl4j-distributed-training-examples/src/main/java/org/deeplearning4j/distributedtrainingexamples/patent/) -- Document classification on ~500GB raw text, demonstrates near-linear scaling

---

### [data-pipeline-examples](data-pipeline-examples/) -- DataVec ETL Pipelines

#### Loading Data

- [Ex01_FileSplitExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex01_FileSplitExample.java) -- FileSplit for loading files
- [Ex02_CollectionSplitExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex02_CollectionSplitExample.java) -- Split from a collection of URIs
- [Ex03_NumberedFileInputSplitExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex03_NumberedFileInputSplitExample.java) -- Numbered file patterns
- [Ex04_TransformSplitExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex04_TransformSplitExample.java) -- Map URIs to new URIs
- [Ex05_SamplingBaseInputSplitExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex05_SamplingBaseInputSplitExample.java) -- Train/validation/test splits
- [Ex06_KFoldIteratorFromDataSet](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex06_KFoldIteratorFromDataSet.java) -- K-Fold cross-validation

#### Transforming & Analyzing Data

- [IrisCSVTransform](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/IrisCSVTransform.java) -- Schema and TransformProcess basics
- [CSVMixedDataTypesLocal](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/CSVMixedDataTypesLocal.java) -- Column removal, filtering, invalid value replacement, datetime parsing
- [CSVMixedDataTypes](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/CSVMixedDataTypes.java) -- Same as above with Apache Spark
- [PrintSchemasAtEachStep](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/debugging/PrintSchemasAtEachStep.java) -- Debug transform pipelines
- [IrisAnalysis](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/analysis/IrisAnalysis.java) -- Dataset analysis as HTML
- [JoinExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/JoinExample.java) -- Dataset joins
- [PivotExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/PivotExample.java) -- Record pivoting by key
- [CustomReduceExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/transform/custom/CustomReduceExample.java) -- Custom reductions

#### Formats

- [SVMLightExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/formats/svmlight/SVMLightExample.java) -- MNIST in SVMLight format
- [ImagePipelineExample](data-pipeline-examples/src/main/java/org/deeplearning4j/datapipelineexamples/formats/image/ImagePipelineExample.java) -- Image pipeline with augmentation transforms

---

### [mvn-project-template](mvn-project-template/) -- Starter Template

- [LeNetMNIST](mvn-project-template/src/main/java/org/deeplearning4j/examples/sample/LeNetMNIST.java) -- Clean-slate project template with a basic LeNet example

---

### [android-examples](android-examples/) -- Android

Android application using DL4J. Uses Gradle (not Maven). See [README](android-examples/README.md).

---

### [oreilly-book-dl4j-examples](oreilly-book-dl4j-examples/) -- Legacy (beta2)

Legacy examples from the O'Reilly book. Uses older `1.0.0-beta2` version with **old Maven groupIds** (`org.deeplearning4j`, `org.nd4j`, `org.datavec`). Not updated for the 1.0.0 release. Kept for historical reference.

---

## Feedback & Contributions

While these examples don't cover all the features available in DL4J the intent is to cover functionality required for most users -- beginners and advanced. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) if you have feedback or feature requests that are not covered here.

We welcome contributions from the community. More information can be found [here](CONTRIBUTORS.md).

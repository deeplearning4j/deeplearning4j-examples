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

### Graph Neural Networks (NEW)
* [Ex1_GcnNodeClassification.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex1_GcnNodeClassification.java)
GCN node classification on a two-community graph -- `sd.gnn().gcnConv` over a symmetric-normalised adjacency, trained with a manual SGD loop
* [Ex2_GraphClassification.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex2_GraphClassification.java)
Graph-level classification (triangles vs paths) -- `sd.gnn().ginConv` node embeddings + `GraphPooling.globalMeanPool` readout over a batch of graphs
* [Ex3_LinkPrediction.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex3_LinkPrediction.java)
Variational Graph Auto-Encoder (VGAE) link prediction -- GCN encoder + `sd.gnn().vgaeReparam` / `vgaeKlLoss` / `innerProductDecoder`
* [Ex4_NeighborSampling.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex4_NeighborSampling.java)
Mini-batch GraphSAGE for large graphs -- `GraphSampler` multi-hop neighbour sampling feeding `sd.gnn().sageMean`
* [Ex5_GnnLayerGallery.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex5_GnnLayerGallery.java)
Forward gallery of the `sd.gnn()` conv layers -- GCN, GATv2, GraphSAGE, GIN, ChebConv, PNA, GCNII (the namespace also includes APPNP, JKNet, RGCN, HAN, NNConv, temporal GCN, PairNorm/GraphNorm)
* [Ex6_KnowledgeGraphCompletion.java](./src/main/java/org/nd4j/examples/samediff/gnn/Ex6_KnowledgeGraphCompletion.java)
Knowledge-graph link prediction -- `sd.kge().distMult` embeddings trained with `KgeTripleSampler` negative sampling + margin loss, evaluated with `KgeEvaluation` MRR / Hits@K (the `sd.kge()` namespace also provides TransE, TransH, ComplEx, RotatE and time-aware TransET; `sd.gnn()` adds the relational/heterogeneous encoders compGcnConv, rgatConvHead and hgtConvHead for knowledge graphs)

### LLM / Text Generation (NEW)
* [QwenTextGenerationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/llm/QwenTextGenerationExample.java)
Full Qwen LLM pipeline: download GGUF from HuggingFace, import into SameDiff, tokenize with HuggingFace tokenizer, generate text with sampling strategies and chat templates
* [GGMLImportExportExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/llm/GGMLImportExportExample.java)
GGML/GGUF format detection, model import, export, low-level GGUF I/O, quantization/dequantization
* [LLMGraphOptimizerExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/llm/LLMGraphOptimizerExample.java)
GraphOptimizer on a REAL imported Qwen3.5-0.8B: op histograms before/after, fusion evidence, logits-equivalence verification, FP16 weight pre-cast memory savings, forward-pass timing, pass-set control, and the GenerationPipeline integration point
* [QuantizationPerplexityComparisonExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/llm/QuantizationPerplexityComparisonExample.java)
What quantization actually costs, measured: Qwen3.5-0.8B at Q4_K_M vs Q8_0 scored on identical WikiText-2 sliding windows — perplexity / bits-per-byte / file size / import time / sample generation side by side, with sequential full-teardown between levels
* [AbliterationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/llm/AbliterationExample.java)
Activation ablation / "abliteration" (Arditi et al. NeurIPS 2024) — training-free residual-stream model editing: RefusalDirectionFinder (diff-in-means / PCA / projected), WeightOrthogonalizer (W' = W - alpha·(W@d)dᵀ), and the end-to-end AbliterationWorkflow. Self-contained: a synthetic model + activations separated along a planted direction, so recovery (|cos|≈1) and orthogonalization are verified in milliseconds with no download; the real-model workflow shape is shown as a reference block

### Vision-Language Models (NEW)
* [SmolDoclingVLMExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/vlm/SmolDoclingVLMExample.java)
SmolDocling 256M VLM for document understanding -- OCR, table extraction, markdown conversion from scanned pages/PDFs
* [SmolDoclingPdfToMarkdownExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/vlm/SmolDoclingPdfToMarkdownExample.java)
End-to-end document conversion: multi-page PDF (bring your own via `-Dexample.pdf.path`, or an auto-generated report) rendered with PDFBox, tiled through the VisionEncoder, decoded page-by-page with ONE reusable GenerationPipeline, parsed from DocTags into a document tree and written out as Markdown -- with per-page throughput and DSP plan-lifecycle reporting
* [VideoVLMExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/vlm/VideoVLMExample.java)
Video VLM preprocessing pipeline -- frame extraction, temporal sampling, video-to-text

### Audio (NEW)
* [WhisperSpeechToTextExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/audio/WhisperSpeechToTextExample.java)
OpenAI Whisper speech-to-text: model download, transcription, mel spectrogram extraction, audio preprocessing, tokenizer inspection
* [TtsTrainingPipelineExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/modeling/audio/TtsTrainingPipelineExample.java)
Text-to-speech training pipeline with SameDiff and the DL4J audio processing stack

### Pipeline: Model Loading & Tokenization (NEW)
* [HuggingFaceToTextExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/pipeline/HuggingFaceToTextExample.java)
The complete ingestion-to-output arc on a real model: the LLMModel×QuantType download catalog with offline cache checks and downloadCustom for arbitrary repos; tokenizer.json vs GGUF-embedded tokenizer metadata CROSS-CHECKED on the same model (finds the real BOS mismatch and the padded-vocab landmine — 248320 embedding rows vs 248070 decodable tokens); encoding anatomy (ids/tokens/mask, specials, round-trip, chat templates); and the correct prefix-delta incremental detokenization recipe for streaming UIs, shown against the broken naive per-token decode
* [AutoModelExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/pipeline/AutoModelExample.java)
AutoModel.fromPretrained() for GGUF/SafeTensors/ONNX/SDZ model loading, LoadConfig, OmniHub integration
* [TokenizerExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/pipeline/TokenizerExample.java)
HuggingFaceTokenizer: encode/decode, batch encoding, vocab operations, chat template formatting

### SameDiff Operations (NEW)
* [SameDiffOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/SameDiffOpsExample.java)
SameDiff operations namespace overview
* [CNNOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/CNNOpsExample.java)
`sd.cnn()` namespace -- conv2d, pooling, batch norm, separable convolution
* [RNNOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/RNNOpsExample.java)
`sd.rnn()` namespace -- LSTM, GRU, SRU cells
* [TransformerOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/TransformerOpsExample.java)
`sd.nn()` namespace -- Multi-head attention, RoPE, RMS norm, KV cache management
* [TransformerOpsAdvancedExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/TransformerOpsAdvancedExample.java)
FlashAttention, Grouped Query Attention, RoPE, Fused RoPE -- advanced transformer ops
* [MoEAndSSMOpsExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/operations/MoEAndSSMOpsExample.java)
Mixture of Experts (MoE) and Mamba-2 State Space Model (SSM) ops
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
* [LLMInstructionFineTuningExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/LLMInstructionFineTuningExample.java)
Real-pipeline SFT + LoRA: the actual Qwen3.5 BPE tokenizer and ChatML template, EXACT token-level response masking, a compact LLaMA-style decoder built from the production fused ops, full fine-tune via TrainingConfig/fit, LoRA adapters via PeftModel (frozen-base verification, adapter toggling, merge-and-unload, save), the SFTTrainingPipeline orchestration API, held-out perplexity before/after, and GGUF export
* [QwenToStudentDistillationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/QwenToStudentDistillationExample.java)
Knowledge distillation from a REAL teacher: Qwen3.5-0.8B imported from GGUF teaches a 26x-smaller student on WikiText-2 -- offline FP16 teacher-logit caching, the full Hinton loss (temperature-scaled KL + masked CE) built into the student graph and trained with fit(), teacher-student top-1 agreement, PerplexityEvaluator reference metrics, and side-by-side generations (teacher through a real GenerationPipeline)
* [SFTLoRATrainingConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/SFTLoRATrainingConfigExample.java)
SFT, LoRA, GRPO, DPO, and mixed precision training configurations
* [AdvancedPEFTConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/AdvancedPEFTConfigExample.java)
Advanced PEFT (Parameter-Efficient Fine-Tuning) method configurations
* [SpecializedPEFTConfigExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/SpecializedPEFTConfigExample.java)
Specialized PEFT methods
* [MixedPrecisionTrainingExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/MixedPrecisionTrainingExample.java)
FP16/BF16 mixed precision training API reference
* [FP8TrainingExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/FP8TrainingExample.java)
FP8 (E4M3/E5M2) mixed precision training with per-tensor scaling
* [Adam8bitGradientAccumulationExample.java](./src/main/java/org/nd4j/examples/samediff/quickstart/training/Adam8bitGradientAccumulationExample.java)
8-bit Adam optimizer and gradient accumulation for memory-efficient training
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

### Advanced: Dynamic Shape Plan (DSP) Execution

DSP is SameDiff's production execution engine. It compiles a computation graph
into an optimized plan (flat integer-indexed slots), then replays it at near-zero
overhead once shapes and pointers stabilize. The lifecycle is:
`SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING`.

Run any example with:
```
cd samediff-examples
mvn compile exec:java -Dexec.mainClass=org.nd4j.examples.samediff.advanced.execution.<ClassName>
```

All examples use the CPU backend (nd4j-native) and complete in under 10 seconds.
They describe what differs on CUDA/GPU where relevant.

* [DSPExecutionExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPExecutionExample.java)
  Builds a 2-layer feedforward network, executes inference with dynamic batch sizes (1 / 16 / 64 / 128),
  and shows how DSP compiles a new plan on shape change and reuses it on the same shape.
  Starting point for DSP.

* [DSPAdvancedExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPAdvancedExample.java)
  Complete API reference in Javadoc and log output: `sd.compileNativeDynamicShapePlan()`,
  `DspHandle` slot/segment/phase introspection, `DynamicShapePlan` serialization/visualization,
  distributed execution stubs (TensorParallelConfig, PipelineParallelRunner, DDP).
  Executes a real graph to show dynamic shape caching across batch sizes.

* [DSPBackendsAndKernelSelectionExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPBackendsAndKernelSelectionExample.java)
  Lists all 19 `GraphExecutionMode` enum values with native codes and flags.
  Demonstrates `KernelSelectionConfig` (strategy, engine priority, auto-tune, env config),
  `ExecutionPhase` and `PlanPhase` lifecycle enums, all three GPU JIT pipelines
  (Triton / NVRTC / PTX), CPU backends (oneDNN, ACL, MLX, OpenVINO), and all 24
  `GraphOptimizer` passes.

* [DSPDiagnosticsAndDebuggingExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiagnosticsAndDebuggingExample.java)
  Shows `DspDiagnostics` (18 category bitmask flags, 3 detail levels, JSON report),
  `DspDebugger.attach()` for plan analysis and step validation, `DspHandle` live
  introspection (slots, segments, NaN debugging, capture stats, buffer pool),
  and `DspPlanAssertions` for test/health-check assertions.

* [DSPDiskCacheAndTritonExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPDiskCacheAndTritonExample.java)
  Demonstrates `DspPlanDiskCache` (enable/disable, list cached hashes, load by model
  identity hash, store/invalidate), `TritonCacheManager` (export/import portable
  `.tkcache` bundles, arch validation), Triton compilation config (fusion scoring,
  graph capture, TF32, debug dumps), and the full 4-tier cache lookup order.

* [DSPPlanReuseAndBatchPlanningExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPPlanReuseAndBatchPlanningExample.java)
  Demonstrates the path to steady-state replay: compiles a 3-layer graph, tracks
  `DspHandle.StepSnapshot` across 8 warmup reps showing phase transitions, pre-warms
  the plan cache for expected batch sizes {1,4,8,16,32,64}, benchmarks plan swapping
  (sub-millisecond pointer swap between warm shapes vs compile latency for unseen shapes),
  and measures warmup vs steady-state throughput (samples/sec).

* [DSPReplayModesExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPReplayModesExample.java)
  Compares all four replay modes on the same graph: `SLOT_BY_SLOT` (correctness
  baseline), `EMULATED_REPLAY` (full lifecycle tracking without GPU graph APIs),
  `CUDA_GRAPHS` (hardware graph capture/replay, falls back gracefully on CPU),
  and `TRITON` (JIT-compiled fused kernels). Includes L2 accuracy comparison,
  autoregressive decode simulation (tokens/sec at batch=1), and batch throughput
  benchmark across modes.

* [DSPTrainingSteadyStateExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/execution/DSPTrainingSteadyStateExample.java)
  Shows DSP training (forward + backward + updater in a single compiled plan) with
  per-step phase tracking (`PlanPhase.REPLAYING` detection), warmup vs steady-state
  throughput comparison, segment-level capture stats, and a direct DSP-vs-non-DSP
  speedup benchmark using `sd.fit()` with a fixed batch size.

### Advanced: LLM Generation Pipeline (NEW)
* [LLMGenerationPipelineExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/LLMGenerationPipelineExample.java)
Complete API reference for GenerationPipeline, SamplingConfig, KV cache, streaming generation, vision-language embeddings
* [GenerationSessionContinuationExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/GenerationSessionContinuationExample.java)
Resumable decoding with GenerationSession on a real Qwen3.5-0.8B GGUF: generate in chunks, continue from the retained KV cache with no re-prefill, verify the greedy chunked==one-shot invariant, and run-to-completion in steps (the streaming/chat-server pattern, with cooperative cancel)
* [DraftModelSpeculativeDecodingExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/DraftModelSpeculativeDecodingExample.java)
Draft-model speculative decoding harness on real models: SmolLM2-135M draft configured against the SmolLM2-1.7B target (shared vocabulary — the platform benchmark's own pairing), baseline-vs-speculative comparison with acceptance-rate metrics and a lossless-verification equivalence oracle. NOTE: the pipeline does not yet wire speculation into its decode loops (it warns at create() and this example reports the zero metrics honestly) — this is the ready-made validation harness for when that lands
* [SpeculativeDecodingExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/SpeculativeDecodingExample.java)
Speculative decoding: NgramSpeculator, DraftModelSpeculator, SpeculativeDecodeLoop, acceptance rate tuning
* [ContinuousBatchingExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/generation/ContinuousBatchingExample.java)
Continuous batching: ContinuousBatchScheduler, ChunkedPrefillEngine, slot management, throughput optimization

### Advanced: LLM Evaluation (NEW)
* [LLMEvalBenchmarkExample.java](./src/main/java/org/nd4j/examples/samediff/advanced/evaluation/LLMEvalBenchmarkExample.java)
LLM evaluation harness: MMLU, ARC, GSM8K, HellaSwag, TruthfulQA, Winogrande benchmarks, custom datasets, metrics

### Custom DL4J Layers and Vertices
DL4J has supported custom layers for a long time. Using SameDiff layers has some advantages described [here](src/main/java/org/nd4j/examples/samediff/customizingdl4j/README.md).

* [Ex1BasicSameDiffLayerExample.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex1BasicSameDiffLayerExample.java)
Implement a custom DL4J layer using SameDiff.
* [Ex2LambdaLayer.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex2LambdaLayer.java)
Implement a simple custom DL4J lambda layer using SameDiff.
* [Ex3LambdaVertex.java](./src/main/java/org/nd4j/examples/samediff/customizingdl4j/Ex3LambdaVertex.java)
Implement a simple custom DL4J lambda vertex using SameDiff.

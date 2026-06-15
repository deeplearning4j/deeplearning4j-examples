# Eclipse Deeplearning4J Examples

## Branch: `1.0.0-rewrite` -- Bridge Release

> **This branch is a bridge release targeting DL4J `1.0.0-SNAPSHOT`.**
>
> The Maven groupId has been fully migrated to **`org.eclipse.deeplearning4j`** for all artifacts.
> Java source packages (`org.deeplearning4j`, `org.nd4j`, `org.datavec`) have **not** been renamespaced yet --
> that is planned for a future release. This branch exists to let users build and run examples against
> the current `1.0.0-SNAPSHOT` artifacts while the full package renamespacing is still in progress.
>
> **What changed from master:**
> - Java 11 minimum (was Java 8)
> - Snapshot repository moved to `https://central.sonatype.com/repository/maven-snapshots/`
> - CUDA backend updated to `nd4j-cuda-12.9-platform` (was `nd4j-cuda-10.2-platform`)
> - All module versions normalized to `1.0.0-SNAPSHOT`

---

## Introduction

The **Eclipse Deeplearning4J** (DL4J) ecosystem is a set of projects intended to support all the needs of a JVM-based deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks.

The DL4J stack comprises of:
- **DL4J**: High level API to build MultiLayerNetworks and ComputationGraphs with a variety of layers, including custom ones. Supports importing Keras models from h5, including tf.keras models and also supports distributed training on Apache Spark
- **ND4J**: General purpose linear algebra library with over 500 mathematical, linear algebra and deep learning operations. ND4J is based on the highly-optimized C++ codebase LibND4J that provides CPU (AVX2/512) and GPU (CUDA) support and acceleration by libraries such as OpenBLAS, OneDNN (MKL-DNN), cuDNN, cuBLAS, etc
- **SameDiff**: Part of the ND4J library, SameDiff is the automatic differentiation / deep learning framework. SameDiff uses a graph-based (define then run) approach, similar to TensorFlow graph mode. SameDiff supports importing TensorFlow frozen model format .pb (protobuf) models and ONNX models. DL4J also has full SameDiff support for writing custom layers and loss functions. New in 1.0.0: SameDiff LLM, VLM, and audio pipeline modules for running large language models, vision-language models, and audio models on the JVM
- **DataVec**: ETL for machine learning data in a wide variety of formats and files (HDFS, Spark, Images, Video, Audio, CSV, Excel etc)
- **LibND4J**: C++ library that underpins everything. For more information on how the JVM accesses native arrays and operations refer to [JavaCPP](https://github.com/bytedeco/javacpp)

All projects in the DL4J ecosystem support Windows, Linux and macOS. Hardware support includes CUDA GPUs (12.9), x86 CPU (x86_64, avx2, avx512), ARM CPU (arm, arm64, armhf) and PowerPC (ppc64le).

## Prerequisites

- **Java 11** or higher (Java 8 is no longer supported as of 1.0.0)
- **Maven** >= 3.3.1

This example repo consists of several separate Maven Java projects, each with their own pom files. There is no shared parent POM at the root -- each top-level directory is a self-contained project that must be built from within its own directory.

Users can also refer to the [simple sample project provided](./mvn-project-template/pom.xml) to get started with a clean project from scratch.

## Maven Coordinates

All DL4J artifacts now use the unified groupId:

```xml
<groupId>org.eclipse.deeplearning4j</groupId>
```

Artifact IDs remain the same (e.g., `deeplearning4j-core`, `nd4j-native`, `datavec-api`).

> **Note on Java packages:** While the Maven groupId is `org.eclipse.deeplearning4j`, Java import
> packages are still `org.deeplearning4j`, `org.nd4j`, and `org.datavec`. A complete package
> renamespacing is planned for a future release.

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

## Example Content

Projects are based on what functionality the included examples demonstrate to the user and not necessarily which library in the DL4J stack the functionality lives in.

Examples in a project are in general separated into "quickstart" and "advanced".

Each project README also lists all the examples it contains, with a recommended order to explore them in.

| Module | Description |
|--------|-------------|
| [dl4j-examples](dl4j-examples/README.md) | High level DL4J API: feedforward, CNN, RNN, NLP, autoencoders, transfer learning |
| [nd4j-ndarray-examples](nd4j-ndarray-examples/README.md) | ND4J NDArray operations (NumPy equivalent for the JVM) |
| [samediff-examples](samediff-examples/README.md) | SameDiff auto-differentiation, LLM/VLM/audio pipelines, GGML model loading |
| [tensorflow-keras-import-examples](tensorflow-keras-import-examples/README.md) | Import TensorFlow .pb and Keras .h5 models |
| [onnx-import-examples](onnx-import-examples/README.md) | ONNX model import via SameDiff, OmniHub pretrained models |
| [dl4j-distributed-training-examples](dl4j-distributed-training-examples/README.md) | Distributed training on Apache Spark |
| [data-pipeline-examples](data-pipeline-examples/README.md) | DataVec ETL pipelines for data loading/preprocessing |
| [mvn-project-template](mvn-project-template/) | Clean-slate starter template for new DL4J projects |
| [android-examples](android-examples/README.md) | Android application using DL4J (Gradle, not Maven) |
| [oreilly-book-dl4j-examples](oreilly-book-dl4j-examples/) | Legacy examples (older beta2 version, uses old groupIds) |

## Feedback & Contributions

While these examples don't cover all the features available in DL4J the intent is to cover functionality required for most users -- beginners and advanced. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) if you have feedback or feature requests that are not covered here.

We welcome contributions from the community. More information can be found [here](CONTRIBUTORS.md).

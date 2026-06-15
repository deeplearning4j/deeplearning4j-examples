# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch: `1.0.0-rewrite`

This is a **bridge release** branch. The Maven groupId migration to `org.eclipse.deeplearning4j` is complete, but Java package renamespacing (e.g., `org.deeplearning4j` -> new namespace) has **not** happened yet. That is planned for a future release.

## Repository Structure

This is the **Eclipse Deeplearning4J examples** repository -- a collection of **independent Maven projects** (no shared parent POM at root). Each top-level directory is a self-contained project that must be built from within its own directory.

| Module | Purpose |
|---|---|
| `dl4j-examples/` | High-level DL4J API: feedforward, CNN, RNN, NLP, autoencoders, transfer learning |
| `nd4j-ndarray-examples/` | ND4J NDArray operations (NumPy equivalent for the JVM) |
| `samediff-examples/` | SameDiff auto-differentiation, LLM/VLM/audio pipelines, GGML model loading |
| `tensorflow-keras-import-examples/` | Import TensorFlow .pb and Keras .h5 models |
| `onnx-import-examples/` | ONNX model import via SameDiff, OmniHub pretrained models |
| `dl4j-distributed-training-examples/` | Distributed training on Apache Spark |
| `data-pipeline-examples/` | DataVec ETL pipelines for data loading/preprocessing |
| `android-examples/` | Android application using DL4J (Gradle, not Maven) |
| `mvn-project-template/` | Clean-slate starter template for new DL4J projects |
| `oreilly-book-dl4j-examples/` | Legacy examples (older beta2 version, uses old groupIds -- not updated) |

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

**Requirements:** Java 11+, Maven >= 3.3.1

## DL4J Version

All modules target DL4J stack version **1.0.0-SNAPSHOT** (`dl4j-master.version` property in each pom.xml).

Maven groupId is the unified **`org.eclipse.deeplearning4j`** for all artifacts. Artifact IDs remain the same (e.g., `nd4j-native`, `deeplearning4j-core`, `datavec-api`).

Java import packages are still the legacy names (`org.deeplearning4j`, `org.nd4j`, `org.datavec`) -- only the Maven groupId has changed so far.

Snapshot dependencies resolve from `https://central.sonatype.com/repository/maven-snapshots/`.

## GPU / CPU Backend Switching

Every module's pom.xml declares `<nd4j.backend>nd4j-native</nd4j.backend>` (CPU). To use CUDA, change this property to `nd4j-cuda-12.9-platform` either in the pom.xml or via `-Dnd4j.backend=nd4j-cuda-12.9-platform` on the command line.

## Test Structure

Tests use **JUnit 5** (Jupiter 5.8.0-M1) via `maven-surefire-plugin` with `surefire-junit-platform`. Each module has a single `QuickTest.java` in `src/test/java/` that acts as a sanity check by calling example `main()` methods directly. These are not unit tests in the traditional sense -- they verify that examples run without crashing.

## Code Conventions

- Java 11 source (compiler release=11)
- 4-space indentation, UTF-8, LF line endings (see `.editorconfig`)
- Apache 2.0 license headers on all source files
- Examples are organized under `src/main/java/` by topic (e.g., `quickstart/`, `advanced/`, `recurrent/`, `feedforward/`)
- Most examples are standalone classes with a `public static void main(String[] args)` entry point

## Key Architectural Patterns

- **MultiLayerNetwork**: Sequential stack of layers (most dl4j-examples)
- **ComputationGraph**: DAG-structured networks with multiple inputs/outputs (advanced examples)
- **SameDiff**: Define-then-run computation graphs with automatic differentiation (samediff-examples)
- **SameDiff LLM/VLM/Audio**: New pipeline modules for running large language models, vision-language models, and audio models on the JVM
- **DataVec pipelines**: `RecordReader` -> `DataSetIterator` pattern for loading/transforming data before feeding to networks
- **Model import**: TensorFlow/Keras/ONNX models are imported into DL4J's native format (`SameDiff.importFrozenTF()`, `KerasModelImport`, etc.)
- **OmniHub**: Model hub for downloading and managing pretrained models
- **Spark training**: `SparkDl4jMultiLayer` / `SparkComputationGraph` wrappers with `TrainingMaster` for distributed SGD

## CI

GitHub Actions workflow (`.github/workflows/example-sanity-check.yml`) runs `mvn clean test` in every module (except `mvn-project-template`) on a 12-hour cron schedule.

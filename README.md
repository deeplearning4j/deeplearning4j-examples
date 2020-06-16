<pre>
                                ########  ##       ##              ##
                                ##     ## ##       ##    ##        ##
                                ##     ## ##       ##    ##        ##
                       **$**    ##     ## ##       ##    ##        ##    **$**
                                ##     ## ##       ######### ##    ##
                                ##     ## ##             ##  ##    ##
                                ########  ########       ##   ######
              .   :::: :   :    :   :     : ::::  :     ::::    :::::  :::: ::::  :::::   .
              .   :    :   :   : :  ::   :: :   : :     :       :   :  :    :   : :   :   .
              .   :     : :   :   : : : : : :   : :     :       :   :  :    :   : :   :   .
              .   :::    :    :   : :  :  : ::::  :     :::     :::::  :::  ::::  :   :   .
              .   :     : :   ::::: :     : :     :     :       :  :   :    :     :   :   .
              .   :    :   :  :   : :     : :     :     :       :   :  :    :     :   :   .
              .   :::: :   :  :   : :     : :     ::::: ::::    :    : :::: :     :::::   .
</pre>

## Introduction
The **Eclipse Deeplearning4J** (DL4J) ecosystem is a set of projects intended to support all the needs of a JVM based deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks.

The DL4J stack comprises of:
- **DL4J**: High level API to build MultiLayerNetworks and ComputationGraphs with a variety of layers, including custom ones. Supports importing Keras models from h5, including tf.keras models (as of 1.0.0-beta7) and also supports distributed training on Apache Spark
- **ND4J**: General purpose linear algebra library with over 500 mathematical, linear algebra and deep learning operations. ND4J is based on the highly-optimized C++ codebase LibND4J that provides CPU (AVX2/512) and GPU (CUDA) support and acceleration by libraries such as OpenBLAS, OneDNN (MKL-DNN), cuDNN, cuBLAS, etc
- **SameDiff** : Part of the ND4J library, SameDiff is our automatic differentiation / deep learning framework. SameDiff uses a graph-based (define then run) approach, similar to TensorFlow graph mode. Eager graph (TensorFlow 2.x eager/PyTorch) graph execution is planned. SameDiff supports importing TensorFlow frozen model format .pb (protobuf) models. Import for ONNX, TensorFlow SavedModel and Keras models are planned. Deeplearning4j also has full SameDiff support for easily writing custom layers and loss functions.
- **DataVec**: ETL for machine learning data in a wide variety of formats and files (HDFS, Spark, Images, Video, Audio, CSV, Excel etc)
- **Arbiter**: Library for hyperparameter search
- **LibND4J** : C++ library that underpins everything. For more information on how the JVM acceses native arrays and operations refer to [JavaCPP](https://github.com/bytedeco/javacpp)

All projects in the DL4J ecosystem support Windows, Linux and macOS. Hardware support includes CUDA GPUs (10.0, 10.1, 10.2 except OSX), x86 CPU (x86_64, avx2, avx512), ARM CPU (arm, arm64, armhf) and PowerPC (ppc64le).

## Prerequisites
This example repo consists of several separate Maven Java projects, each with their own pom files. Maven is a popular build automation tool for Java Projects. The contents of a "pom.xml" file dictate the configurations. Read more about how to configure Maven [here](https://deeplearning4j.konduit.ai/config/maven).

Users can also refer to the [simple sample project provided](./mvn-project-template/pom.xml) to get started with a clean project from scratch.

Build tools are considered standard software engineering best practice. Besides this the complexities posed by the projects in the DL4J ecosystem make dependencies too difficult to manage manually. All the projects in the DL4J ecosystem can be used with other build tools like Gradle, SBT etc. More information on that can be found [here](https://deeplearning4j.konduit.ai/config/buildtools).

## Example Content
Projects are based on what functionality the included examples demonstrate to the user and not necessarily which library in the DL4J stack the functionality lives in.

Examples in a project are in general separated into "quickstart" and "advanced".

Each project README also lists all the examples it contains, with a recommended order to explore them in.

- [dl4j-examples](dl4j-examples/README.md)
This project contains a set of examples that demonstrate use of the high level DL4J API to build a variety of neural networks.
Some of  these examples are end to end, in the sense they start with raw data, process it and then build and train neural networks on it.

- [tensorflow-keras-import-examples](tensorflow-keras-import-examples/README.md)
This project contains a set of examples that demonstrate how to import Keras h5 models and TensorFlow frozen pb models into the DL4J ecosystem. Once imported into DL4J these models can be treated like any other DL4J model - meaning you can continue to run training on them or modify them with the transfer learning API or simply run inference on them.

- [dl4j-distributed-training-examples](dl4j-distributed-training-examples/README.md)
This project contains a set of examples that demonstrate how to do distributed training, inference and evaluation in DL4J on Apache Spark. DL4J distributed training employs a "hybrid" asynchronous SGD approach - further details can be found in the distributed deep learning documentation [here](https://deeplearning4j.konduit.ai/distributed-deep-learning/intro)

- [cuda-specific-examples](cuda-specific-examples/README.md)
This project contains a set of examples that demonstrate how to leverage multiple GPUs for data-parallel training of neural networks for increased performance.

- [samediff-examples](samediff-examples/README.md)
This project contains a set of examples that demonstrate the SameDiff API. SameDiff (which is part of the ND4J library) can be used to build lower level auto-differentiating computation graphs. An analogue to the SameDiff API vs the DL4J API is the low level TensorFlow API vs the higher level of abstraction Keras API.

- [data-pipeline-examples](data-pipeline-examples/README.md)
This project contains a set of examples that demonstrate how raw data in various formats can be loaded, split and preprocessed to build serializable (and hence reproducible) ETL pipelines.

- [nd4j-ndarray-examples](nd4j-ndarray-examples/README.md)
This project contains a set of examples that demonstrate how to manipulate NDArrays. The functionality of ND4J demonstrated here can be likened to NumPy.

- [arbiter-examples](arbiter-examples/README.md)
This project contains a set of examples that demonstrate usage of the Arbiter library for hyperparameter tuning of Deeplearning4J neural networks.

- [rl4j-examples](rl4j-examples/README.md)
This project contains examples of using RL4J, the reinforcement learning library in DL4J.

- [android-examples](android-examples/README.md)
This project contains an Android example project, that shows DL4J being used in an Android application.

## Feedback & Contributions
While these set of examples don't cover all the features available in DL4J the intent is to cover functionality required for most users - beginners and advanced.  File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) if you have feedback or feature requests that are not covered here. We are also available via our [community forum](https://community.konduit.ai/) for questions.
We welcome contributions from the community. More information can be found [here](CONTRIBUTORS.md)
We **love** hearing from you. Cheers!

## Eclipse Deeplearning4j: Distributed Training Examples

This project contains a set of examples that demonstrate how to do distributed training in DL4J. DL4J distributed training employs a "hybrid" asynchronous SGD based on Niko Strom's paper linked [here](http://nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf). More information on DL4J's distributed training methods and how they work can be found [here](https://deeplearning4j.konduit.ai/distributed-deep-learning/intro). DL4J's distributed training implementation is also fault tolerant.

Of note - Spark is only relied upon for three specific tasks: 1) Broadcasting the initial neural network parameters to all workers 2) Distributing the RDD datasets to the workers 3) Spark's fault tolerance system to detect and bring up a replacement workers. For all other communication between nodes like transferring quantized gradient updates Aeron is used.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

The examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.

* [tinyimagenet](src/main/java/org/deeplearning4j/distributedtrainingexamples/tinyimagenet)
Train a CNN network from scratch on the Tiny ImageNet dataset. A local (single machine) version is also available.

* [Patent Classification](src/main/java/org/deeplearning4j/distributedtrainingexamples/patent/README.md)
A real world document classification example on ~500GB of raw text. A local (single machine) version is also provided to demonstrate the reduction in training time to converge to the same level of the accuracy. Experiments have demonstrated a near linear scaling with the number of workers in the cluster!

NOTE: For parallel inference take a look at the

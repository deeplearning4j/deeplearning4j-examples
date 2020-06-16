##  Eclipse Deeplearning4j: Model Import

The DL4J supports models created in the popular Python Tensorflow and Keras frameworks. As of 1.0.0-beta7, Keras models (including tf.keras) can be imported into Deeplearning. TensorFlow frozen format models can be imported into SameDiff.

Models in Tensorflow have to be converted to "frozen" pbs (protobuf). More information on freezing Tensorflow models can be found [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py#L15) for Tensorflow 1.X and [here](https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/) for Tensorflow 2.X. Keras models have to be saved in h5 format. More information can be found [here](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model). Importing both Keras 1 and Keras 2 models are supported. Of note - importing models saved with tf.keras is also supported. Currently general TensorFlow operations within Keras models (i.e., those not part of the tf.keras API) are currently importable but support inference only. Full training is supported for anything that is part of the Keras API.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

The examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.
There is an FAQ gathered from the example READMEs available [here](FAQ.md) as well.

## Keras

### Quickstart
* [SimpleSequentialMlpImport.java](./src/main/java/org/deeplearning4j/modelimportexamples/keras/quickstart/SimpleSequentialMlpImport.java)
Basic example for importing a Keras Sequential model into DL4J for training or inference.
* [SimpleFunctionalMlpImport.java](./src/main/java/org/deeplearning4j/modelimportexamples/keras/quickstart/SimpleFunctionalMlpImport.java)
Basic example for importing a Keras functional Model into DL4J for training or inference.

### Advanced
* [ImportDeepMoji.java](./src/main/java/org/deeplearning4j/modelimportexamples/keras/advanced/deepmoji/ImportDeepMoji.java)
Import of DeepMoji application. Demonstrates implementing a custom layer for import.


## Tensorflow

### Quickstart
* [MNISTMLP.java](./src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/MNISTMLP.java)
Basic example imports a frozen TF model trained on mnist. Python scripts used available.
* [BostonHousingPricesModel.java](./src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/BostonHousingPricesModel.java)
Another basic example with the boston housing prices dataset
* [ModifyMNISTMLP.java](./src/main/java/org/deeplearning4j/modelimportexamples/tf/quickstart/ModifyMNISTMLP.java)
Import a frozen TF model. Demonstrate static execution, modify the graph and then execute it dynamically.

### Advanced
* [ImportMobileNetExample.md](./src/main/java/org/deeplearning4j/modelimportexamples/tf/advanced/mobilenet/ImportMobileNetExample.md)
Import MobileNet and run inference on it to give the same metrics as those obtained in Tensorflow.
* [TFGraphRunnerExample.java](./src/main/java/org/deeplearning4j/modelimportexamples/tf/advanced/tfgraphrunnerinjava/TFGraphRunnerExample.java)
Runs a tensorflow graph from java using the tensorflow graph runner.
* [MobileNetTransferLearningExample.md](./src/main/java/org/deeplearning4j/modelimportexamples/tf/advanced/mobilenet/MobileNetTransferLearningExample.md)
Transfer learning on an imported TF mobile net model for CIFAR10
* [BertInferenceExample.md](./src/main/java/org/deeplearning4j/modelimportexamples/tf/advanced/bert/BertInferenceExample.md)
Run inference on a BERT model trained in Tensorflow to give the same metrics as those obtained in Tensorflow.


## Eclipse Deeplearning4j: CUDA Specific Examples

Switching from a CPU only backend to a GPU backend is as simple as changing one dependency - one line in the pom.xml file for Maven users. Instead of specifying the nd4j-native-platform module specify the nd4j-cuda-X-platform where X indicated the version of CUDA. It is recommended to install cuDNN for better GPU performance. Runs will log warnings if cuDNN is not found. For more information, please refer to documentation [here](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn)

Users with acces to multiple gpus systems can use DL4J to further speed up the training process by training the models in parallel on them. Ideally these GPUs have the same speed and networking capabilities. This project contains a set of examples that demonstrate how to leverage performance from a multiple gpus setup. More documentation can be found [here](https://deeplearning4j.konduit.ai/getting-started/tutorials/using-multiple-gpus)

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

The list of examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.

## QUICKSTART
* [MultiGPULeNetMNIST.java](./src/main/java/org/deeplearning4j/examples/multigpu/quickstart/MultiGPULeNetMNIST.java)
Introduction to ParallelWrapper by modifying the original [LenetMnistExample](./../dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNIST.java)
* [GradientsSharingLeNetMNIST.java](./src/main/java/org/deeplearning4j/examples/multigpu/quickstart/GradientsSharingLeNetMNIST.java)
Uses gradient sharing instead of the default averaging every n iterations. More information on the gradient sharing algorithm can be found [here](https://deeplearning4j.konduit.ai/distributed-deep-learning/intro)
* [GradientSharingVGG16TinyImageNet.java](./src/main/java/org/deeplearning4j/examples/multigpu/quickstart/GradientSharingVGG16TinyImageNet.java)
Gradient sharing with VGG16 on TinyImageNet

## ADVANCED
* [ImdbReviewClassificationRNN.java](./src/main/java/org/deeplearning4j/examples/multigpu/advanced/w2vsentiment/ImdbReviewClassificationRNN.java)
A multiple gpus version of the example of the same name in the dl4j-examples repo [here](./../dl4j-examples//src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/pretrainedword2vec/ImdbReviewClassificationRNN.java) This example also includes how to presave the dataset to save time when training on multiple epochs.
* [GenerateTxtModel.java](./src/main/java/org/deeplearning4j/examples/multigpu/advanced/charmodelling/GenerateTxtModel.java)
CharModelling: A multiple gpus version of the example of the same name in the dl4j-examples repo, [here](./../dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/charmodelling/generatetext/GenerateTxtModel.java).
* [FeaturizedPreSave.java](./src/main/java/org/deeplearning4j/examples/multigpu/advanced/transferlearning/vgg16/FeaturizedPreSave.java) & [FitFromFeaturized.java](./src/main/java/org/deeplearning4j/examples/multigpu/advanced/transferlearning/vgg16/FitFromFeaturized.java)
Transferlearning: A multiple gpus version of the example of the same name in the dl4j-examples repo [here](./../dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/editlastlayer/presave)


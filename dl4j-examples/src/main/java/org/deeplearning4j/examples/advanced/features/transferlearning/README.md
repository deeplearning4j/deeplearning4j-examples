##### TransferLearning
Demonstrates use of the dl4j transfer learning API which allows users to construct a model based off an existing model by modifying the architecture, freezing certain parts selectively and then fine tuning parameters. Read the documentation for the Transfer Learning API at [https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning](https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning).
* [EditLastLayerOthersFrozen.java](./editlastlayer/EditLastLayerOthersFrozen.java)
Modifies just the last layer in vgg16, freezes the rest and trains the network on the flower dataset.
* [FeaturizedPreSave.java](./editlastlayer/presave/FeaturizedPreSave.java) & [FitFromFeaturized.java](./editlastlayer/presave/FitFromFeaturized.java)
Save time on the forward pass during multiple epochs by "featurizing" the datasets. FeaturizedPreSave saves the output at the last frozen layer and FitFromFeaturize fits to the presaved data so you can iterate quicker with different learning parameters.
* [EditAtBottleneckOthersFrozen.java](./editfrombottleneck/EditAtBottleneckOthersFrozen.java)
A more complex example of modifying model architecure by adding/removing vertices
* [FineTuneFromBlockFour.java](./finetuneonly/FineTuneFromBlockFour.java)
Reads in a saved model (training information and all) and fine tunes it by overriding its training information with what is specified

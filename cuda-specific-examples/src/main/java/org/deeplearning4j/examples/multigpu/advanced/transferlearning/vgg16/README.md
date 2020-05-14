##### TransferLearning
Demonstrates use of the dl4j transfer learning API which allows users to construct a model based off an existing model by modifying the architecture, freezing certain parts selectively and then fine tuning parameters. Read the documentation for the Transfer Learning API at [https://deeplearning4j.org/transfer-learning](https://deeplearning4j.org/transfer-learning).  

More more examples refer to the section in the dl4j-example repo [here](../../../../../../../../../../../dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/README.md)

* [FeaturizedPreSave.java](src/main/java/org/deeplearning4j/examples/multigpu/advanced/transferlearning/vgg16/FeaturizedPreSave.java) & FitFromFeaturized.java (src/main/java/org/deeplearning4j/examples/multigpu/advanced/transferlearning/vgg16/FitFromFeaturized.java)
Save time on the forward pass during multiple epochs by "featurizing" the datasets. FeaturizedPreSave saves the output at the last frozen layer and FitFromFeaturize fits to the presaved data so you can iterate quicker with different learning parameters.

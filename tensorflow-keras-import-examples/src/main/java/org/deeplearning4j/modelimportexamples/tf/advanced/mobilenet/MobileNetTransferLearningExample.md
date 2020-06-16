### SUMMARY
This example demonstrates importing the tensorflow mobile net model into samediff and appylying transfer learning to classify images in the CIFAR10 dataset

#### Modelling Statement:
Transfer learning saves time and effort. Using a pretrained model the example demonstrates how to modify only the last layer and train it to classify the CIFAR10 dataset. The model is fine tuned i.e the majority of the model parameters are the pretrained weights which are frozen. The only learnable parameters are the ones associated with the last layer.

##### Modeling Metrics:
This example runs only for one epoch. More epochs will give better results.

##### Example complexity: **MODERATE**
This is a moderately complex example. Users should at the very least have some exposure to the SameDiff API before diving into this.The following prerequisites are recommended:

- Familiarity with Evaluation classes in DL4J
- SameDiff quickstart examples for basics [here](../../../../../../../../../../samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart) in particular the [CustomListenerExample.java](../../../../../../../../../../samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart/modeling/CustomListenerExample.java) example
- Examples of importing and doing inference on familiar simple models like the ones [here](../../quickstart/)
- MobileNet import example (ImportMobileNetExample.java)

[Show me the code](./MobileNetTransferLearningExample.java)


#### Further Reading:
1) A more advanced example of importing a TF model with BERT [here](../bert/BertInferenceExample.md)
2) DL4J's transfer learning API. Examples found [here](../../../../../../../../../../dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/) and more documentation found [here](https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning).

###### FAQ CallBack from Example: This example answers the following questions.
1) How to import a TF frozen graph (.pb) into the DL4J ecosystem?
2) How to perform advanced operations in SameDiff?
3) How to modify a TF model in DL4J and run training on it?
4) How to set up a CustomListener in SameDiff to print shape information?

### SUMMARY
Import a Tensorflow model, specifically mobilenet into SameDiff and run inference on it to give the same metrics as those obtained in Tensorflow.

#### Modelling Statement:
This is an import of the Google mobile net example available here - https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz

##### Modeling Metrics:
Accuracy: 84 % 
F1 Score: 0.89

##### Example complexity: **INTERMEDIATE**  

- Familiarity with the NDArray indexing and basics
- Familiarity with what a ND4J dynamic op is and why it is used.
- Familiarity with Evaluation classes in DL4J
- SameDiff quickstart examples for basics [here](../../../../../../../../../../samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart)

[Show me the code](./ImportMobileNetExample.java)

#### Further Reading:
* A more advanced example that imports mobilenet and modifies it for CIFAR classification [here](MobileNetTransferLearningExample.md)
* A more advanced example of importing a TF model with BERT [here](../bert/BertInferenceExample.md)

###### FAQ CallBack from Example: This example answers the following questions.
1) How to import a TF frozen graph (.pb) into the DL4J ecosystem?
2) How to resize an image using bilinear interpolation?  
3) How to run inference on an imported Tensorflow model?



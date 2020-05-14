### SUMMARY
This example demonstrates how to import a fine tuned BERT model from Tensorflow into Samediff and run inference on it.

#### Modelling Statement:
A model based on BERT was finetuned in Tensorflow to classify document pairs from the MRPC dataset as like or unlike.
The trained model was converted to a pb file and imported into samediff. Python scripts for tuning and pb generation can be found in the repo [here](https://github.com/KonduitAI/dl4j-dev-tools/tree/master/import-tests/model_zoo/bert)
Once this model was imported into SameDiff we run inference on it with the MRPC test dataset to get the same evaluation metrics we get in Tensorflow.
We also run inference on a single minibatch and see that the ouputs are identical to what is obtained in Tensorflow.

##### Modeling Metrics:
Accuracy: 84 % 
F1 Score: 0.89
Link to a sample log from a successful run, modified for brevity, is [here](./.BertInferenceExample.logout)

##### Example complexity: **COMPLEX**  
This is a complex example. The main reason being that the frozen pb graph from TensorFlow has hard coded operations for dropout.
As a result the imported same diff graph has to be modified. Users are recommended to go through the following prerequisites before diving into this example.  

- Familiarity with the SentenceProvider class in DataVec(DL4J?) 
- Familiarity with the DL4J iterator classes
- Familiarity with Evaluation classes in DL4J
- SameDiff quickstart examples for basics [here](../../../../../../../../../../samediff-examples/src/main/java/org/nd4j/examples/samediff/quickstart)
- MobileNet transfer learning example for slightly less complicated example of modifying same diff graphs [here](./mobilenet/MobileNetTransferLearningExample.md)

[Show me the code](./BertInferenceExample.java)

#### Current Limitations:
* Only inference available on BERT models i.e. further training an imported BERT model in samediff is not available. Github issue linked [here](https://github.com/eclipse/deeplearning4j/issues/8052)
* Batchsizes in TF graphs are fixed and therefore also fixed in samediff 
* Bert tokenizer only available for the sentence classification task

#### Further Reading:
* BERT unit tests [here](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/BERTGraphTest.java#L49) and links to the python scripts for TF training and pb generation [here](https://github.com/KonduitAI/dl4j-dev-tools/tree/master/import-tests/model_zoo/bert).
* BERTTokenizer javadoc
* BERTIterator javadoc

###### FAQ CallBack from Example: This example answers the following questions.
1) How to import a TF frozen graph (.pb) into the DL4J ecosystem?
2) How to perform advanced operations in samediff? For eg. removing hard coded drop out ops in the TF graph for inference.
3) How to set up a SameDiff graph to generate Evaluation data?
4) How to set up an NLP pipeline for BERT going from raw text in a csv file to a BertIterator that gives feeds the model with the featurized text?
5) How to run inference on a single minibatch on a samediff graph?

## Eclipse Deeplearning4J Examples

This project contains a set of examples that demonstrate use of the high level DL4J API to build a variety of neural networks. The DL4J ecosystem also allows users to build neural networks with SameDiff (part of the ND4J library) with a more fine grained API. More information on that can be found [here](../samediff-examples)

The pom file in this project can be used as a template for a user's own project. The examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.



### QUICKSTART

#### Modeling Examples

##### Feedforward Neural Networks

###### Classification
* [IrisClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/IrisClassifier.java)
Basic end to end example that introduces important concepts like RecordReaders, MultiLayerConfiguration etc
* [LinearDataClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/LinearDataClassifier.java)
Basic end to end example with plots
* [MNISTSingleLayer.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MNISTSingleLayer.java)
Classify MNIST
* [MNISTDoubleLayer.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MNISTDoubleLayer.java)
Classify MNIST with more layers
* [ModelXOR.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/ModelXOR.java)
Model a two input XOR function (ie. a simple non linearly separable function)
* [MoonClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MoonClassifier.java)
Model data that separates into a "moon" shape and visualize the results
* [SaturnClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/SaturnClassifier.java)
Model data that separates into a "saturn" shape and visualize the results

###### Regression
* [CSVDataModel.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/CSVDataModel.java)
Basic end to end example with plots
* [MathFunctionsModel.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/MathFunctionsModel.java)
Model various mathematical functions
* [SumModel.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/SumModel.java)
Model addition on a synthetic dataset with noise
* [ImageDrawer.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/regression/ImageDrawer.java)
Train a model to draw an image!

###### Unsupervised
* [MNISTAutoencoder.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/unsupervised/MNISTAutoencoder.java)
A basic introduction to how to build an autoencoder


##### Convolutional Neural Networks
* [LeNetMNIST.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNIST.java)
The classic LeNet example for classifying hand-drawn digits (MNIST)
* [LeNetMNISTReLu.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNISTReLu.java)
The same as above with minor modifications
* [CIFARClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/CIFARClassifier.java)
Classify the CIFAR dataset
* [CenterLossLeNetMNIST.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/CenterLossLeNetMNIST.java)
Train an embedding using the center loss model, on MNIST

##### Recurrent Neural Networks
* [UCISequenceClassification.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/UCISequenceClassification.java)
Time series (sequence) classification on the UCI syntetic control time series dataset
* [MemorizeSequence.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/MemorizeSequence.java)
Train a RNN to memorize a sequence of characters
* [RNNEmbedding.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/RNNEmbedding.java)
Use an EmbeddingLayer (equivalent to using a DenseLayer with a one-hot representation for the input) as the first layer in an RNN
* [VideoFrameClassifier.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/recurrent/VideoFrameClassifier.java)
Classify shapes that appear in a video frame. Demonstrates how to combine RNN, CNN and fully connected dense layers in a single neural network. This is a memory consuming example. You need at least 7G of off heap memory. Refer [here](https://deeplearning4j.konduit.ai/config/config-memory) to configure memory off heap.



##### Variational Auto Encoder
* [VaeMNISTAnomaly.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/variationalautoencoder/VaeMNISTAnomaly.java)
Unsupervised anomaly detection on MNIST using a variational autoencoder
* [VaeMNIST2dPlots.java](./src/main/java/org/deeplearning4j/examples/quickstart/modeling/variationalautoencoder/VaeMNIST2dPlots.java)
Train a variational autoencoder on MNIST and plot MNIST digit reconstructions vs. the latent space as well as the latent space values for the MNIST test set as training progresses


#### Features

* [SaveLoadMultiLayerNetwork.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/modelsavingloading/SaveLoadMultiLayerNetwork.java)
Save and load a multilayer neural network
* [SaveLoadComputationGraph.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/modelsavingloading/SaveLoadComputationGraph.java)
Save and load a computational graph
* [EarlyStoppingMNIST.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/earlystopping/EarlyStoppingMNIST.java)
Early stopping example on a subset of MNIST i.e conduct training and use the parameters that give the minimum test set loss
* [PreSaveFirst.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/presavingdatasets/PreSaveFirst.java) & [LoadPreSavedLenetMnistExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/presavingdatasets/LoadPreSavedLenetMnistExample.java)
Save time when training with multiple epochs by presaving datasets
* [WeightedLossFunctionExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/classimbalance/WeightedLossFunctionExample.java)
Out of the box loss function that can be used with imbalanced classes
* [BasicUIExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/BasicUIExample.java)
Basic UI example
* [UIStorageExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/UIStorageExample.java)
Save training data to a file and reload it later to display in the UI
* [RemoteUIExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/userinterface/RemoteUIExample.java)
If you need the UI to be hosted in a separate JVM for training.
* [TSNEStandardExample.java](./src/main/java/org/deeplearning4j/examples/quickstart/features/tsne/TSNEStandardExample.java)
Basic TSNE

### ADVANCED

#### Modeling Examples

##### Computer Vision
* [TinyYoloHouseNumberDetection.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/objectdetection/TinyYoloHouseNumberDetection.java)
Transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC to perform object detection with bounding boxes on The Street View House Numbers Dataset.
* [NeuralStyleTransfer.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/styletransfer/NeuralStyleTransfer.java)
Neural Style Transfer Algorithm
* [MultiDigitNumberRecognition.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/captcharecognition/MultiDigitNumberRecognition.java)
Captcha recognition

##### Natural Language Processing

###### Text Classification
With pretrained word2vec:
* [ImdbReviewClassificationRNN.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/pretrainedword2vec/ImdbReviewClassificationRNN.java)
Sentiment Classification on the IMDB dataset with a RNN model
* [ImdbReviewClassificationCNN.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/pretrainedword2vec/ImdbReviewClassificationCNN.java)
Sentiment Classification on the IMDB dataset with a CNN model

###### Generating Embeddings:
* [Paragraph Vectors](./src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/paragraphvectors)
* [Sequence Vectors](./src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/sequencevectors)
* [Word2Vec](./src/main/java/org/deeplearning4j/examples/advanced/modelling/embeddingsfromcorpus/word2vec)

Modeling with a word2vec model trained on a custom corpus:
* [PrepareWordVector.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/customcorpusword2vec/PrepareWordVector.java), [TrainNews.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/textclassification/customcorpusword2vec/TrainNews.java)
Sentence classification using a word2vec model training on a custom corpus


###### Char Modelling
* [GenerateTxtModel.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/charmodelling/generatetext/GenerateTxtModel.java) & [GenerateTxtCharCompGraphModel.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/charmodelling/generatetext/GenerateTxtCharCompGraphModel.java)
MultiLayerNetwork and ComputationGraph versions of a model that is trained to "write Shakespeare" one character at a time, inspired by Andrej Karpathy's now famous blog post.

##### Other Sequence Modeling Examples
* [SequenceAnomalyDetection.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/sequenceanomalydetection/SequenceAnomalyDetection.java)
Anomaly detection on sequence sensor data
* [TrainLotteryModelSeqPrediction.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/sequenceprediction/TrainLotteryModelSeqPrediction.java)
Model trained on a synthetic dataset that attempts to uncover the contrived pattern.


##### Specific Models and Special Architectures
* [AlphaGoZeroTrainer.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/alphagozero/AlphaGoZeroTrainer.java)
Train AlphaGo Zero model on dummy data.
* [DenseNetMain.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/densenet/DenseNetMain.java)
Builds dense net to classify a small set of animal images. Augments the dataset with transforms like blur etc.
* [AdditionModelWithSeq2Seq.java](./src/main/java/org/deeplearning4j/examples/advanced/modelling/seq2seq/AdditionModelWithSeq2Seq.java)
A seq2seq model that learns to add

#### Features

##### Customizing DL4J
* [CustomActivationUsageEx.java](./src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/activationfunctions/CustomActivationUsageEx.java)
Implement custom activation functions
* [CustomLayerUsageEx.java](./src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/customlayers/CustomLayerUsageEx.java)
Implement custom layers
* [CustomLossUsageEx.java](./src/main/java/org/deeplearning4j/examples/advanced/features/customizingdl4j/lossfunctions/CustomLossUsageEx.java)
Implement custom loss functions

**NOTE**: SameDiff which is part of ND4J gives users a way to customize DL4J. More information on that is found [here](../samediff-examples)

##### Performance
* [ParallelInferenceExample.java](./src/main/java/org/deeplearning4j/examples/advanced/features/inference/ParallelInferenceExample.java)
How to run parallel inference in DL4J

##### Debugging
* [CSVExampleEvaluationMetaData.java](./src/main/java/org/deeplearning4j/examples/advanced/features/metadata/CSVExampleEvaluationMetaData.java)
Trace where data from each example comes from and get metadata on prediction errors
* [MultiLayerNetworkExternalErrors.java](./src/main/java/org/deeplearning4j/examples/advanced/features/externalerrors/MultiLayerNetworkExternalErrors.java)
Train a MultiLayerNetwork where the errors come from an external source, instead of using an Output layer and a labels array.

##### TransferLearning
Demonstrates use of the dl4j transfer learning API which allows users to construct a model based off an existing model by modifying the architecture, freezing certain parts selectively and then fine tuning parameters. Read the documentation for the Transfer Learning API at [https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning](https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning).
* [EditLastLayerOthersFrozen.java](./src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/editlastlayer/EditLastLayerOthersFrozen.java)
Modifies just the last layer in vgg16, freezes the rest and trains the network on the flower dataset.
* [FeaturizedPreSave.java](./src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/editlastlayer/presave/FeaturizedPreSave.java) & [FitFromFeaturized.java](./src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/editlastlayer/presave/FitFromFeaturized.java)
Save time on the forward pass during multiple epochs by "featurizing" the datasets. FeaturizedPreSave saves the output at the last frozen layer and FitFromFeaturize fits to the presaved data so you can iterate quicker with different learning parameters.
* [EditAtBottleneckOthersFrozen.java](./src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/editfrombottleneck/EditAtBottleneckOthersFrozen.java)
A more complex example of modifying model architecure by adding/removing vertices
* [FineTuneFromBlockFour.java](./src/main/java/org/deeplearning4j/examples/advanced/features/transferlearning/finetuneonly/FineTuneFromBlockFour.java)
Reads in a saved model (training information and all) and fine tunes it by overriding its training information with what is specified


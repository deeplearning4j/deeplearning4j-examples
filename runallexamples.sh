#!/usr/bin/env bash
#set -eu

## declare an array variable
declare -a arr=(
"org.deeplearning4j.examples.convolution.LenetMnistExample"
"org.deeplearning4j.examples.feedforward.xor.XorExample"
"org.deeplearning4j.examples.feedforward.regression.RegressionSum"
"org.deeplearning4j.examples.feedforward.regression.RegressionMathFunctions"
"org.deeplearning4j.examples.feedforward.anomalydetection.MNISTAnomalyExample"
"org.deeplearning4j.examples.feedforward.classification.MLPClassifierSaturn"
"org.deeplearning4j.examples.feedforward.classification.MLPClassifierMoon"
"org.deeplearning4j.examples.feedforward.classification.MLPClassifierLinear"
"org.deeplearning4j.examples.feedforward.mnist.MLPMnistTwoLayerExample"
"org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample"
"org.deeplearning4j.examples.nlp.paragraphvectors.ParagraphVectorsTextExample"
"org.deeplearning4j.examples.nlp.paragraphvectors.ParagraphVectorsClassifierExample"
"org.deeplearning4j.examples.nlp.word2vec.Word2VecUptrainingExample"
"org.deeplearning4j.examples.nlp.word2vec.Word2VecRawTextExample"
"org.deeplearning4j.examples.nlp.sequencevectors.SequenceVectorsTextExample"
"org.deeplearning4j.examples.nlp.tsne.TSNEStandardExample"
"org.deeplearning4j.examples.nlp.glove.GloVeExample"
"org.deeplearning4j.examples.recurrent.video.VideoClassificationExample"
"org.deeplearning4j.examples.recurrent.basic.BasicRNNExample"
"org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRNN"
"org.deeplearning4j.examples.recurrent.character.CompGraphLSTMExample"
"org.deeplearning4j.examples.recurrent.character.GravesLSTMCharModellingExample"
"org.deeplearning4j.examples.recurrent.seq2seq.AdditionRNN"
"org.deeplearning4j.examples.misc.earlystopping.EarlyStoppingMNIST"
"org.deeplearning4j.examples.unsupervised.deepbelief.DeepAutoEncoderExample"
"org.deeplearning4j.examples.arbiter.BasicHyperparameterOptimizationExample"
"org.deeplearning4j.examples.dataExamples.PreprocessNormalizerExample"
"org.deeplearning4j.examples.dataExamples.ImagePipelineExample"
"org.deeplearning4j.examples.dataExamples.CSVExample"
"org.deeplearning4j.examples.dataExamples.BasicCSVClassifier"
)

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
  java -cp dl4j-examples/target/dl4j-examples-0.4-rc0-SNAPSHOT-bin.jar "$i"

done

# You can access them using echo "${arr[0]}", "${arr[1]}" also


## declare an array variable
declare -a arr=(
"org/datavec/transform/basic/BasicDataVecExample"
"org/datavec/transform/analysis/IrisAnalysis"
"org/datavec/transform/debugging/PrintSchemasAtEachStep")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
  java -cp datavec-examples/target/datavec-examples-0.4-rc0-SNAPSHOT-bin.jar "$i"

done

# You can access them using echo "${arr[0]}", "${arr[1]}" also


#!/usr/bin/env bash


## declare an array variable
declare -a arr=("org.deeplearning4j.examples.feedforward.classification.MLPClassifierLinear" "org.deeplearning4j.examples.feedforward.classification.MLPClassifierMoon" "org.deeplearning4j.examples.feedforward.MLPClassifierSaturn" "org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample" "org.deeplearning4j.examples.feedforward.MLPMnistTwoLayerExample" "org.deeplearning4j.examples.feedforward.regression.RegressionSum" "org.deeplearning4j.examples.feedforward.regression.RegressionMathFunctions" "org.deeplearning4j.examples.misc.csv.CSVExample" "org.deeplearning4j.examples.misc.earlystopping.EarlyStoppingMNIST" "org.deeplearning4j.examples.recurrent.basic.BasicRNNExample" "org.deeplearning4j.examples.unsupervised.StackedAutoEncoderMnistExample" "org.deeplearning4j.examples.unsupervised.DBNMnistFullExample" "org.deeplearning4j.examples.unsupervised.deepbelief.DeepAutoEncoderExample" "org.deeplearning4j.examples.xor.XorExample")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
  java -cp target/deeplearning4j-examples-0.4-rc0-SNAPSHOT-bin.jar "$i"

done

# You can access them using echo "${arr[0]}", "${arr[1]}" also


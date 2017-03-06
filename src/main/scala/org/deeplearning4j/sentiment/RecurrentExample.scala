package org.deeplearning4j.sentiment

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.sentiment.iterator.SentimentExampleIterator
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.spark.api.Repartition

/**
  * Created by agibsonccc on 5/2/16.
  */
object RecurrentExample {

  def main(args: Array[String]) {

    val batchSize = 50     //Number of examples in each minibatch
    val vectorSize = 300   //Size of the word vectors. 300 in the Google News model
    val nEpochs = 5        //Number of epochs (full passes of training data) to train on
    val truncateReviewsToLength = 300  //Truncate reviews with length (# words) greater than this
    val WORD_VECTORS_PATH = "WORD_VECTORS_PATH"
    //Set up network configuration
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .updater(Updater.RMSPROP)
      .regularization(true).l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .learningRate(0.0018)
      .list()
      .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
        .activation("softsign").build())
      .layer(1, new RnnOutputLayer.Builder().activation("softmax")
        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
      .pretrain(false).backprop(true).build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    val sparkConf = new SparkConf()
    sparkConf.setMaster("local[" + 8 + "]")
    sparkConf.setAppName("LSTM_Char")
    sparkConf.set("AVERAGE_EACH_ITERATION", String.valueOf(true))
    val sc = new JavaSparkContext(sparkConf)

    val miniBatchSizePerWorker = 10
    val averagingFrequency = 5
    val numberOfAveragings = 3

    val tm = new ParameterAveragingTrainingMaster.Builder(5, 1)
      .averagingFrequency(averagingFrequency)
      .batchSizePerWorker(miniBatchSizePerWorker)
      .saveUpdater(true)
      .workerPrefetchNumBatches(0)
      .repartionData(Repartition.Always)
      .build();

    val sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm)
    val wordVectors: WordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false)
    val iterator = new SentimentExampleIterator("", wordVectors, 10, 100, true)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    sparkNetwork.fit(rdd)

  }
}

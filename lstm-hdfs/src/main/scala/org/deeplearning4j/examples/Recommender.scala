package org.deeplearning4j.examples

import java.util.concurrent.TimeUnit

import org.apache.commons.io.FilenameUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j

import org.deeplearning4j.examples.utils.ReflectionsHelper
import org.deeplearning4j.examples.utils.LoggingEarlyStoppingListener

import com.typesafe.scalalogging._
import org.slf4j.LoggerFactory

/**
 * @author: Ousmane A. Dia
 */
class Recommender(batchSize: Int = 50, featureSize: Int, nEpochs: Int, hiddenUnits: Int,
    miniBatchSizePerWorker: Int = 10, averagingFrequency: Int = 5, numberOfAveragings: Int = 3,
    learningRate: Double = 0.1, regularization: Double = 0.001, labelSize: Int,
    dataDirectory: String, sc: SparkContext) extends Serializable {

  val logger = LoggerFactory.getLogger(this.getClass)

  ReflectionsHelper.registerUrlTypes()

  val tm = new ParameterAveragingTrainingMaster.Builder(5, 1)
    .averagingFrequency(averagingFrequency)
    .batchSizePerWorker(miniBatchSizePerWorker)
    .saveUpdater(true)
    .workerPrefetchNumBatches(0)
    .repartionData(Repartition.Always)
    .build()

  Nd4j.getRandom().setSeed(12345)

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .regularization(true).l1(regularization)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(20.0)
    .momentum(0.3675)
    .dropOut(0.5)
    .updater(Updater.ADAGRAD)
    .iterations(1)
    .seed(12345)
    .graphBuilder()
    .addInputs("input")
    .addLayer("firstLayer", new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenUnits)
      .activation("relu").build(), "input")
    .addLayer("secondLayer", new GravesLSTM.Builder().nIn(hiddenUnits).nOut(hiddenUnits)
      .activation("relu").build(), "firstLayer")
    .addLayer("outputLayer", new RnnOutputLayer.Builder().activation("softmax")
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(hiddenUnits).nOut(labelSize).build(), "secondLayer")
    .setOutputs("outputLayer")
    .pretrain(false).backprop(true)
    .build()

  val sparkNet = new SparkComputationGraph(sc, conf, tm)
  sparkNet.setCollectTrainingStats(false)

  def trainUsingEarlyStopping(modelDir: String) = {

    val trainIter = new MDSIterator(sc.hadoopConfiguration, dataDirectory, batchSize, featureSize, labelSize, 1, 0)
    val testIter = new MDSIterator(sc.hadoopConfiguration, dataDirectory, batchSize, featureSize, labelSize, 1, 1)

    val saver = new LocalFileGraphSaver(FilenameUtils.concat(modelDir, "Recommender/"))

    val esConf = new EarlyStoppingConfiguration.Builder()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs))
      .evaluateEveryNEpochs(1)
      .scoreCalculator(new DataSetLossCalculatorCG(testIter, true))
      .saveLastModel(true)
      .modelSaver(saver)
      .build()

    val trainer = new EarlyStoppingGraphTrainer(esConf, sparkNet.getNetwork,
      trainIter, new LoggingEarlyStoppingListener())

    val result: EarlyStoppingResult[ComputationGraph] = trainer.fit()

    logger.info("Termination reason: " + result.getTerminationReason())
    logger.info("Termination details: " + result.getTerminationDetails())
    logger.info("Total epochs: " + result.getTotalEpochs())
    logger.info("Best epoch number: " + result.getBestModelEpoch())
    logger.info("Score at best epoch: " + result.getBestModelScore())

    result.getBestModel

  }

  def predict(graphPath: String, items: List[String]): RDD[List[(String, Double)]] = {
    val graph = ModelSerializer.restoreComputationGraph(graphPath)
    predict(graph, items)
  }

  def predict(graph: ComputationGraph, items: List[String]): RDD[List[(String, Double)]] = {

    val iterator: MDSIterator =
      new MDSIterator(sc.hadoopConfiguration, dataDirectory, batchSize, featureSize, labelSize, 1, 1)

    var itemScores = List[List[(String, Double)]]()
    while (iterator.hasNext) {
      val next = iterator.next
      val score = graph.output(next.getFeatures(0).dup()).apply(0).data().asDouble()
      itemScores = itemScores :+ items.zip(score)
    }
    sc.parallelize(itemScores)
  }

}

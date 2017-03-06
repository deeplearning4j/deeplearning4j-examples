package com.codor.alchemy.forecast

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.JavaSparkContext.toSparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import com.codor.alchemy.forecast.BaseParameterAveragingTrainingMaster
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

import com.codor.alchemy.forecast.utils.Logs
import org.apache.spark.SparkContext
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import com.codor.alchemy.forecast.utils.ReflectionsHelper

class ChurnPredictor(batchSize: Int = 50, featureSize: Int, nEpochs: Int, hiddenUnits: Int,
    miniBatchSizePerWorker: Int = 10, averagingFrequency: Int = 5, numberOfAveragings: Int = 3,
    learningRate: Double = 0.0018, l2Regularization: Double = 1e-5,
    dataDirectory: String, sc: SparkContext, exclude: RDD[String]) extends Serializable with Logs {
  
  //Nd4j.create(1)  
  ReflectionsHelper.registerUrlTypes()
  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nEpochs) //
    .updater(Updater.RMSPROP)
    .regularization(true).l2(l2Regularization)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0)
    .learningRate(learningRate)
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenUnits) // 200
      .activation("relu").build()) // softsign relu
    .layer(1, new RnnOutputLayer.Builder().activation("identity") // relu
      .lossFunction(LossFunctions.LossFunction.MSE).nIn(hiddenUnits).nOut(1).build()) // MCXENT 200
    .pretrain(false).backprop(true).build()

  val net = new MultiLayerNetwork(conf)
  net.init()

  //val sparkConf = new SparkConf()
  //sparkConf.setMaster("local[" + 8 + "]")
  //sparkConf.setAppName("ChurnPredictor")
  //sparkConf.set("AVERAGE_EACH_ITERATION", String.valueOf(true))
  //val sc = new JavaSparkContext(sparkConf)

  val tm = new BaseParameterAveragingTrainingMaster.Builder(5, 1) //ParameterAveragingTrainingMaster.Builder(5)
    .averagingFrequency(averagingFrequency)
    .batchSizePerWorker(miniBatchSizePerWorker)
    .saveUpdater(true)
    .workerPrefetchNumBatches(0)
    //.repartionData(Repartition.Always)
    .build();

  val sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm)
  var instanceIndices: List[String] = null

  def train() = {
   
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, true)
    
    import scala.collection.JavaConversions._
    iterator.setExclude(exclude.collect().toList)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    info("RDD Train: " + rdd.count() + " - Row: " + rdd.first().getFeatureMatrix.rows() 
	+ " - Col: " + rdd.first().getFeatureMatrix.columns() + " - LRow: " + rdd.first().getLabels.rows()
	+ " - LCol: " + rdd.first().getLabels.columns())
    
    info("Feature: " + rdd.first())

    instanceIndices = iterator.getAdvisors().toList

    sparkNetwork.fit(rdd)
  }

  def prepare(dataset: RDD[(String, Iterable[String])]) = {
    val predict = dataset.map(f => (f._1, f._2.filter { x => x.contains("201605") }))
      .map(f => (f._1, f._2.map { x => x.reverse.tail.reverse }))
    val training = dataset.subtract(predict)

    //training.map { x => sc.parallelize(x._2.toList).repartition(1).saveAsTextFile(dataDirectory + "train/") }
    //predict.map { x => sc.parallelize(x._2.toList).repartition(1).saveAsTextFile(dataDirectory + "test/") }
   
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    training.map(f => f._2.map { x => (f._1, x) }).flatMap(f => f)
      .toDF("key", "value").write.partitionBy("key").save("/user/odia/mackenzie/forecast/train")

    predict.map(f => f._2.map { x => (f._1, x) }).flatMap(f => f)
      .toDF("key", "value").write.partitionBy("key").save("/user/odia/mackenzie/forecast/test")    

    this
  }

  def predict(network: MultiLayerNetwork): RDD[(String, String)] = { //RDD[Array[Double]] = {
    import scala.collection.JavaConversions._
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, false)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)
 
    instanceIndices = iterator.getAdvisors().toList

    val indices = sc.parallelize(instanceIndices).zipWithIndex().map(f => (f._2, f._1))
 
    info("RDD Predict: " + rdd.count() + " - Row: " + rdd.first().getFeatureMatrix.rows() 
        + " - Col: " + rdd.first().getFeatureMatrix.columns() + " - LRow: " + rdd.first().getLabels.rows()
        + " - LCol: " + rdd.first().getLabels.columns())
 
   // info("Feature: " + rdd.first() + " - " + indices.count() + " - " + indices.first())

    val tm = new NewParameterAveragingTrainingMaster.Builder(5, 1)
      .averagingFrequency(averagingFrequency)
      .batchSizePerWorker(miniBatchSizePerWorker)
      .saveUpdater(true)
      .workerPrefetchNumBatches(0)
      .repartionData(Repartition.Always)
      .build();
 
    val trainedNetworkWrapper = new SparkDl4jMultiLayer(sc, network, tm)
      
    //val examples = MLLibUtil.fromDataSet(sc, rdd).rdd

    //val examples = rdd.map { x =>
    val examples = rdd.map { x =>
    //  var features: List[Array[Double]] = List()
    //  (0 until x.getFeatureMatrix.rows).foreach { i =>
    //    features = features :+ x.getFeatureMatrix.getRow(i).data().asDouble()
    //  }
     val labels = x.getLabels.data().asDouble()
    //  features.zip(labels).map(f => new LabeledPoint(f._2, Vectors.dense(f._1)))
    //}.flatMap { x => x }
    //  val labels = x.getLabels.data().asDouble().toList
    //  new LabeledPoint(labels(0), Vectors.dense(x.getFeatureMatrix.dup().data().asDouble()))
    //}

    //examples.map { point =>
      val vector = MLLibUtil.toVector(x.getFeatureMatrix.dup())
      //val score = trainedNetworkWrapper.predict(vector).toArray(0) // point.features
      val score = network.output(x.getFeatureMatrix.dup()).data().asDouble()//(0)
      //Array(labels(0), score) // point.label
      //Array(labels.mkString(","), score.mkString(","))
       (labels.mkString(","), score.mkString(","))
    }
    //indices.join(examples.zipWithIndex().map(f => (f._2, f._1))).map(f => (f._2._1, f._2._2.mkString(",")))
    examples 
 }
}

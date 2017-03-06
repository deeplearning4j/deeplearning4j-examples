package com.codor.alchemy.forecast

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import com.codor.alchemy.conf.Constants

object RecommenderDriver {
  def main(args: Array[String]): Unit = {
    new RecommenderDriver(args)
  }
}

class RecommenderDriver(args: Array[String]) extends Serializable with Logs {

  if (args.length < 1) {
    fatal("RecommenderDriver was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: ChurnPredictorDriver <configFile>")
    System.exit(-1)
  }

  var config: Config = null

  try {
    info("Loading driver configs from: " + args(0))
    config = ConfigFactory.parseResourcesAnySyntax(args(0))
  } catch {
    case e: Exception =>
      fatal("An error occurred when loading the configuration in the driver.", e)
      System.exit(-1)
  }

  val sc = new SparkContext(new SparkConf().setAppName("RecommenderDriver").setMaster("local[*]")) //TODO

  var items: List[String] = null
  var test: RDD[(Long, String)] = null

  var userRules: RDD[SmartStringArray] = null
  var itemInclusion: RDD[SmartStringArray] = null

  try {
    items = SmartStringArray.tableFromTextFile(Constants.TRAINING, ',', sc)
      .map { x => x(2) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList
    test = SmartStringArray.tableFromTextFile(Constants.TEST, ',', sc)
      .map { x => (x(0), x(0)) }.distinct().sortByKey(true).map(f => f._1).zipWithIndex()
      .map(f => (f._2, f._1))
 
    userRules = SmartStringArray.tableFromTextFile(Constants.USER_PREFERENCES, ',' , sc)
    itemInclusion = SmartStringArray.tableFromTextFile(Constants.ITEM_INCLUSION, ',', sc)

  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the instances in the driver.", e)
      System.exit(-1)
  }
  
  val predictor = new Recommender(batchSize = 1,
    featureSize = 138, // (113 + 25), // 657, (1 + 8) * 73 //520, // (64 + 1) * 8
    nEpochs = 35, //20,
    hiddenLayers = 50, //70, //100,
    miniBatchSizePerWorker = 30,
    averagingFrequency = 5,
    numberOfAveragings = 3,
    learningRate = 0.00005, // 0.0001, //0.0001
    l2Regularization = 0.00001, //0.00001,
    labelSize = 113,  // 111
    dataDirectory = Constants.DATA_DIR, sc)

  /*** Call the recommender train then predict methods ***/
  def predict() = {
     val net = predictor.trainUsingEarlyStopping(Constants.MODEL_SER_DIR)
     predictor.predict(net, items).zipWithIndex().map(f => f._1.map(x => (f._2, x))).flatMap(f => f)
  }

  /*** Load the model already built ***/
  def load() = {
     predictor.predict(Constants.MODEL_SER_FILE, items).zipWithIndex().map(f => f._1.map(x => (f._2, x))).flatMap(f => f)
  }

  val predictions = predict() // Or you can call load() if the model is already built and saved.
  
  //val predictions = load()

  val formatted = test.join(predictions).map(f => (f._2))
  formatted.saveAsTextFile(Constants.RESULTS)

  /*** Purging recommendations ***/
  def purge() = {

     var recos = formatted.map { x => (x._1, (x._2._1, x._2._2)) }.filter(f => f._2._1 != "").groupByKey().sortByKey(true)

     val userItemEligibility = userRules.map { x => (x(1), x(0)) }.groupByKey().sortByKey(true)
         .leftOuterJoin(itemInclusion.map { x => (x(0), x(1)) }.groupByKey().sortByKey(true))
         .map { f => f._2._1.map { x => (x, f._2._2.getOrElse(List())) } }.flatMap(f => f).sortByKey(true)
  
     val merged = userItemEligibility.join(recos).map(f => (f._1, f._2._1.toList, f._2._2.toList))
  
     val purgedRecos = merged.map(t => t._3.filter(x => t._2.isEmpty || t._2.contains(x._1))
           .map(f => (t._1, f._1, f._2))).flatMap(f => f).distinct()
     purgedRecos.saveAsTextFile(Constants.PURGED)
 }

}


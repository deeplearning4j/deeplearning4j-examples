package com.codor.alchemy.forecast

import scala.reflect.runtime.universe

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import com.codor.alchemy.forecast.utils.SmartStringArray

object ChurnPredictorDriver {
  def main(args: Array[String]): Unit = {
    new ChurnPredictorDriver(args)
  }
}

class ChurnPredictorDriver(args: Array[String]) extends Serializable with Logs {

  info("Starting Churn Prediction Driver Class")

  if (args.length < 1) {
    fatal("ChurnPredictorDriver was called with incorrect arguments: " +
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
      fatal("An error occurred when loading the configuration in the driver. Terminating driver process.", e)
      System.exit(-1)
  }

  var sc = new SparkContext(new SparkConf().setAppName("AtRiskPredictorDriver"))

  var instances: RDD[SmartStringArray] = null
  var exclude: RDD[String] = null

  try {
    instances = SmartStringArray.tableFromTextFile("/user/odia/mackenzie/forecast/data", ',', sc)
    exclude = SmartStringArray.tableFromTextFile("/user/odia/mackenzie/forecast/ignore", ',', sc)
	.map(f => f.toArray(0))
  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the instances in the driver. Terminating ...", e)
      System.exit(-1)
  }

  //val dataset = instances.map { x => (x.toArray.toList.head, x.toArray.toList.tail.mkString(",")) }
  //  .groupByKey().sortByKey(true)

  //val predict = dataset.map(f => (f._1, f._2.filter { x => x.contains("201605") }))
  //  .map(f => (f._1, f._2.map { x => x.reverse.tail.reverse }))
  //val training = dataset.subtract(predict)

  val predictor = new ChurnPredictor(batchSize = 1,
    featureSize = 71, //62,//71, //85,//86,
    nEpochs = 10, //5, //12, //10, //10, //7, //10, //15, //15, //40, //30, //30, //30, //20, //15, //5, //5, //10, //15, //20, //30, //10,
    hiddenUnits = 20, //26, //20, //32, //30, //30, //35, //30, //50, //50, //30, //40, //40, //40, //40, //45, //40, //30, //25,//35, //50,
    miniBatchSizePerWorker = 10,
    averagingFrequency = 5,
    numberOfAveragings = 3,
    learningRate = 0.003, //0.0018,
    l2Regularization = 1e-5,
    dataDirectory = "/user/odia/mackenzie/forecast/", 
    sc,
    exclude)

  val net = predictor.train()
  val predictions = predictor.predict(net).zipWithIndex().map(f => (f._2, f._1))

  predictions.saveAsTextFile("/user/odia/mackenzie/forecast/at_risk_rslts")
  //dataset.zipWithIndex().map(f => (f._2, f._1)).join(predictions)
  //  .map(f => f._2._1 + f._2._2.mkString(","))//.saveAsTextFile("/user/hduser/forecast/rslts")
  //  .saveAsTextFile("/user/odia/mackenzie/forecast/rslts")

  //val targets = predict.map(f => (f._1, f._2.toList.reverse.tail(0))).zipWithIndex().map(f => (f._2, f._1))
  
  //val targets = instances.map { x => (x.toArray.toList.head, x.toArray.toList.tail) }
  //  .filter(f => f._2.reverse.head == "201605").map(f => (f._1, f._2.reverse.tail(0)))
  //  .sortByKey(true).zipWithIndex().map(f => (f._2, f._1))

  //targets.join(predictions).map(f => f._2).saveAsTextFile("/user/odia/mackenzie/forecast/rslts")

  //targets.join(predictions).map(f => f._2._1 + "," + f._2._2.mkString(","))
  //  .saveAsTextFile("/user/odia/mackenzie/forecast/rslts")

  //sc.parallelize(predictor.instanceIndices).zipWithIndex().map(f => (f._2, f._1))
  //.join(predictions).map(f => f._2._1 + "," + f._2._2.mkString(","))
  //  .saveAsTextFile("/user/odia/mackenzie/forecast/rslts")

  val configs = net.getLayerWiseConfigurations.toJson()

  import scala.collection.JavaConverters._
  val params = scala.util.parsing.json.JSONObject(net.paramTable().asScala.toMap).toString()

  //val configuration = new Configuration();
  //configuration.set("fs.defaultFS", "hdfs://nn-galepartners.s3s.altiscale.com:8020")
  
  //val fs = FileSystem.get(configuration)  
  //val os = fs.create(new Path("/user/odia/mackenzie/forecast/at_risk_model"))
  //os.write((configs + "\n" + params.toString()).getBytes)
  
  info("CONFIGS:" + configs)
  info("PARAMS:" + params)
}

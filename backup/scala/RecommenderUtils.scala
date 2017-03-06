package com.codor.alchemy.forecast

import java.io.DataOutputStream
import java.nio.file.Files
import java.nio.file.Paths

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.nd4j.linalg.factory.Nd4j

import com.codor.alchemy.conf.Constants
import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory

object RecommenderUtils {
  def main(args: Array[String]): Unit = {
    new RecommenderUtils(args)
  }

  def saveModel(model: SparkComputationGraph, modelName: String) = {
    val fos = Files.newOutputStream(Paths.get(System.getProperty("user.home") + "/coefficients_" +
      modelName + ".bin"))
    val dos = new DataOutputStream(fos)
    Nd4j.write(model.getNetwork.params(), dos)
    dos.flush();
    dos.close();
  }

}

class RecommenderUtils(args: Array[String]) extends Serializable with Logs {

  info("Starting RecommenderUtils Class")

  if (args.length < 1) {
    fatal("RecommenderDriver was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: RecommenderDriver <configFile>")
    System.exit(-1)
  }
  val sc = new SparkContext(new SparkConf().setAppName("RecommenderUtils"))
  var config: Config = null

  try {
    info("Loading driver configs from: " + args(0))
    config = ConfigFactory.parseResourcesAnySyntax(args(0))
  } catch {
    case e: Exception =>
      fatal("An error occurred when loading the configuration in the driver.", e)
      System.exit(-1)
  }

  var recos: RDD[(String, Iterable[(String, Double)])] = null
  var userRules: RDD[(String, Iterable[String])] = null
  var itemInclusion: RDD[(String, Iterable[String])] = null

  try {
    recos = SmartStringArray.tableFromTextFile(Constants.RESULTS, ',', sc).map { x =>
      (x(0).replace("(", ""), (
        x(1).replace("(", ""), x(2).replace(")", "").toDouble))
    }.filter(f => f._2._1 != "").groupByKey().sortByKey(true)

    userRules = SmartStringArray.tableFromTextFile("", ',', sc).map { x => (x(1), x(0)) }
      .groupByKey().sortByKey(true)
    itemInclusion = SmartStringArray.tableFromTextFile("", ',', sc).map { x => (x(0), x(1)) }
      .groupByKey().sortByKey(true)

  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the instances in the driver.", e)
      System.exit(-1)
  }

  val userItemEligibility = userRules.leftOuterJoin(itemInclusion)
    .map { f => f._2._1.map { x => (x, f._2._2.getOrElse(List())) } }.flatMap(f => f).sortByKey(true)

  val purgedRecos = userItemEligibility.join(recos).map(f => (f._1, f._2._1.toList, f._2._2.toList))
    .map(t => t._3.filter(x => t._2.isEmpty || t._2.contains(x._1))
      .map(f => (t._1, f._1, f._2))).flatMap(f => f)

}

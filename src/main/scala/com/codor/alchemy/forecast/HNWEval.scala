package com.codor.alchemy.forecast

import scala.util.Random

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.SmartStringArray
import org.apache.spark.Accumulator

object HNWEval {
  def main(args: Array[String]): Unit = {
    new HighNetWorthEval(args)
  }
}

class HighNetWorthEval(args: Array[String]) extends Serializable with Logs {

  info("Starting HighNetWorthEval Driver Class")

  if (args.length < 1) {
    fatal("HighNetWorthEval was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: HighNetWorthEval <configFile>")
    System.exit(-1)
  }

  val sc = new SparkContext(new SparkConf().setAppName("MomentumEval"))

  var transactions: RDD[SmartStringArray] = null
  var allLeads: RDD[(String, String)] = null

  try {
    transactions = SmartStringArray.tableFromTextFile("", ',', sc)
    allLeads = SmartStringArray.tableFromTextFile("", ',', sc)
      .map { x => (x(0), x(1)) }
  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the transactions in the driver. ", e)
      System.exit(-1)
  }

  val rawData = transactions.map { x => (x(0), x(1), x.toArray.toList.tail.tail.map { x => x.toDouble }) }.persist()

  val dataset = rawData.map(f => (f._1, f._3(0), f._3(1))).sortBy(f => f._1).persist()

  val leads = allLeads.sortByKey(true).join(dataset.map(f => (f._1, (f._2, f._3)))
    .sortByKey(true)).keys.collect().toList

  val inputFromFile = SmartStringArray.tableFromTextFile("/user/odia/mackenzie/hnw/universe",
    ',', sc).map(x => (x(0), x(0))).sortByKey(true).join(rawData.map(f => (f._1, f._2)))
    .sortByKey(true).map(f => (f._1, f._2._2)).collect().toList
    
  var counter = 0
  val numSamples = 2000

  val input = population(leads, dataset)

  info("Coverage: " + input.size)
  
  sc.parallelize(input).saveAsTextFile("/user/odia/mackenzie/hnw/universe")
  
  evaluate(rawData)

  def evaluate(rawData: RDD[(String, String, List[Double])]) = {

    var sample: List[String] = List()
    (2 until 9).foreach { i =>

      val accum = sc.accumulator(0, "Eval Results")
      val netAccum = sc.accumulator(0.0d, "net trans")
      while (counter < numSamples) {
        val iter = input.iterator
        sample = sampling(iter, leads.size).toList
        eval(rawData.map(f => (f._1, f._2(1), f._2(i))), sample, leads,
          accum, netAccum)
        counter += 1
      }
      info("Results Eval" + i + " : " + accum.value / (1.0 * numSamples))
      info("Incrementals" + i + " : " + netAccum.value / (1.0 * numSamples))
    }

    info("Leads: " + leads.mkString(","))
    info("Sample List: " + sample.mkString(","))
  }

  info("Leads: " + leads.mkString(","))
  info("First Elem:" + dataset.first())

  def population(leads: List[String], dataset: RDD[(String, Double, Double)]) = {

    val max = dataset.map(f => (f._2, f._3)).max()
    val normalized = dataset.map(f => (f._1, f._2 / max._1, f._3 / max._2))

    val distances = normalized.cartesian(normalized).filter(f => f._1._1 != f._2._1).map { f =>
      (f._1._1, f._2._1, math.sqrt(math.pow((f._1._2 - f._2._2), 2) + math.pow((f._1._3 - f._2._3), 2)))
    }.filter(x => x._1 < x._2)

    val minST = mst(List(distances.first()._1), distances.keyBy(f => f._1).sortByKey(true).persist())

    val mstRDD = sc.parallelize(minST).zipWithIndex().map(f => (f._2, f._1))

    val pathRDD = mstRDD.join(mstRDD.map { f => (f._1 - 1, f._2) }).map(f => f._2)

    val radius = pathRDD.union(pathRDD.map(f => (f._2, f._1))).keyBy(f => f).sortByKey(true)
      .join(distances.map(f => ((f._1, f._2), f._3)).sortByKey(true)).map(f => f._2._2).max

    info("Radius: " + radius)

    distances.filter(f => leads.contains(f._1) && f._3 <= radius).map(f => f._2).collect().distinct.toList
  }

  def mst(nodes: List[String], dist: RDD[(String, (String, String, Double))]): List[String] = {
    var tree: List[String] = nodes

    info("Size: " + tree.size)

    if (tree.size == 5) { //dist.map(f => f._1).distinct().collect().size == tree.size) {
      nodes
    } else {
      val treeRDD = sc.parallelize(nodes).map { x => (x, x) }.sortByKey(true)
      val candidate = dist.join(treeRDD).map { f => f._2._1 }.filter(f =>
        !nodes.contains(f._2)).min._2
      tree = tree :+ candidate
      mst(tree, dist)
    }
  }

  def sampling(input: Iterator[String], k: Int): Array[String] = {
    val reservoir = new Array[String](k)
    var i = 0
    while (i < k && input.hasNext) {
      val item = input.next()
      reservoir(i) = item
      i += 1
    }

    if (i < k) {
      val trimReservoir = new Array[String](i)
      System.arraycopy(reservoir, 0, trimReservoir, 0, i)
      trimReservoir
    } else {
      var l = i.toLong
      while (input.hasNext) {
        val item = input.next()
        l += 1
        val replacementIndex = (Random.nextDouble() * l).toLong
        if (replacementIndex < k) {
          reservoir(replacementIndex.toInt) = item
        }
      }
      reservoir
    }
  }
  
  
   def populations(leads: List[String], dataset: RDD[(String, String, Double, Double)]) = {

    val max = dataset.map(f => (f._2, f._3, f._4)).max()
    val normalized = dataset.map(f => (f._1, f._3 / max._2, f._4 / max._3))

    val distances = normalized.cartesian(normalized).filter(f => f._1._1 != f._2._1).map { f =>
      (f._1._1, f._2._1, math.sqrt(math.pow((f._1._2 - f._2._2), 2) + math.pow((f._1._3 - f._2._3), 2)))
    }.filter(x => x._1 < x._2)

    val minST = mst(List(distances.first()._1), distances.keyBy(f => f._1).sortByKey(true).persist())

    val mstRDD = sc.parallelize(minST).zipWithIndex().map(f => (f._2, f._1))

    val pathRDD = mstRDD.join(mstRDD.map { f => (f._1 - 1, f._2) }).map(f => f._2)

    pathRDD.saveAsTextFile("/user/odia/mackenzie/hnw/mst")

    val radius = pathRDD.union(pathRDD.map(f => (f._2, f._1))).keyBy(f => f).sortByKey(true)
      .join(distances.map(f => ((f._1, f._2), f._3)).sortByKey(true)).map(f => f._2._2).max

    info("Radius: " + radius)

    val universe = distances.filter(f => leads.contains(f._1) && f._3 <= 2 * radius / 3).map(f => f._2)
      .collect().distinct.toList
    dataset.filter(f => universe.contains(f._1)).map(f => (f._1, f._2)).collect().distinct.toList
  }

  def eval(dataset: RDD[(String, Double, Double)], sample: List[String], leads: List[String],
    accum: Accumulator[Int], netAccum: Accumulator[Double]) = {
    val avgSample = dataset.filter(f => sample.contains(f._1)).map(f => (f._3)).sum / sample.size
    val avgLead = dataset.filter(f => leads.contains(f._1)).map(f => (f._3)).sum / leads.size
    if (avgSample < avgLead) accum += 1 else accum += 0
    netAccum += (avgLead - avgSample)
  }

}

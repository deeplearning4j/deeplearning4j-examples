package com.codor.alchemy.forecast

import com.codor.alchemy.forecast.utils.Logs
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.codor.alchemy.forecast.utils.SmartStringArray
import scala.reflect.ClassTag
import org.apache.spark.util.random.XORShiftRandom
import scala.util.Random
import org.apache.spark.Accumulator

object MomentumEval {
  def main(args: Array[String]): Unit = {
    new MomentumEval(args)
  }
}

class MomentumEval(args: Array[String]) extends Serializable with Logs {

  info("Starting MomentumEval Driver Class")

  if (args.length < 1) {
    fatal("MomentumEval was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: MomentumEval <configFile>")
    System.exit(-1)
  }

  val sc = new SparkContext(new SparkConf().setAppName("MomentumEval"))

  var transactions: RDD[SmartStringArray] = null
  var allLeads: RDD[(String, String)] = null

  try {
    transactions = SmartStringArray.tableFromTextFile("", ',', sc) // Constants.TRANSACTIONS
    allLeads = SmartStringArray.tableFromTextFile("", ',', sc).map { x => (x(0), x(1)) }
      
  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the transactions in the driver. ", e)
      System.exit(-1)
  }

  // groupname, license, priority, cal_year_month
  val rawData = transactions.map { x => (x(0), x(1), x(2), x(3), 
      x.toArray.toList.reverse.take(3).reverse.map { x => x.toDouble }) }.persist()

  def momentumEval(rawData: RDD[(String, String, String, String, List[Double])]) = {

    val filterBy = rawData.map(f => f._4).distinct().collect().toList.sortWith((a, b) => a< b)
    val filterIter = filterBy.iterator

    while (filterIter.hasNext) {
      
      val next = filterIter.next
      val dataset = rawData.filter(f => f._4 == next).map(f => (f._1, f._5(0), f._5(2)))

      dataset.count()
      
      //    val dataset = transactions.map { x => (x(0), x(1).toDouble, x(2), x(3).toDouble) }
      //      .filter(f => f._3 == "IIROC" && f._2 >= 0).map(f => (f._1, f._2, f._4))

      val leads = allLeads.sortByKey(true).join(dataset.map(f => (f._1, (f._2, f._3))).sortByKey(true))
        //.filter(f => f._2._1 == "IIROC" && f._2._2._1 >= 0)
        .keys.collect().toList

      var counter = 0
      val numSamples = 500

      val predictions = dataset.filter(f => leads.contains(f._1)).map(f => f._2).collect()
      val mean = predictions.sum / predictions.size
      val std = predictions.map { x => math.pow(x - mean, 2) }.sum / (predictions.size - 1)

      val input = population(dataset, mean, std).collect().map(f => f._1)

      val accum = sc.accumulator(0, "Eval Results")
      val netAccum = sc.accumulator(0.0d, "net trans")

      var sample: List[String] = List()
      while (counter < numSamples) {
        val iter = input.iterator
        sample = sampling(iter, leads.size).toList
        eval(dataset, sample, leads, accum, netAccum)
        counter += 1
      }

      info("Leads " + next + " : " + leads.mkString(","))
      info("Statistics " + next + " : " + mean + ", " + std)
      //info("Sample List: " + sample.mkString(","))
      info("Results Eval " + next + " : " + accum.value / (1.0 * numSamples))
      info("Incrementals " + next + ": " + netAccum.value / (1.0 * numSamples))
      
    }
  }

  def population(dataset: RDD[(String, Double, Double)], mean: Double, std: Double) = {
    dataset.map(x => (x._1, x._2)).filter(f => f._2 >= mean - 5 * math.sqrt(std) &&
      f._2 <= mean + 5 * math.sqrt(std))
  }

  def population(dataset: RDD[(String, Double, Double)]) = {
    val max = dataset.map(f => (f._2, f._3)).max()
    val normalized = dataset.map(f => (f._1, f._2 / max._1, f._3 / max._2))
    val dist = normalized.cartesian(normalized).filter(f => f._1._1 != f._2._1).map { f =>
      (f._1._1, f._2._1, math.sqrt(math.pow((f._1._2 - f._2._2), 2) +
        math.pow((f._1._3 - f._2._3), 2)))
    }.persist()

    mst(List(dist.first()._1), dist.keyBy(f => f._1).sortByKey(true))
  }

  def mst(nodes: List[String], dist: RDD[(String, (String, String, Double))]): List[String] = {
    var tree: List[String] = nodes

    info("Size:" + tree.size)

    if (dist.map(f => f._1).distinct().collect().size == nodes.size) {
      nodes
    } else {
      val treeRDD = sc.parallelize(nodes).map { x => (x, x) }.sortByKey(true)
      val candidate = dist.join(treeRDD).map { f => f._2._1 }.filter(f =>
        !nodes.contains(f._2)).min._2
      tree = tree :+ candidate
      mst(tree, dist)
    }
  }

  def population(leads: List[String], dataset: RDD[(String, Double, Double)]) = {

    val max = dataset.map(f => (f._2, f._3)).max()
    val normalized = dataset.map(f => (f._1, f._2 / max._1, f._3 / max._2))

    val distances = normalized.cartesian(normalized).filter(f => f._1._1 != f._2._1).map { f =>
      (f._1._1, f._2._1, math.sqrt(math.pow((f._1._2 - f._2._2), 2) +
        math.pow((f._1._3 - f._2._3), 2)))
    }.persist()

    val minST = mst(List(distances.first()._1), distances.keyBy(f => f._1).sortByKey(true))

    val mstRDD = sc.parallelize(minST).zipWithIndex().map(f => (f._2, f._1))

    val pathRDD = mstRDD.join(mstRDD.map { f => (f._1 - 1, f._2) }).map(f => f._2)
    val radius = pathRDD.union(pathRDD.map(f => (f._2, f._1))).keyBy(f => f).sortByKey(true)
      .join(distances.map(f => ((f._1, f._2), f._3))).map(f => f._2._2).max

    info("Radius:" + radius)
      
    distances.filter(f => leads.contains(f._1) && f._3 <= radius).map(f => f._2).collect().distinct.toList
  }
  
  def sampling(input: Iterator[String/*(String, Double)*/], k: Int): Array[String] = {
    val reservoir = new Array[String](k)
    var i = 0
    while (i < k && input.hasNext) {
      val item = input.next()
      reservoir(i) = item//._1
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
            reservoir(replacementIndex.toInt) = item//._1
        }
      }
      reservoir
    }
  }

  def eval(dataset: RDD[(String, Double, Double)], sample: List[String], leads: List[String], 
      accum: Accumulator[Int], netAccum: Accumulator[Double]) = {
    val avgSample = dataset.filter(f => sample.contains(f._1)).map(f => (f._3)).sum / sample.size
    val avgLead = dataset.filter(f => leads.contains(f._1)).map(f => (f._3)).sum / leads.size
    if (avgSample < avgLead) accum += 1 else accum += 0
    netAccum +=  (avgLead - avgSample)
  }

}
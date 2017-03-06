package com.codor.alchemy.forecast

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config

class UserItemFeaturer(config: Config, sc: SparkContext, nonulls: RDD[SmartStringArray]) extends Logs {

  val items = nonulls.map { x => x(1) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList
  val users = nonulls.map { x => x(0) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList

  /**
   * Featurizes the input data
   */
  def featurize(): RDD[(Long, (String, Iterable[(String, String, Double)]))] = {

    val riskScoreIndex = 2
    val riskScore = nonulls.map { t =>
      if (t(riskScoreIndex) == "NULL") -1 else
        t(riskScoreIndex).toDouble
    }.countByValue()

    val variables = List(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) // TODO: conf

    // RDD[((String, String), Iterable[(String, List[(Int, Double)])])]
    val tuples = nonulls.map { x => // keys: user, year_month
      ((x(0), x(13)), (x(1), variables.map { f =>
        (f, if (x(f) == "NULL") -1 else x(f.toInt).toDouble)
      }))
    }.groupByKey()

    // RDD[((String, String), Iterable[List[(Int, Double)]])]
    val discretized = tuples.map { f =>
      (f._1, f._2.map { x =>
        val index = x._2.map(t => t._1).indexOf(riskScoreIndex)
        val old = x._2(index)
        (x._1, x._2.updated(index, (riskScoreIndex,
          riskScore.getOrElse(old._2, 0L).toDouble)))
      })
    }.filter(f => users.contains(f._1._1)).sortByKey(true)

    val normalized = discretized.map { x => // discretized tuples
      (x._1, x._2.map { t =>
        t._2.map { a => (a._1, (t._1, a._2)) }
      }.flatMap(f => f).groupBy(f => f._1))
    }.map { x =>
      (x._1, x._2.map { y =>
        val values = y._2.map(a => a._2)
        (y._1, values.map { x => (x._1, math.abs(x._2) / (values.map(f => f._2).sum + 0.0001)) })
      })
    }.sortByKey(true)

    //user index, (user, Iterable[(item, yearmon, Double)])
    discretized.map { t => (t._1, t._2.map(f => f._1)) }.join(normalized)
      .map(f => (f._2._2.values.flatMap(f => f).map(t => (f._1._1, (t._1, f._1._2, t._2)))))
      .flatMap(f => f).groupByKey().map(f => (users.indexOf(f._1).toLong, f))
  }

  def createList(size: Int) : List[Double] = {
   var list = Array[Double]()
   (0 until size).foreach { i => list = list :+ 0.0 }
   list.toList
  }
  
  /**
   * Generates the embeddings for every (advisor, yearmon) combination.
   */
  def getEmbeddings(factors: RDD[(Long, Array[Double])]): RDD[((String, Int), (List[Double], List[Double]))] = {

    val features = featurize()

    val itemFactors = factors.filter(f => f._1 >= users.size && f._1 < users.size +
      items.size).map(f => (f._1, f._2)).collect().map(f => f._2)

    //user index, (user, Iterable[(item, yearmon, Double)], fmFactor)
    val userFactors = features.join(factors).map(f => (f._2._1._1, f._2._1._2, f._2._2))

    userFactors.map { t =>
      var label = createList(items.size)
      val featureMap = t._2.map(f => (f._1, f._3)).groupBy(f => f._1).map(f => (f._1,
        f._2.map(t => t._2).toList))
      featureMap.keysIterator.foreach { x => label = label.updated(items.indexOf(x), 1.0) }

      t._2.map { u =>
        val size = itemFactors(0).size
        //var embedding = List.fill(111 * (1 + size))(0.0)
        var embedding = createList(items.size * (1 + size))
        (0 until t._3.size).foreach { i => embedding = embedding.updated(i, t._3(i)) }

        val index = itemFactors.indexOf(u._1)
        ((size * (1 + index)) until (size * (1 + index) + size)).foreach { i =>
          embedding = embedding.updated(i, itemFactors(index)(i))
        }
        embedding = embedding ::: featureMap.getOrElse(u._1, List())
        // yearmon converted to int to be able to sort in decreasing order
        ((t._1, u._2.toInt), (embedding, label))
      }
    }.flatMap(f => f)
  }
}

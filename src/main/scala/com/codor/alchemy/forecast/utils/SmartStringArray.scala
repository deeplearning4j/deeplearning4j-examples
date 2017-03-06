package com.codor.alchemy.forecast.utils

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

object SmartStringArray  extends Serializable with Logs {
  
   def groupTableFromTextFileBy(location: String, columnSeparator: Char, columnToGroupBy: Int, sc: SparkContext): 
        RDD[(String, ArrayBuffer[SmartStringArray])] = {
        tableFromTextFile(location, columnSeparator, sc)
          .map(x => {
              var ab: ArrayBuffer[SmartStringArray] = new ArrayBuffer[SmartStringArray](2)
              ab += x
              (x(columnToGroupBy), ab)
          })
          .reduceByKey((a, b) => {
              a ++= b
              a
          })

   }

   def keyTableFromTextFileBy(location: String, columnSeparator: Char, columnsToGroupBy: Array[Int], sc: SparkContext): 
        RDD[(String, ArrayBuffer[SmartStringArray])] = {
        tableFromTextFile(location, columnSeparator, sc)
          .map(x => {
              var ab: ArrayBuffer[SmartStringArray] = new ArrayBuffer[SmartStringArray]
              ab += x
              (columnsToGroupBy.map(y => x(y)).reduce((a,b) => a + "," + b), ab)
          })

   }

   def tableFromTextFile(location: String, columnSeparator: Char, sc: SparkContext): RDD[SmartStringArray] = {
        sc.textFile(location)
          .map(x => new SmartStringArray(x, columnSeparator))
   }

   def tableFromTextFile(location: String, columnSeparator: Char, sc: SparkContext, startDate: Long, endDate: Long): RDD[SmartStringArray] = {
       val fs = FileSystem.get(new Configuration())
       val status = fs.listStatus(new Path(location))
       val toUse  = status.filter(x => {
                                     val thePath = x.getPath.getName
                                     try{
                                         val timeStamp = thePath.toLong
                                         if(timeStamp >= startDate && timeStamp <= endDate)
                                             true
                                         else
                                             false
                                     }catch{
                                         case e: Exception => false
                                     }
                                 })
                           .map(x => x.getPath.toString)
                           .reduce((a,b) => (a + "," + b))
        sc.textFile(toUse)
          .map(x => new SmartStringArray(x, columnSeparator))
 
   }

   def tableFromLatestTextFileInDirectory(location: String, columnSeparator: Char, sc: SparkContext): RDD[SmartStringArray] = {
       val fs = FileSystem.get(new Configuration())
       val status = fs.listStatus(new Path(location))
       val last   = (status.map(x => (x, x.getPath.getName.toLong))
                          .toArray
                          .sortBy(-_._2))
       val last2 = last(0)
                          
       tableFromTextFile(last2._1.getPath.toString, columnSeparator, sc) 
   }
}

class SmartStringArray(source: String, val separator: Char) extends Serializable{


   private val data    = {if(source == null){null}else{source.toCharArray()}}
   private val offsets = new ArrayBuffer[Int]   

   if (source != null){
       offsets += 0
       for (i <- 0 until data.length)
           if (data(i) == separator)
               offsets += ((i + 1))
   }

   def apply(i: Int): String = {
       if(i >= offsets.length || i < 0){
           throw new ArrayIndexOutOfBoundsException("Requested Index of " + i + " was invalid " +
                                                    "for SmartStringArray of length " + offsets.length + ".")
       }else if(i != offsets.length - 1){
           new String(data, offsets(i), offsets(i + 1) - offsets(i) - 1)
       }else{
           new String(data, offsets(i), ((data.length - 1) - offsets(i)) + 1) 
       }
   }

   def toArray: Array[String] = { (new String(data)).split(separator) }

   def length: Int = offsets.length

   def assertLength(expected: Int) = { length == expected}
 
   override def toString(): String = { new String(data) } 
}
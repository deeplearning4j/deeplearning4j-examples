package com.codor.alchemy.forecast.utils

import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat
import org.apache.hadoop.io.NullWritable
import org.apache.spark.HashPartitioner


/**
 * @author: Ousmane A. Dia
 */
  class RDDMultipleTextOutputFormat extends MultipleTextOutputFormat[Any, Any] {

    override def generateActualKey(key: Any, value: Any): Any =
      NullWritable.get()

    override def generateFileNameForKeyValue(key: Any, value: Any, name: String): String =
      key.asInstanceOf[String]
  }

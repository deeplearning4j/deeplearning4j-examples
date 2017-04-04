package com.codor.alchemy.conf

import com.codor.alchemy.forecast.utils.Logs

/**
 * @author: Ousmane A. Dia
 */

object Constants extends Serializable with Logs {
  
  val TIME_INDEX = 1 //-1 (if time is not in the data, strictly positive number if in)
  
  val START_SEQ = 201605

  val END_SEQ = 201608
  
}


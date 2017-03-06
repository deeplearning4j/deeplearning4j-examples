package com.codor.alchemy.conf

import com.codor.alchemy.forecast.utils.Logs

case class ModelConf(holdout: Double, startSeq: Int, endSeq: Int)

object Configs extends Serializable with Logs {

  val MODEL_CONF = Map(("train", ModelConf(0.8, 201603, 201611)), ("test", ModelConf(0.2, 201604, 201612)))
  
  val TIME_INDEX = 1 //-1 (if time is not in the data, strictly positive number if in)

  val START_SEQ = 201603

  val END_SEQ = 201611 // keep a month out for testing

  val MODEL_SER_DIR = "/mnt/ebs0/odia/mackenzie"

  val MODEL_SER_FILE = "/mnt/ebs0/odia/mackenzie/Recommender/bestGraph.bin"

  val FM_FACTORS = "/user/odia/mackenzie/product_recos/fm_interactions"

  val TRANSACTIONS = "/user/odia/mackenzie/product_recos/transactions_201604_201609"

  val EMBEDDINGS = "/user/odia/mackenzie/lstm_recos/embeddings"

  val TRAINING = "/user/odia/mackenzie/product_recos/mack_product_recos_lstm_data_201602_201612"
  //"/user/odia/mackenzie/product_recos/training_data_fm_201604_201608_en_purged"

  val TEST = "/user/odia/mackenzie/product_recos/training_data_fm_201609"

  val RESULTS = "/user/odia/mackenzie/lstm_recos/rslts"

  val DATA_DIR = "/user/odia/mackenzie/december_run/"
  // "/user/odia/mackenzie/lstm_recos/"

  val USER_PREFERENCES = "/user/odia/mackenzie/product_recos/advisor_companies"

  val ITEM_INCLUSION = "/user/odia/mackenzie/product_recos/fund_inclusion"

  val PURGED = "/user/odia/mackenzie/lstm_recos/final"
}
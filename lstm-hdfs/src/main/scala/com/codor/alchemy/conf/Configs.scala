package com.codor.alchemy.conf

import com.codor.alchemy.forecast.utils.Logs

case class ModelConf(holdout: Double, startSeq: Int, endSeq: Int)

object Configs extends Serializable with Logs {

  val MODEL_CONF = Map(("train", ModelConf(0.8, 201603, 201611)), ("test", ModelConf(0.2, 201604, 201612)))
  
  val TIME_INDEX = 1 //-1 (if time is not in the data, strictly positive number if in)

  val START_SEQ = 201603

  val END_SEQ = 201611 // keep a month out for testing

  val MODEL_SER_DIR = "/mnt/ebs0/home/model_dir"

  val MODEL_SER_FILE = "/mnt/ebs0/home/model_dir/Recommender/bestGraph.bin"

  val FM_FACTORS = "/user/home/model_dir/product_recos/fm_interactions"

  val TRANSACTIONS = "/user/home/model_dir/product_recos/transactions_201604_201609"

  val EMBEDDINGS = "/user/home/model_dir/lstm_recos/embeddings"

  val TRAINING = "/user/home/model_dir/product_recos/training"

  val TEST = "/user/home/model_dir/product_recos/training_data_fm_201609"

  val RESULTS = "/user/home/model_dir/lstm_recos/rslts"

  val DATA_DIR = "/user/home/model_dir/lstm_recos/"

  val USER_PREFERENCES = "/user/home/model_dir/product_recos/users_prefs"

  val ITEM_INCLUSION = "/user/home/model_dir/product_recos/fund_inclusion"

  val PURGED = "/user/home/model_dir/lstm_recos/final"
}

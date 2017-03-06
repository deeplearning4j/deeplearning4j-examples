package com.codor.alchemy.conf

import com.codor.alchemy.forecast.utils.Logs

object Constants extends Serializable with Logs {
  
  val TIME_INDEX = 1 //-1 (if time is not in the data, strictly positive number if in)
  
  val START_SEQ = 201605

  val END_SEQ = 201608

  val MODEL_SER_DIR = "/mnt/ebs0/myhome/myfolder"

  val MODEL_SER_FILE = "/mnt/ebs0/myhome/myfolder/Recommender/bestGraph.bin"

  val FM_FACTORS = "/user/myhome/myfolder/product_recos/fm_interactions"

  val TRANSACTIONS = "/user/myhome/myfolder/product_recos/transactions_201604_201609"

  val EMBEDDINGS = "/user/myhome/myfolder/lstm_recos/embeddings"

  val TRAINING = "/user/myhome/myfolder/product_recos/training_data_fm_201604_201608_en_purged"

  val TEST = "/user/myhome/myfolder/product_recos/training_data_fm_201609"

  val RESULTS = "/user/myhome/myfolder/lstm_recos/rslts"

  val DATA_DIR = "/user/myhome/myfolder/lstm_recos/"

  val USER_PREFERENCES = "/user/myhome/myfolder/product_recos/advisor_companies"

  val ITEM_INCLUSION = "/user/myhome/myfolder/product_recos/fund_inclusion"

  val PURGED = "/user/myhome/myfolder/lstm_recos/final"
  
  val CORRELATIONS = "/user/myhome/myfolder/product_recos/recommendations"
  
}


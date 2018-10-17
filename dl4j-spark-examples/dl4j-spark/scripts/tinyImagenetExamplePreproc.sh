#!/usr/bin/env bash

#REQUIRED arguments: Set these before running
MASTER_IP=...                                  #IP address of master node
LOCALSAVEDIR=/Users/susaneraly/Desktop/tinyimageneteg

#Optional argumenst: Set these only if the defaults aren't suitable
#SPARKSUBMIT=/opt/spark/bin/spark-submit
SPARKSUBMIT=spark-submit
EXECUTOR_MEMORY=12G
MASTER_PORT=7077                                #Port for the spark master. Default is 7077
MINIBATCH=32                                    #Minibatch size for preprocessed datasets

#Other variables. Don't modify these
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/../target/dl4j-spark-1.0.0-beta2-bin.jar


#   --master spark://${MASTER_IP}:${MASTER_PORT}
#   --deploy-mode client
CMD="${SPARKSUBMIT}
    --class org.deeplearning4j.tinyimagenet.PreprocessLocal
    --master local
    --executor-memory ${EXECUTOR_MEMORY}
    ${JARFILE}
	--localSaveDir ${LOCALSAVEDIR}
    --batchSize ${MINIBATCH}
    "

eval $CMD

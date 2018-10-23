#!/usr/bin/env bash

#This script assumes a very small spark standalone cluster, 3 machines with 2 GPUs each. 
#Different settings may be required for different cluster managers (YARN, Mesos, etc)

SPARKSUBMIT=/opt/spark/bin/spark-submit
EXECUTOR_MEMORY=12G
MASTER_PORT=7077                                # Port for the spark master. Default is 7077
MINIBATCH=32                                    # Minibatch size for preprocessed datasets
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/../target/dl4j-spark-1.0.0-beta2-bin.jar

#Memory
OFFHEAP_MEM_JAVACPP=20G         # Maximum amount of off-heap memory
OFFHEAP_JAVACPP_MAX_PHYS=32G    # Maximum amount of on-heap PLUS off-heap memory

CMD="${SPARKSUBMIT} --class org.deeplearning4j.tinyimagenet.TrainSpark
    --master spark://10.0.2.27:7077
    --executor-memory ${EXECUTOR_MEMORY}
        --deploy-mode client
        --conf 'spark.locality.wait=0'
        --conf 'spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS'
        --driver-java-options '-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS'
    ${JARFILE}
        --dataPath wasbs://tinyimagenet@sparkbench.blob.core.windows.net/
        --masterIP 10.0.2.27
        --networkMask 10.0.0.0/16
        --numNodes 3
        --numWorkersPerNode 2
    "

eval $CMD

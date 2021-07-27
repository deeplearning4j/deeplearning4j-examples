#!/usr/bin/env bash

################################################################################
# Copyright (c) 2020 Konduit K.K.
# Copyright (c) 2015-2019 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

# Example script for org.deeplearning4j.examples.tinyimagenet.TrainSpark

#This script assumes a very small spark standalone cluster, 5 machines with 2 GPUs each.
#Different settings may be required for different cluster managers (YARN, Mesos, etc)

# You will need to set:
# 1. Spark master
# 2. Data path (see PreprocessLocal/PreprocessSpark)
# 3.

SPARKSUBMIT=/opt/spark/bin/spark-submit
EXECUTOR_MEMORY=12G
MASTER_PORT=7077                                # Port for the spark master. Default is 7077
MINIBATCH=32                                    # Minibatch size for preprocessed datasets
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/../target/dl4j-spark-1.0.0-M1.1-bin.jar

#Memory
OFFHEAP_MEM_JAVACPP=20G         # Maximum amount of off-heap memory
OFFHEAP_JAVACPP_MAX_PHYS=32G    # Maximum amount of on-heap PLUS off-heap memory

CMD="${SPARKSUBMIT} --class org.deeplearning4j.examples.tinyimagenet.TrainSpark
    --master spark://10.0.2.27:7077
    --executor-memory ${EXECUTOR_MEMORY}
        --deploy-mode client
        --conf 'spark.locality.wait=0'
        --conf 'spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS'
        --driver-java-options '-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS'
    ${JARFILE}
        --dataPath hdfs://YOUR_TINY_IMAGENET_DATA_PATH
        --masterIP <SET MASTER IP>
        --networkMask <SET NETWORK MASK>
        --numNodes 5
        --numWorkersPerNode 2
    "

eval $CMD

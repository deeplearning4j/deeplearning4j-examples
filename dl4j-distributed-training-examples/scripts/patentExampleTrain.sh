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

#REQUIRED arguments: Set these before running
MASTER_IP=...                                   #IP address of master node
NETWORK_MASK=...                                #Network maske For example, 10.0.0.0/16
AZURE_STORAGE_ACCT=...                          #Azure storage account name. 3-24 chars, lowercase alphanumeric only
AZURE_STORAGE_ACCT_KEY=...                      #Azure storage account key. Should be approx 88 characters, usually ends with "=="
AZURE_CONTAINER_PREPROC=patentpreproc           #Name of the container for preprocessed data. Should match value set in scripts/patentExamplePreproc.sh
LOCAL_SAVE_DIR=...                              #Local address for writing results
NUM_NODES=...                                   #Number of nodes in the cluster


#Optional argumenst: Set these only if the defaults aren't suitable
SPARKSUBMIT=/opt/spark/bin/spark-submit
MASTER_PORT=7077                                #Port for the spark master. Default is 7077
MINIBATCH=32                                    #Minibatch size for preprocessed datasets
    # For memory config, see https://deeplearning4j.konduit.ai/config/config-memory
JAVA_HEAP_MEM=10G
OFFHEAP_MEM_JAVACPP=20G
OFFHEAP_JAVACPP_MAX_PHYS=30G
    #Aeron buffer. Default of 32MB is fine for this example. Larger neural nets may require larger: 67108864 or 134217728. Must be a power of 2 exactly
AERON_BUFFER=33554432

#Other variables. Don't modify these
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/../target/dl4j-spark-1.0.0-beta7-bin.jar
AZURE_ACCT=fs.azure.account.key.${AZURE_STORAGE_ACCT}.blob.core.windows.net


CMD="${SPARKSUBMIT}
    --class org.deeplearning4j.examples.patent.TrainPatentClassifier
    --conf 'spark.hadoop.${AZURE_ACCT}=${AZURE_STORAGE_ACCT_KEY}'
    --conf spark.locality.wait=0
    --conf 'spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --conf 'spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --driver-java-options '-Dorg.bytedeco.javacpp.maxbytes=$OFFHEAP_MEM_JAVACPP -Dorg.bytedeco.javacpp.maxphysicalbytes=$OFFHEAP_JAVACPP_MAX_PHYS -Daeron.term.buffer.length=${AERON_BUFFER}'
    --master spark://${MASTER_IP}:${MASTER_PORT}
    --deploy-mode client
    --executor-memory ${JAVA_HEAP_MEM}
    ${JARFILE}
    --numNodes ${NUM_NODES}
    --azureStorageAcct ${AZURE_STORAGE_ACCT}
    --azureContainerPreproc ${AZURE_CONTAINER_PREPROC}
    --outputPath ${LOCAL_SAVE_DIR}
    --masterIP ${MASTER_IP}
    --networkMask ${NETWORK_MASK}
    --minibatch ${MINIBATCH}
    "

eval $CMD

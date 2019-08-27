#!/usr/bin/env bash

#REQUIRED arguments: Set these before running
MASTER_IP=...                                   #IP address of master node
AZURE_STORAGE_ACCT=...                          #Azure storage account name. 3-24 chars, lowercase alphanumeric only
AZURE_STORAGE_ACCT_KEY=...                      #Azure storage account key. Should be approx 88 characters, usually ends with "=="
AZURE_CONTAINER_ZIPS=patentzips                 #Name of the container to store the raw ZIP files. 3-63 chars, lowercase alphanumeric and dash only
AZURE_CONTAINER_PREPROC=patentpreproc           #Name of the container for storing preprocessed data. 3-63 chars, lowercase alphanumeric and dash only

#Optional argumenst: Set these only if the defaults aren't suitable
SPARKSUBMIT=/opt/spark/bin/spark-submit
EXECUTOR_MEMORY=24G
MASTER_PORT=7077                                #Port for the spark master. Default is 7077
MINIBATCH=32                                    #Minibatch size for preprocessed datasets

#Other variables. Don't modify these
SCRIPTDIR=$(dirname "$0")
JARFILE=${SCRIPTDIR}/../target/dl4j-spark-1.0.0-beta4-bin.jar
AZURE_ACCT=fs.azure.account.key.${AZURE_STORAGE_ACCT}.blob.core.windows.net


CMD="${SPARKSUBMIT}
    --class org.deeplearning4j.patent.DownloadPreprocessPatents
    --conf 'spark.hadoop.${AZURE_ACCT}=${AZURE_STORAGE_ACCT_KEY}'
    --master spark://${MASTER_IP}:${MASTER_PORT}
    --deploy-mode client
    --executor-memory ${EXECUTOR_MEMORY}
    ${JARFILE}
    --azureStorageAcct ${AZURE_STORAGE_ACCT}
    --azureContainerZips ${AZURE_CONTAINER_ZIPS}
    --azureContainerPreproc ${AZURE_CONTAINER_PREPROC}
    --minibatch ${MINIBATCH}
    "

eval $CMD

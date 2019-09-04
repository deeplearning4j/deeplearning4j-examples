
# Patent Spark Example

This example is set up to run distributed training of a neural network document classifier on a large real-world dataset.
It is also used as a benchmark to demonstrate convergence of Spark training.
Convergence is demonstrated by periodically saving copies of the neural network parameters, and then later evaluating them
to produce an "accuracy vs. time" chart.

**Dataset**: United States Patent and Trademark Office (USPTO) patents - 1976 to 2018

**Task**: Document classification - predict the primary classification of the patent from the patent description

Number of classes: 398

Number of documents/examples (after preprocessing): approx. 5.7 million (training set) plus approx. 170000 (test set)

Dataset size: approx. 86 GB (zip format), 464 GB raw text. Note the example performs preprocessing from the compressed ZIP format.
Requires an additional 20GB of storage space for preprocessing  

**Neural Network**: a CNN classifier for text classification. Approximately 600,000 parameters


## Requirements for Example

This example is set up for the following environment:
1. Execution on a Spark cluster
2. Cluster should have access to Azure storage (blob storage, etc)

Before running this example, you should have set up:

1. A Spark cluster that you can submit tasks to using Spark submit
2. An Azure storage account for storing data
3. Access key for the your Azure storage account


**Finding your Azure storage access key**: This can be accessed via the Azure web portal.
Simply navigate to your storage account, and select the "Access keys" section.
Your key should be approximately 88 characters long, and will usually end with the characters "==".

*IMPORTANT: This account key protects access to your Azure storage account and files. It should not be shared publicly.*

## Setup and Running

There are two parts to this example. You will first need to run preprocessing of the data.
After this preprocessing has been completed, you can run training.

### Stage 1: Clone and Build the Project

If you have not done so already, you should clone and build the example repository.
You can do this on a machine with access to spark-submit for your cluster, or on your
local machine and simply copy the uber-jar from your local machine to a remote machine
with access to spark-submit.

**First**: clone and build the project. To do this, you can use the following:

```
#If required: git clone this repo
cd dl4j-examples/dl4j-spark-examples/dl4j-spark
mvn package -DskipTests
#Or, if required: mvn clean package -Dspark.version=2.1.0 -Ddl4j.spark.version=1.0.0-beta4_spark_2 -Ddatavec.spark.version=1.0.0-beta4_spark_2
```

### Stage 2: Run Data Download and Preprocessing

**First**: modify the launch scripts.
Open ```dl4j-examples/dl4j-spark-examples/dl4j-spark/scripts/patentExampleProproc.sh```` and edit the following lines:

```
MASTER_IP=...
AZURE_STORAGE_ACCT=...
AZURE_STORAGE_ACCT_KEY=...
AZURE_CONTAINER_ZIPS=patentzips
AZURE_CONTAINER_PREPROC=patentExamplePreproc 
```

Note that some clusters may have the master already configured.
If this is the case, you can remove the ```--master spark://${MASTER_IP}:${MASTER_PORT}``` line.

The ```AZURE_STORAGE_ACCT``` variable should be set to the name of the storage account to use.

For example, if your storage account URLs are of the form ```https://mydata.blob.core.windows.net/```
then your storage account name is ```mydata```.

The ```AZURE_STORAGE_ACCT_KEY``` variable should be set to the key for the storage account set. See
earlier in the readme for finding this.

The ```AZURE_CONTAINER_ZIPS``` and ```AZURE_CONTAINER_PREPROC``` values may be left as
the defaults, or may be set to the name of the containers (in the specified storage) account
to use for storing the original patent zip files and the preprocessed data files respectively.
These can in principle be set to any valid container name, as long as the training script
is pointed to the same value for ```AZURE_CONTAINER_PREPROC```.

**Alternatively to setting storage account**

You can set the storage account credentials in your Hadoop core-site.xml file. See "Configuring Credentials" in this guide for details: [https://hadoop.apache.org/docs/current/hadoop-azure/index.html](https://hadoop.apache.org/docs/current/hadoop-azure/index.html)
 

**Second: Run the Script**

If everything is set correctly, the preprocessing script should now be runnable.

**End result of preprocessing**

After preprocessing is complete, you will have:

1. A copy of the USPTO patent (text only dataset, 1976 to 2018, 86GB) ZIP files at:
    1. For Spark access: ```wasbs://AZURE_CONTAINER_ZIPS@AZURE_STORAGE_ACCT.blob.core.windows.net/```
    2. For HTTP access (if enabled): ```https://AZURE_STORAGE_ACCT.blob.core.windows.net/AZURE_CONTAINER_ZIPS/```
2. Preprocessed training and test data (with default sequence length of 1000 and minibatch size of 32)
   1. For Spark access: ```wasbs://AZURE_CONTAINER_PREPROC@AZURE_STORAGE_ACCT.blob.core.windows.net/seqLength1000_mb32/```
   2. For HTTP access (if enabled): ```https://AZURE_STORAGE_ACCT.blob.core.windows.net/AZURE_CONTAINER_PREPROC/```  

Note that the preprocessed directory will have ```train``` and ```test``` subdirectories.
The format of the files in those train/test directories is a custom format designed to be loaded
by the ```LoadDataSetFunction``` class. The content is simply a list of integers - word indices
according to the Google News 300 word vectors vocabulary, with label values signified by a negative
integer value. These integers will be used to load the corresponding word vectors "on the fly" just
before training.

The reason for this format is to save space: a naive implementation would simply create ND4J/DL4J
DataSet objects with the dense (length 300) word vectors. However, this approach would take in excess
of 5 TB of space for a dataset of this size, compared around 20GB for the custom integer representation.


### Stage 3: Run Training

**First**: Modify the training script

Set the following required arguments to the same values used for the preprocessing script:
```
MASTER_IP=...
AZURE_STORAGE_ACCT=...
AZURE_STORAGE_ACCT_KEY=...
AZURE_CONTAINER_PREPROC=patentExamplePreproc 
```

The following configuration options also need to be set:
```
NETWORK_MASK
LOCAL_SAVE_DIR
```

Your network mask should be set to the network used for spark communication. For example, [10.0.0.0/16]
See the following links for further details:
* [DL4J Distributed Training - Netmask](https://deeplearning4j.org/distributed#netmask)
* [How to Find the IP Address, Subnet Mask & Gateway of a Computer](https://yourbusiness.azcentral.com/ip-address-subnet-mask-gateway-computer-14563.html)
* [What is a Subnet Mask](https://www.iplocation.net/subnet-mask)

```LOCAL_SAVE_DIR``` should be set to a writable directory on the driver's local file system for writing output.

**Second**: Run the training script

Upon completion, you will have the following files written to the directory specified by ```LOCAL_SAVE_DIR```:

* Full network checkpoints under ```LOCAL_SAVE_DIR/nets``` (including configuration, updater, etc)
* Parameter checkpoints under ```LOCAL_SAVE_DIR/paramSnapshots```
* Parameter checkpoint evaluation results under ```LOCAL_SAVE_DIR/evaluation_test```

The results will also be logged to the console.

### Step 1: Set up dependencies

This branch (spark_snapshot_examples) of the repo comes preconfigured for snapshots, plus CUDA 9.2 with CuDNN.
It assumes that the CUDA toolkit plus CuDNN is available on the master and workers. If this is correct, skip to step 2.

To use CPU instead: uncomment lines 65, comment out lines 73 to 84 [here](https://github.com/deeplearning4j/dl4j-examples/blob/spark_snapshot_examples/dl4j-spark-examples/dl4j-spark/pom.xml#L65-L71)

To use a different version of CUDA than 9.2 (CUDA 9.0, 9.2 and 10.0 are only available via snapshots) change the version numbers at line 76 and 82 at the same link above.

If the CUDA toolkit and CuDNN are not available on the worker nodes, see DL4J's Spark docs, "uber-jar" section, step 1, case 2 [here](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-howto#uberjar)


### Step 2 (Optional unless passing an env variable for ports): Configure Ports

If a single fixed UDP port is available on all nodes, skip this step and instead pass the port ("--port 12345" etc) to the training script (Step 5).

To set a different port on each node based on an environment variable, do the following:

Replace ".unicastPort(port)" (line 126 of TrainSpark) with ".portSupplier(new EnvironmentVarPortSupplier("DL4J_PORT_DIR"))" 

Remember to replace "DL4J_PORT_DIR" with the name of the environment variable that specifies the port to use on each node.
Different workers (and the master) can have different values for this environment variable.


### Step 3: Build the JAR

Run "mvn package -U" from the root dl4j-examples directory.


### Step 4: Run preprocessing

You can run either local preprocessing and copy to cluster storage **OR** run spark preprocessing (that requires the tiny imagenet files are available on the cluster).

PreprocessLocal might be easier in this particular case. See the comments in the PreprocessLocal and PreprocessSpark classes.


### Step 5: Run training

The job can be launched using Spark submit using the uber-jar from Step 3. A simple sample script is provided [here](https://github.com/deeplearning4j/dl4j-examples/tree/spark_snapshot_examples/dl4j-spark-examples/dl4j-spark/scripts/tinyImagenetTrain.sh). 

This script assumes a very small spark standalone cluster, 3 machines with 2 GPUs each. Different settings may be required for different cluster managers (YARN, Mesos, etc)

Of note, you will need to configure four main things:
  * dataPath: The location of the data from step 2
  * masterIP: The IP of the Spark master
  * networkMask: The network mask 
  * numNodes: This should be equal to the number of worker nodes (JVMs)

Others are optional, and most can be set to their default values. However, depending on the configuration, you may want to set:
  * numWorkersPerNode: Set to 1 for CPU, or equal to number of GPUs per machine for GPU
  * port: Set the port to use on each machine, unless you configured ports in step 4
  * saveDirectory: Set this to a location to save the net and evaluation results (may be local on driver - file:// or remote - hdfs:// etc)


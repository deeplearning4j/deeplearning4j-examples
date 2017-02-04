#!/bin/sh

echo "Starting DL4J entrypoint server..."
export LIBND4J_HOME=/libnd4j
java -cp /deeplearning4j-examples/deeplearning4j-keras-examples/target/deeplearning4j-keras-examples-0.7.3-SNAPSHOT.jar org.deeplearning4j.keras.Server >& /server.log &
echo "Starting Jupyter..."
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples

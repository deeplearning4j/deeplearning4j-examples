#!/bin/sh

echo "Starting DL4J entrypoint server..."
export LIBND4J_HOME=/libnd4j
java -cp /keras-dl4j/target/dl4j-keras-examples.jar org.deeplearning4j.keras.Server >& /server.log &
echo "Starting Jupyter..."
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples

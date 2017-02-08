#!/bin/sh

echo "Starting DL4J server and Jupyter..."
{ java -cp /keras-dl4j/target/dl4j-keras-examples.jar org.deeplearning4j.keras.Server & jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples; }

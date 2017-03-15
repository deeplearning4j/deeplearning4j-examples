#!/bin/bash

# give user the option for no-track
if [ -z "$NOTRACK" ]; then
    curl -XPOST https://www.google-analytics.com/collect -d 'v=1&t=event&tid=UA-48811288-1&cid=keras-prototype&ec=keras-prototype&ea=boot'
fi

echo "Starting DL4J server and Jupyter..."
phymem=$(free -g|awk '/^Mem:/{print $2}')
echo "There is $phymem available for java"
{ java -Xmx${phymem}g -cp /keras-dl4j/target/dl4j-keras-examples.jar -Dorg.bytedeco.javacpp.nopointergc=true org.deeplearning4j.keras.Server & jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples; }

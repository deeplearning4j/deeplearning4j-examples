# RL4J examples

## General instructions

If 0.5.1 deeplearning4j is not on maven central yet:

* Install libnd4j from source
* Install nd4j from source
* Install deeplearning4j from source
* Install rl4j from source

## Cartpole


![Cartpole](cartpole.gif)


```
mvn clean compile exec:java -Dexec.mainClass="org.deeplearning4j.rl4j.Doom"
```

## VizDoom instructions

```
mkdir vizdoom
cd ..
git clone https://github.com/Marqt/ViZDoom
cd VizDoom
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_JAVA=ON
make
cp bin/java/libvizdoom.so ../rl4j-examples/vizdoom/
cp bin/java/vizdoom.jar ../rl4j-examples/vizdoom/
cp bin/vizdoom ../rl4j-examples/vizdoom/
cp bin/vizdoom.pk3 ../rl4j-examples/vizdoom/
cp -r scenarios/ ../rl4j-examples/vizdoom/
cd ../rl4j-examples/
export MAVEN_OPTS="-Djava.library.path=vizdoom -Xmx30g"
mvn clean compile exec:java -Dexec.mainClass="org.deeplearning4j.rl4j.Doom"
```


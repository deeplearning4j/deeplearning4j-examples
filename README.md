Deeplearning4J Examples
=========================

## NOTE: HOW to interpret these examples

## Data Loading
In this repository, you may likely see custom [datasetiterators](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.java) - these iterators are only for special examples and 1 off use cases. Consult the [gitter](https://gitter.im/deeplearning4j/deeplearning4j) if you are not sure how to proceed.
Once you find a record reader for your use case, you then should use one of pre made iterators that knows how to interpret record reader output, either [RecordReaderDataSetIterator](https://deeplearning4j.org/api/latest/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html) for normal data or [SequenceRecordReaderDataSetIterator](https://deeplearning4j.org/api/latest/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.html) for sequence data. For more on sequences, please see our [rnns page](http://deeplearning4j.org/usingrnns)


We have special iterators for 1 off use cases where normal data does not quite exist, or sometimes it is legacy.
99% of the time you should be using [datavec](https://deeplearning4j.org/datavec) and writing your own custom [record readers](https://deeplearning4j.org/api/latest/org/datavec/api/records/reader/RecordReader.html) if one of our pre provided ones is not suitable. If you are not sure what is available, please again consult the [gitter](https://gitter.im/deeplearning4j/deeplearning4j) - In general, you can find both [normal record readers](https://deeplearning4j.org/api/latest/org/datavec/api/records/reader/RecordReader.html) and [sequence record readers](https://deeplearning4j.org/api/latest/org/datavec/api/records/reader/SequenceRecordReader.html) in the datavec [javadoc](https://deeplearning4j.org/docs/latest/datavec-overview).


## Dependencies

Note that this repository contains all dl4j examples for all modules. It will download about 1.5g of dependencies from maven central when you are first starting out. That being said, this makes it easier to get started without worrying about what to download. This examples repository is meant to be a reference point to get started with most common use cases.
It is broken up in to modules. If you would like to just have a more minimal/simple, guide please go [here](http://www.dubs.tech/guides/maven-essentials/)


Repository of Deeplearning4J neural net examples:

- MLP Neural Nets
- Convolutional Neural Nets
- Recurrent Neural Nets
- TSNE
- Word2Vec & GloVe
- Anomaly Detection
- User interface examples.

DL4J-Examples is released under an Apache 2.0 license. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

---

## Build and Run

Use [Maven](https://maven.apache.org/) to build the examples.

```
mvn clean package
```

This downloads binaries for all platforms, but we can also append `-Djavacpp.platform=` with `android-arm`, `android-x86`, `linux-ppc64le`, `linux-x86_64`, `macosx-x86_64`, or `windows-x86_64` to get binaries for only one platform and produce much smaller archives.

Run the `runexamples.sh` script to run the examples (requires [bash](https://www.gnu.org/software/bash/)). It will list the examples and prompt you for the one to run. Pass the `--all` argument to run all of them. (Other options are shown with `-h`).

```
./runexamples.sh [-h | --help]
```


## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](https://deeplearning4j.org/docs/latest/).

`GradientsListenerExample.java` in dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface uses JavaFX. If you're using Java 8 or greater, it should run as is.  If you're using Java 7 or an earlier version, you should set JAVAFX_HOME to point to the root directory of the JavaFX 2.0 SDK.


### Known issues with JavaFX

If you are running on JDK 1.7 or inferior, the maven-enforcer plugin will require you to set the variable `JAVAFX_HOME` before building.
That variable should point to a directory containing `jfxrt.jar`, a file that is part of the JavaFX 2.0 distrbution.

Please set it to an instance of JavaFX that matches the JDK with which you are trying to use this project. Usually, the Sun JDK comes with JavaFX. However, OpenJDK does not and you may have to install OpenJFX, a free distribution of JavaFX.

Beware that your editor (e.g. IntelliJ) may not be using the JDK that is your system default (and that you may ancounter on the command line).

### On IntelliJ

To run the JavaFX examples from IntelliJ, you'll have to add the `jfxrt.jar` as an exernal dependency of your project. Here's a screencast on how to do it: https://youtu.be/si146q7WkSY

#### On Ubuntu

If you are using OpenJDK, on Ubuntu 16, you can install OpenJFX with `sudo apt-get install openjfx libopenjfx-java`. A typical `JAVAFX_HOME` is then `/usr/share/java/openjfx/jre/lib/ext/`. If you are on Ubuntu 14, you can install OpenJFX with the following process:

- edit `/etc/apt/sources.list.d/openjdk-r-ppa-trusty.list` and uncomment the line for deb-src
- `sudo apt-get update`
- `sudo apt-get install libicu-dev`
- `sudo aptitude build-dep libopenjfx-java`
- `sudo apt-get --compile source libopenjfx-java`
- `ls -1 *.deb|xargs sudo dpkg -i`

#### On JDK 1.8

The Sun version of JDK8 still comes with its own JavaFX, so that there should be no need to configure anything particular there and the build will succeed. If using OpenJDK8, you will still have to install OpenJFX and set `JAVAFX_HOME`, but the maven-enforcer plugin will not catch you — the reason being that it's difficult to distinguish between OpenJDK and Sun's JDK since version 8, with both adoptiong the same Vendor ID.

If you are using OpenJDK 8, install OpenJFX and set JAVAFX_HOME as indicated above. Compile with `mvn clean install -POpenJFX`

## Other Issues

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.


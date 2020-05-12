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
- Word2Vec
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

## Other Issues

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.


Deeplearning4J Examples
=========================

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
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

`GradientsListenerExample.java` in dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface uses JavaFX. If you're using Java 8 or greater, it should run as is.  If you're using Java 7 or an earlier version, you should set JAVAFX_HOME to point to the root directory of the JavaFX 2.0 SDK.

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.


# ScalNet examples

This sbt-project contains ScalNet examples for both
DL4J-style NeuralNet model construction pattern and
keras-style Sequential model construction pattern.

## Build and run

`cd` to the project root directory `scalnet-examples` and run:

```bash
sbt run
```

It will build the project and prompt you to select the example to run.

To specify Scala version one can pass it as an environment variable

```bash
SCALA_VERSION=2.10.6 sbt run
```

or pass it as a system property

```bash
sbt -Dscala.version=2.10.6 run
```


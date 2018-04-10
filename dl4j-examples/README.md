# Gradle Kotlin Examples

Open built-in Terminal from the Android Studio or IntelliJ IDEA.
If you prefer, use the terminal app on your computer.
In the terminal window, `cd` to the project root directory `dl4j-examples` if not already in.
And then type `./gradlew NameOfExampleToRun` to run the examples as shown below.
(_you can use IDE's run-configuration green run button, if you like._)

MLPMnistSingleLayerExample
```
./gradlew MLPMnistSingleLayerExample
```

MLPMnistTwoLayerExample
```
./gradlew MLPMnistTwoLayerExample
```

etc.

# Maven Kotlin Examples

Simply import the repo's root pom file as a project into IntelliJ.
Then add a module via the pom file beside this README.md file.

To run am example
1. Edit run configurations, create an Application
2. Specify `Main Class` for example as `org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample`
3. Specify `Use classpath of module` as `dl4j-examples`
4. Save and hit run button

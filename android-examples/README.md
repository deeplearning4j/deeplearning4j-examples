# Eclipse Deeplearning4J Examples for Android

This project shows a simple example of using Deeplearning4J in an Android application. As Gradle is the preferred build tool for Android, this project uses Gradle instead of Maven.

The neural networks shown in this example are the same as are used in the [Moon](../dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MoonClassifier.java) and [Saturn](../dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/SaturnClassifier.java) examples of the dl4j-examples.

The build configuration for Android can be found in [app/build.gradle](./app/build.gradle).

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

# Known Issues
Due to an unfortunately timed change, the 1.0.0-beta7 release doesn't work well with most Android devices. For this reason, this example project uses the SNAPSHOT version.

Again unfortunately, gradle has issues retrieving SNAPSHOT versions when combined with classifiers, which are used here to retrieve only the android specific backend dependencies.

If you want to run this example locally, you will therefore need to have Maven installed in order to download those specific dependencies.

Once you have installed Maven, you will need to run the following commands from the command line to download the correct android backend files:

```
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-arm
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-arm64
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-x86
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-x86_64
```

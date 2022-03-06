# Eclipse Deeplearning4J Examples for Android

This project shows a simple example of using Deeplearning4J in an Android application. As Gradle is the preferred build tool for Android, this project uses Gradle instead of Maven.

The neural networks shown in this example are the same as are used in the [Moon](../dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/MoonClassifier.java) and [Saturn](../dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/feedforward/classification/SaturnClassifier.java) examples of the dl4j-examples.

The build configuration for Android can be found in [app/build.gradle](./app/build.gradle).

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j/issues) to request new features.


```
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-arm
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-arm64
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-x86
mvn dependency:get -DremoteRepositories=snapshots::::https://oss.sonatype.org/content/repositories/snapshots -Dartifact=org.nd4j:nd4j-native:1.0.0-SNAPSHOT:jar:android-x86_64
```

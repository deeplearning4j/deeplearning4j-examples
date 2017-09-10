name := "scalnet-examples"

val dl4jVersion = "0.9.1"

version := dl4jVersion

scalaVersion :=  "2.11.8"

resolvers += "Typesafe Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % dl4jVersion

libraryDependencies += "org.deeplearning4j" % "scalnet_2.11" % dl4jVersion

publishArtifact := false

publishArtifact in Test := false

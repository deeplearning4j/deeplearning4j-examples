name := "scalnet-examples"

val dl4jVersion = "0.9.1"

version := dl4jVersion

scalaVersion :=  sys.props.get("scala.version").orElse(sys.env.get("SCALA_VERSION")).getOrElse("2.11.8")

resolvers += "Typesafe Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % dl4jVersion

libraryDependencies += "org.deeplearning4j" % s"scalnet_${scalaBinaryVersion.value}" % dl4jVersion

publishArtifact := false

publishArtifact in Test := false

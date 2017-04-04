name := "dl4j-mds-example"

version := "1.0"

scalaVersion := "2.11.8"


resolvers += Resolver.mavenLocal
resolvers += "OSS Sonatype" at "https://repo1.maven.org/maven2/"

libraryDependencies += "org.deeplearning4j" % "dl4j-spark-nlp_2.11" % "0.7.2"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.11" % "0.7.2"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.7.2"
libraryDependencies += "org.springframework" % "spring-core" % "4.3.3.RELEASE"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "com.typesafe" % "config" % "1.2.1"
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.1.3" % "test" 
libraryDependencies += "junit" % "junit" % "4.10"  //adding juni
libraryDependencies += "org.apache.commons" % "commons-collections4" % "4.1"


//mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
//  {
//    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//   case x => MergeStrategy.first
//  }
//}




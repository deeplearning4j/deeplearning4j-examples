name := "dl4j-mds-example"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += Resolver.mavenLocal
resolvers += "OSS Sonatype" at "https://repo1.maven.org/maven2/"

libraryDependencies += "org.javassist" % "javassist" % "3.20.0-GA"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark-nlp_2.11" % "0.6.0"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.11" % "0.6.0"
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"
//libraryDependencies += "org.nd4j" % "nd4j-jblas" % "0.4-rc3.6"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.6.0"
//libraryDependencies += "org.nd4j" % "nd4j-api" % "0.6.0"
//libraryDependencies += "org.nd4j" % "nd4j-scala-api" % "0.0.3.5.5.5"
//libraryDependencies += "org.nd4j" % "nd4j-java" % "0.4-rc3.6"
//libraryDependencies += "org.nd4j" % "nd4j-native" % "0.6.0"
libraryDependencies += "org.springframework" % "spring-core" % "4.3.3.RELEASE"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "com.typesafe" % "config" % "1.2.1"
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.1.3" % "test" 
libraryDependencies += "junit" % "junit" % "4.10"  //adding junit
libraryDependencies += "org.apache.commons" % "commons-collections4" % "4.1"



//mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
//  {
//    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//   case x => MergeStrategy.first
//  }
//}



//libraryDependencies += "org.deeplearning4j" % "dl4j-spark" % "0.0.3.3.3.alpha1-SNAPSHOT"

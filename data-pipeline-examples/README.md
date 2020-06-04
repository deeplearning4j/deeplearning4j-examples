## Eclipse Deeplearning4j: Data pipeline, DataVec Examples

This project contains a set of examples that demonstrate how raw data in various formats can be loaded, split and preprocessed to build serializable (and hence reproducible) ETL pipelines using the DataVec library.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

The examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.

### Loading Data
InputSplit and its implementations are utility classes for defining and managing a catalog of loadable locations (paths/files), in memory, that can later be exposed through an Iterator. In simple terms, they define where your data comes from or should be saved to, when building a data pipeline with DataVec.

* [Ex01_FileSplitExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex01_FileSplitExample.java)
Using FileSplit which loads files in a given location. Constructor overloading allows for varying functionality like filtering files to load, loading recursively etc
* [Ex02_CollectionSplitExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex02_CollectionSplitExample.java)
Create a split from a collection of URIs
* [Ex03_NumberedFileInputSplitExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex03_NumberedFileInputSplitExample.java)
Create a split from numbered files, following a common pattern like file1.txt, file2.txt ... file100.txt
* [Ex04_TransformSplitExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex04_TransformSplitExample.java)
Maps URIs of a given split to new URIs. Useful when features and labels are in different files sharing a common naming scheme, and the name of the output file can be determined given the name of the input file. Eg. a-in.csv and a-out.csv
* [Ex05_SamplingBaseInputSplitExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex05_SamplingBaseInputSplitExample.java)
Generate several splits from the main split say for training, validation and testing.
* [Ex06_KFoldIteratorFromDataSet.java](./src/main/java/org/deeplearning4j/datapipelineexamples/loading/Ex06_KFoldIteratorFromDataSet.java)
Generate a K-Fold iterator from a dataset

### Cleaning, Transforming and Analysing Data
* [IrisCSVTransform.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/IrisCSVTransform.java)
A basic example that introduces users to important concepts like Schema and TransformProcess with categoricalToInteger.
* [CSVMixedDataTypesLocal.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/CSVMixedDataTypesLocal.java)
Common preprocessing steps like removing unnecessary columns, filtering based on column value, replacing invalid values, parsing date time etc
* [CSVMixedDataTypes.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/CSVMixedDataTypes.java)
Same as the above but with Apache Spark
* [PrintSchemasAtEachStep.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/debugging/PrintSchemasAtEachStep.java)
How to print schema at each step which would be useful for debugging transform scripts in a complicated pipeline
* [IrisAnalysis.java](./src/main/java/org/deeplearning4j/datapipelineexamples/analysis/IrisAnalysis.java)
Basic Analysis of the dataset saved and presented as an html file
* [IrisNormalizer.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/IrisNormalizer.java)
Proper useage of preprocessors with min max scaler
* [JoinExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/JoinExample.java)
Perform joins on datasets
* [PivotExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/PivotExample.java)
Combine multiple independent records by key.
* [WebLogDataExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/basic/WebLogDataExample.java)
Preprocessing/aggregation operations on some web log data
* [CustomReduceExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/custom/CustomReduceExample.java)
Custom Reduction example for operations on some simple CSV data that involve a custom reduction.
* [MultiOpReduceExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/transform/custom/MultiOpReduceExample.java)
Reduce example with multiple ops on one column

### Formats
* [CSVtoMapFileConversion.java](./src/main/java/org/deeplearning4j/datapipelineexamples/formats/hdfs/conversion/CSVtoMapFileConversion.java)
A simple example on how to convert a CSV text file to a Hadoop MapFile format for better performance and the convenience of randomization supported by the MapFileRecordReader
* [SVMLightExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/formats/svmlight/SVMLightExample.java)
MNIST SVMLight example
* [ImagePipelineExample.java](./src/main/java/org/deeplearning4j/datapipelineexamples/formats/image/ImagePipelineExample.java)
An imagepipeline that also demonstrates using transforms to augment a small dataset

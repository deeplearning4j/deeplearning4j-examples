package org.datavec.transform.timeseries.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.transform.timeseries.spark.convert.CSVSingleLineToSequenceDataSet;

import org.nd4j.linalg.dataset.DataSet;

public class CSVSingleColSingleFileExample {

	public static void main(String[] args) throws Exception {
		
		boolean useSparkLocal = true;
		
		SparkConf sparkConf = new SparkConf();
        
		if (useSparkLocal) {
        	sparkConf.setMaster("local[*]");
        }
		
		System.out.println("running...");
        
        sparkConf.setAppName("DataVec Timeseries CSV Single Line Spark Job");
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		String pathToECGData = "src/main/resources/CSVData/SingleColumn/csv_timeseries_1_col.csv";
		
		// load the training data
		JavaRDD<String> trainingRecordCSVLines = sc.textFile( pathToECGData ); //dataPath + "/train/train.csv");
		
		System.out.println( "loaded: " + pathToECGData );

		JavaRDD<DataSet> trainingDataSet = trainingRecordCSVLines.map( new CSVSingleLineToSequenceDataSet(new CSVRecordReader(0, ","), 0, 5 ) );
		
  		DataSet d = trainingDataSet.first();
        
        System.out.println( d );
            
        long c = trainingDataSet.count();
        
        System.out.println( "count: " + c );

        System.out.println( "----- Example Complete -----" );

        sc.close();        
        
        
	}	
	
}

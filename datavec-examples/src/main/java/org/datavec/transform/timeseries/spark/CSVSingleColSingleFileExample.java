package org.datavec.transform.timeseries.spark;


//import io.skymind.examples.cdh.spark.rnns.ecg.data.ECGDataToSequenceDataSet;
//import io.skymind.examples.cdh.spark.rnns.ecg.preprocessor.SparkDataNormalization;
//import io.skymind.examples.cdh.spark.rnns.ecg.preprocessor.SparkNormalizerStandardize;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.transform.timeseries.spark.convert.CSVSingleLineToSequenceDataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
//import org.nd4j.linalg.dataset.api.preprocessor.SparkDataNormalization;
//import org.nd4j.linalg.dataset.api.preprocessor.SparkNormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;

public class CSVSingleColSingleFileExample {

	public static void main(String[] args) throws Exception {
		
		//String dataPath = "src/main/resources/uci/";
		boolean writeStats = false;
		boolean copyDataToHDFS = false;
		boolean saveUpdater = true;
		boolean useSparkLocal = true;
		
		
		
		SparkConf sparkConf = new SparkConf();
        
		if (useSparkLocal) {
        	sparkConf.setMaster("local[*]");
        }
		
		System.out.println("running...");
        
        sparkConf.setAppName("ECG Sensor Data - LSTM - Classificiation Spark Job");
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		
		// setup variables
		
		
		//String pathData = "/tmp/ECG5000_TRAIN";
		
		String pathToECGData = "src/main/resources/CSVData/SingleColumn/csv_timeseries_1_col.csv";
		
		// load the training data
		JavaRDD<String> trainingRecordCSVLines = sc.textFile( pathToECGData ); //dataPath + "/train/train.csv");
		
		System.out.println( "loaded: " + pathToECGData );
		//System.out.println( "count: " + trainingRecordCSVLines.count() );
		
		
		//JavaRDD<List<Writable>> trainingWritables
		JavaRDD<DataSet> trainingDataSet = trainingRecordCSVLines.map( new CSVSingleLineToSequenceDataSet(new CSVRecordReader(0, ","), 0, 5 ) );
		
//        JavaRDD<DataSet> trainingDataSet = trainingWritables.map(new DataVecDataSetFunction(0, labelCount, false));

		DataSet d = trainingDataSet.first();
        
        //int col = d.getFeatures().columns();
        //int row = d.getFeatures().rows();
        
        //System.out.println( "rows: " + row + ", col: " + col );
		
        //for(DataSet l : d){
            
        	System.out.println( d );
            
        	//f.call( l ); 

        //}   
		
		
        
        
        long c = trainingDataSet.count();
        
        System.out.println( "count: " + c );

/*        
     // normalize
        SparkDataNormalization normalization = new SparkNormalizerStandardize();
        normalization.fit( trainingDataSet );

       ((SparkNormalizerStandardize)normalization).debugStats();
        

        // apply normalization of train set to both tr and te
        JavaRDD<DataSet> trainRDDNorm = normalization.preProcess( trainingDataSet );
        //JavaRDD<DataSet> testRDDNorm = normalization.preProcess(dataSetsTE);

        // create batch dataset
        //JavaRDD<DataSet> trainRDD = trainRDDNorm.mapPartitions(new BatchDataSetsFunction(miniBatchSize));
        //JavaRDD<DataSet> testRDD = testRDDNorm.mapPartitions(new BatchDataSetsFunction(miniBatchSize));        
        
        System.out.println( "post standardize count: " + trainRDDNorm.count() );
        */
        System.out.println( "----- Example Complete -----" );

        sc.close();        
        
        
	}	
	
}

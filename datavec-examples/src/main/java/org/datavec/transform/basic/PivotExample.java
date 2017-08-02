package org.datavec.transform.basic;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.List;

/**
 * In this simple example: We'll show how to combine multiple independent records by key.
 * Specifically, assume we have data like "person,country_visited,entry_time" and we want to know how many times
 * each person has entered each country.
 *
 *
 * @author Alex Black
 */
public class PivotExample {

    public static void main(String[] args) throws Exception {

        //=====================================================================
        //                 Step 1: Define the input data schema
        //=====================================================================

        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        Schema inputDataSchema = new Schema.Builder()
            .addColumnString("person")
            .addColumnCategorical("country_visited", Arrays.asList("USA","Japan","China","India"))
            .addColumnString("entry_time")
            .build();

        //=====================================================================
        //            Step 2: Define the operations we want to do
        //=====================================================================

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            //First: let's parse the date format to epoch format
            //Format for parsing times is as per http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
            .stringToTimeTransform("entry_time", "YYYY/MM/dd", DateTimeZone.UTC )

            //Let's first take the "country_visited" column and expand it to a one-hot representation
            //So, "USA" becomes [1,0,0,0], "Japan" becomes [0,1,0,0], "China" becomes [0,0,1,0] etc
            .categoricalToOneHot("country_visited")

            //Reduction: For each person, count up the number of times they have
            .reduce(new Reducer.Builder(ReduceOp.Sum)
                .keyColumns("person")
                .maxColumn("entry_time")
                .build())

            //We can rename our columns
            .renameColumn("max(entry_time)", "most_recent_entry")

            .build();



        //=====================================================================
        //      Step 3: Load our data and execute the operations on Spark
        //=====================================================================

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        //Define the path to the data file. You could use a directory here if the data is in multiple files
        //Normally just define your path like "file:/..." or "hdfs:/..."
        String path = new ClassPathResource("BasicDataVecExample/PivotExampleData.csv").getFile().getAbsolutePath();
        JavaRDD<String> stringData = sc.textFile(path);

        //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        //Now, let's execute the transforms we defined earlier:
        JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);

        //For the sake of this example, let's collect the data locally and print it:
        JavaRDD<String> processedAsString = processedData.map(new WritablesToStringFunction(","));
        //processedAsString.saveAsTextFile("file://your/local/save/path/here");   //To save locally
        //processedAsString.saveAsTextFile("hdfs://your/hdfs/save/path/here");   //To save to hdfs

        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = stringData.collect();

        sc.stop();
        Thread.sleep(2000); //Wait a few seconds for Spark to stop logging to console


        //Finally: we'll print the schemas and results
        System.out.println("\n\n---- Original Data Schema ----");
        System.out.println(inputDataSchema);

        System.out.println("\n\n---- Final Data Schema ----");
        Schema finalDataSchema = tp.getFinalSchema();
        System.out.println(finalDataSchema);

        System.out.println("\n\n---- Original Data ----");
        for(String s : inputDataCollected) System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        for(String s : processedCollected) System.out.println(s);


        System.out.println("\n\nDONE");




    }

}

package org.datavec.transform.basic;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.datavec.python.PythonTransform;
import org.datavec.python.PythonCondition;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.Arrays;
import java.util.List;

/**
 * Basic DataVec example for preprocessing operations on some simple CSV data using python code.
 *
 * The premise here is that some data regarding transactions is available in CSV format, and we want to do some
 * operations on this data, including:
 * 2. Filtering examples to keep only examples with values "USA" or "CAN" for the "MerchantCountryCode" column
 * 3. Replacing some invalid values in the "TransactionAmountUSD" column
 * 4. Parsing the date string, and extracting the hour of day from it to create a new "HourOfDay" column
 *
 * @author Alex Black
 */
public class BasicDataVecPythonExample {

    public static  void main(String[] args) throws Exception {

        //=====================================================================
        //                 Step 1: Define the input data schema
        //=====================================================================

        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        Schema inputDataSchema = new Schema.Builder()
            //We can define a single column
            .addColumnString("DateTimeString")
            //Or for convenience define multiple columns of the same type
            .addColumnsString("CustomerID", "MerchantID")
            //We can define different column types for different types of data:
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
            //Some columns have restrictions on the allowable values, that we consider valid:
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();

        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());


        //=====================================================================
        //            Step 2: Define the operations we want to do
        //=====================================================================


        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)

            .filter(new PythonCondition(
                //Suppose we only want to analyze transactions involving merchants in USA or Canada. Let's filter out
                // everything except for those countries.
                "f = lambda: MerchantCountryCode not in ['USA', 'CAN']"
            ))


            .transform(
                new PythonTransform(
                    //Let's suppose our data source isn't perfect, and we have some invalid data: negative dollar amounts that we want to replace with 0.0
                    //For positive dollar amounts, we don't want to modify those values
                    "if TransactionAmountUSD < 0: TransactionAmountUSD=0\n" +
                        //At this point, we have our date/time format stored in a format like "2016/01/01 17:50.000
                        //Suppose we only care about the hour of the day. Let's derive a new column for that, from the DateTime column
                        "HourOfDay = int(DateTimeString.split(' ')[1].split(':')[0])\n"
                )
            )
            .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


        //=====================================================================
        //      Step 3: Load our data and execute the operations on Spark
        //=====================================================================

        //We'll use Spark local to handle our data

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        //Define the path to the data file. You could use a directory here if the data is in multiple files
        //Normally just define your path like "file:/..." or "hdfs:/..."
        String path = new ClassPathResource("BasicDataVecExample/exampledata.csv").getFile().getAbsolutePath();
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


        System.out.println("\n\n---- Original Data ----");
        for(String s : inputDataCollected) System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        for(String s : processedCollected) System.out.println(s);


        System.out.println("\n\nDONE");
    }

}

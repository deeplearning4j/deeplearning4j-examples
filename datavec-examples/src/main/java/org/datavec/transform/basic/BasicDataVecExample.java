package org.datavec.transform.basic;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * Basic DataVec example for preprocessing operations on some simple CSV data. If you just want to load CSV data
 * and pass it on for learning take a look at {@see org.deeplearning4j.examples.dataExample.CSVExample}.
 *
 * The premise here is that some data regarding transactions is available in CSV format, and we want to do some some
 * operations on this data, including:
 * 1. Removing some unnecessary columns
 * 2. Filtering examples to keep only examples with values "USA" or "CAN" for the "MerchantCountryCode" column
 * 3. Replacing some invalid values in the "TransactionAmountUSD" column
 * 4. Parsing the date string, and extracting the hour of day from it to create a new "HourOfDay" column
 *
 * @author Alex Black
 */
public class BasicDataVecExample {

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

        //Lets define some operations to execute on the data...
        //We do this by defining a TransformProcess
        //At each step, we identify column by the name we gave them in the input data schema, above

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            //Let's remove some column we don't need
            .removeColumns("CustomerID","MerchantID")

            //Now, suppose we only want to analyze transactions involving merchants in USA or Canada. Let's filter out
            // everything except for those countries.
            //Here, we are applying a conditional filter. We remove all of the examples that match the condition
            // The condition is "MerchantCountryCode" isn't one of {"USA", "CAN"}
            .filter(new ConditionFilter(
                new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet<>(Arrays.asList("USA","CAN")))))

            //Let's suppose our data source isn't perfect, and we have some invalid data: negative dollar amounts that we want to replace with 0.0
            //For positive dollar amounts, we don't want to modify those values
            //Use the ConditionalReplaceValueTransform on the "TransactionAmountUSD" column:
            .conditionalReplaceValueTransform(
                "TransactionAmountUSD",     //Column to operate on
                new DoubleWritable(0.0),    //New value to use, when the condition is satisfied
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0

            //Finally, let's suppose we want to parse our date/time column in a format like "2016/01/01 17:50.000"
            //We use JodaTime internally, so formats can be specified as follows: http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
            .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)

            //However, our time column ("DateTimeString") isn't a String anymore. So let's rename it to something better:
            .renameColumn("DateTimeString", "DateTime")

            //At this point, we have our date/time format stored internally as a long value (Unix/Epoch format): milliseconds since 00:00.000 01/01/1970
            //Suppose we only care about the hour of the day. Let's derive a new column for that, from the DateTime column
            .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime")
                .addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay())
                .build())

            //We no longer need our "DateTime" column, as we've extracted what we need from it. So let's remove it
            .removeColumns("DateTime")

            //We've finished with the sequence of operations we want to do: let's create the final TransformProcess object
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

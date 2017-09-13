package org.datavec.transform.basic;

import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.joda.time.DateTimeZone;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Custom Reduction example for operations on some simple CSV data that involve a custom reduction.
 *
 * @author François Garillot
 */

public class MultiOpReduceExample {

    public static  void main(String[] args) throws Exception {

        //=====================================================================
        //                 Step 1: Define the input data schema as in the Basic Example
        //=====================================================================

        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        Schema inputDataSchema = new Schema.Builder()
            .addColumnString("DateTimeString")
            .addColumnsString("CustomerID", "MerchantID")
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
            //Some columns have restrictions on the allowable values, that we consider valid:
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();



        //Lets define some operations to execute on the data...
        //We do this by defining a TransformProcess
        //At each step, we identify column by the name we gave them in the input data schema, above

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            //Let's remove some column we don't need
            .removeColumns("CustomerID","MerchantID", "MerchantCountryCode", "FraudLabel")

            //Finally, let's suppose we want to parse our date/time column in a format like "2016/01/01 17:50.000"
            //We use JodaTime internally, so formats can be specified as follows: http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
            .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)

            //However, our time column ("DateTimeString") isn't a String anymore. So let's rename it to something better:
            .renameColumn("DateTimeString", "DateTime")

            //We no longer need our "DateTime" column, as we've extracted what we need from it. So let's remove it
            .reduce(new Reducer.Builder(ReduceOp.TakeFirst)
                .keyColumns("DateTime", "DateTimeString")
                .maxColumn("NumItemsInTransaction")
                // This is the multiple Op example : one column, three operations
                .multipleOpColmumns(
                    new ArrayList<ReduceOp>(Arrays.asList(ReduceOp.TakeFirst, ReduceOp.TakeLast, ReduceOp.Max)),
                    "TransactionAmountUSD")
                .build())

            //We've finished with the sequence of operations we want to do: let's create the final TransformProcess object
            .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


    }

}

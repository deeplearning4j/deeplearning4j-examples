package org.datavec.transform.debugging;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.HashSet;

/**
 * This is a simple example for the DataVec transformation functionality (building on BasicDataVecExample)
 * It is designed to simply demonstrate that it is possible to obtain the schema after each step of a transform process.
 * This can be useful for debugging your TransformProcess scripts.
 *
 * @author Alex Black
 */
public class PrintSchemasAtEachStep {

    public static void main(String[] args){

        //Define the Schema and TransformProcess as per BasicDataVecExample
        Schema inputDataSchema = new Schema.Builder()
            .addColumnsString("DateTimeString", "CustomerID", "MerchantID")
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            .removeColumns("CustomerID","MerchantID")
            .filter(new ConditionFilter(new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet<>(Arrays.asList("USA","CAN")))))
            .conditionalReplaceValueTransform(
                "TransactionAmountUSD",     //Column to operate on
                new DoubleWritable(0.0),    //New value to use, when the condition is satisfied
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0
            .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
            .renameColumn("DateTimeString", "DateTime")
            .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime").addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay()).build())
            .removeColumns("DateTime")
            .build();


        //Now, print the schema after each time step:
        int numActions = tp.getActionList().size();

        for(int i=0; i<numActions; i++ ){
            System.out.println("\n\n==================================================");
            System.out.println("-- Schema after step " + i + " (" + tp.getActionList().get(i) + ") --");

            System.out.println(tp.getSchemaAfterStep(i));
        }


        System.out.println("DONE.");
    }

}

package org.datavec.transform.basic;

import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.ops.AggregableMultiOp;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.reduce.AggregableColumnReduction;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.UnsafeWritableInjector;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Custom Reduction example for operations on some simple CSV data that involve a custom reduction.
 *
 * @author François Garillot
 */

public class CustomReduceExample {

    private static class CustomReduceTakeSecond implements AggregableColumnReduction {

        @Override
        public IAggregableReduceOp<Writable, List<Writable>> reduceOp() {
            //For testing: let's take the second value
            return new AggregableMultiOp<>(Collections.<IAggregableReduceOp<Writable, Writable>>singletonList(new AggregableSecond<Writable>()));
        }

        @Override
        public List<String> getColumnsOutputName(String columnInputName) {
            return Collections.singletonList("myCustomReduce(" + columnInputName + ")");
        }

        @Override
        public List<ColumnMetaData> getColumnOutputMetaData(List<String> newColumnName, ColumnMetaData columnInputMeta) {
            ColumnMetaData thiscolumnMeta = new StringMetaData(newColumnName.get(0));
            return Collections.singletonList(thiscolumnMeta);
        }

        public static class AggregableSecond<T> implements IAggregableReduceOp<T, Writable> {
            private T firstMet = null;
            private T elem = null;

            protected T getFirstMet(){
                return firstMet;
            }

            protected T getElem(){
                return elem;
            }

            @Override
            public void accept(T element) {
                if (firstMet == null) firstMet = element;
                else {
                    if (elem == null) elem = element;
                }
            }

            @Override
            public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
                if (accu instanceof AggregableSecond && elem == null) {
                    if (firstMet == null) { // this accumulator is empty, import accu
                        AggregableSecond<T> accumulator = (AggregableSecond) accu;
                        T otherFirst = accumulator.getFirstMet();
                        T otherElement = accumulator.getElem();
                        if (otherFirst != null) firstMet = otherFirst;
                        if (otherElement != null) elem = otherElement;
                    } else { // we have the first element, they may have the rest
                        AggregableSecond<T> accumulator = (AggregableSecond) accu;
                        T otherFirst = accumulator.getFirstMet();
                        if (otherFirst != null) elem = otherFirst;
                    }
                }
            }

            @Override
            public Writable get() {
                return UnsafeWritableInjector.inject(elem);
            }
        }

        /**
         * Get the output schema for this transformation, given an input schema
         *
         * @param inputSchema
         */
        @Override
        public Schema transform(Schema inputSchema) {
            return null;
        }

        /**
         * Set the input schema.
         *
         * @param inputSchema
         */
        @Override
        public void setInputSchema(Schema inputSchema) {

        }

        /**
         * Getter for input schema
         *
         * @return
         */
        @Override
        public Schema getInputSchema() {
            return null;
        }

        /**
         * The output column name
         * after the operation has been applied
         *
         * @return the output column name
         */
        @Override
        public String outputColumnName() {
            return null;
        }

        /**
         * The output column names
         * This will often be the same as the input
         *
         * @return the output column names
         */
        @Override
        public String[] outputColumnNames() {
            return new String[0];
        }

        /**
         * Returns column names
         * this op is meant to run on
         *
         * @return
         */
        @Override
        public String[] columnNames() {
            return new String[0];
        }

        /**
         * Returns a singular column name
         * this op is meant to run on
         *
         * @return
         */
        @Override
        public String columnName() {
            return null;
        }
    }

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
                .customReduction("", new CustomReduceTakeSecond())
                .build())

            //We've finished with the sequence of operations we want to do: let's create the final TransformProcess object
            .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


    }

}

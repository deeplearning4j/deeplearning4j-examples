/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package org.deeplearning4j.examples.quickstart.datapipeline;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.List;

/**
 * DataVec Data Pipeline - Complete API Reference
 *
 * This example covers the DataVec ETL pipeline for loading and preprocessing data:
 *
 *   1. CSVRecordReader - Load CSV files
 *   2. Schema - Define column types and names
 *   3. TransformProcess - Apply transforms (remove columns, convert types, etc.)
 *   4. RecordReaderDataSetIterator - Bridge from DataVec to DL4J
 *   5. Normalizers - StandardScaler, MinMaxScaler
 *
 * Key pipeline pattern:
 *   RecordReader -> (TransformProcess) -> RecordReaderDataSetIterator -> DataSet
 *
 * Key classes:
 *   - org.datavec.api.records.reader.impl.csv.CSVRecordReader
 *   - org.datavec.api.transform.schema.Schema
 *   - org.datavec.api.transform.TransformProcess
 *   - org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
 *   - org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
 */
public class DataPipelineExample {

    public static void main(String[] args) throws Exception {

        File tempDir = Files.createTempDirectory("datapipeline_example").toFile();

        // Create a sample CSV dataset (Iris-like)
        File csvFile = createSampleCSV(tempDir);

        // ============================================================
        // 1. BASIC CSV READING
        // ============================================================
        System.out.println("=== Basic CSV Reading ===");
        {
            // CSVRecordReader: skipLines=1 (header), delimiter=','
            RecordReader reader = new CSVRecordReader(1, ',');
            reader.initialize(new FileSplit(csvFile));

            System.out.println("  Reading CSV records:");
            int count = 0;
            while (reader.hasNext() && count < 3) {
                List<Writable> record = reader.next();
                System.out.println("    Row " + count + ": " + record);
                count++;
            }
            reader.reset();

            // Batch reading: get multiple records at once
            List<List<Writable>> batch = reader.next(5);
            System.out.println("  Batch of 5 records: " + batch.size() + " rows");
            System.out.println("  First row columns: " + batch.get(0).size());

            reader.close();
        }

        // ============================================================
        // 2. SCHEMA DEFINITION
        // ============================================================
        System.out.println("\n=== Schema Definition ===");
        Schema inputSchema;
        {
            // Define the schema for our CSV
            inputSchema = new Schema.Builder()
                    .addColumnDouble("sepal_length")
                    .addColumnDouble("sepal_width")
                    .addColumnDouble("petal_length")
                    .addColumnDouble("petal_width")
                    .addColumnInteger("species")    // 0, 1, or 2
                    .build();

            System.out.println("  Schema columns: " + inputSchema.numColumns());
            System.out.println("  Column names: " + inputSchema.getColumnNames());
            System.out.println("  Column types: " + inputSchema.getColumnTypes());
        }

        // ============================================================
        // 3. TRANSFORM PROCESS
        // ============================================================
        System.out.println("\n=== Transform Process ===");
        {
            // Build a transform pipeline
            TransformProcess tp = new TransformProcess.Builder(inputSchema)
                    // Remove columns you don't want
                    // .removeColumns("unwanted_column")

                    // Example: keep all columns as-is for this demo
                    .build();

            Schema outputSchema = tp.getFinalSchema();
            System.out.println("  Input columns:  " + inputSchema.numColumns());
            System.out.println("  Output columns: " + outputSchema.numColumns());
            System.out.println("  Output names:   " + outputSchema.getColumnNames());
        }

        // ============================================================
        // 4. RECORD READER -> DATASET ITERATOR
        // ============================================================
        System.out.println("\n=== RecordReaderDataSetIterator ===");
        {
            RecordReader reader = new CSVRecordReader(1, ',');
            reader.initialize(new FileSplit(csvFile));

            int labelIndex = 4;    // column index of the label
            int numClasses = 3;    // number of distinct classes
            int batchSize = 10;

            // Create the iterator that bridges DataVec -> DL4J
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    reader,
                    batchSize,
                    labelIndex,
                    numClasses);

            // Iterate through batches
            int batchCount = 0;
            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                if (batchCount == 0) {
                    System.out.println("  First batch:");
                    System.out.println("    Features shape: " + java.util.Arrays.toString(ds.getFeatures().shape()));
                    System.out.println("    Labels shape:   " + java.util.Arrays.toString(ds.getLabels().shape()));
                    System.out.println("    Features (row 0): " + ds.getFeatures().getRow(0));
                    System.out.println("    Labels (row 0):   " + ds.getLabels().getRow(0) + " (one-hot)");
                }
                batchCount++;
            }
            System.out.println("  Total batches: " + batchCount);

            reader.close();
        }

        // ============================================================
        // 5. NORMALIZER: STANDARD SCALER (z-score)
        // ============================================================
        System.out.println("\n=== NormalizerStandardize (z-score) ===");
        {
            RecordReader reader = new CSVRecordReader(1, ',');
            reader.initialize(new FileSplit(csvFile));
            DataSetIterator iterator = new RecordReaderDataSetIterator(reader, 150, 4, 3);

            // Collect all data to fit normalizer
            DataSet allData = iterator.next();

            // Fit z-score normalizer: mean=0, std=1
            NormalizerStandardize normalizer = new NormalizerStandardize();
            normalizer.fit(allData);

            System.out.println("  Before normalization:");
            System.out.println("    Feature means: " + allData.getFeatures().mean(0));
            System.out.println("    Feature stds:  " + allData.getFeatures().std(0));

            // Apply normalization
            normalizer.transform(allData);

            System.out.println("  After normalization:");
            System.out.println("    Feature means: " + allData.getFeatures().mean(0) + " (should be ~0)");
            System.out.println("    Feature stds:  " + allData.getFeatures().std(0) + " (should be ~1)");

            // Revert normalization (for interpretability)
            normalizer.revert(allData);
            System.out.println("  After revert:");
            System.out.println("    Feature means: " + allData.getFeatures().mean(0) + " (original scale)");

            reader.close();
        }

        // ============================================================
        // 6. NORMALIZER: MIN-MAX SCALER
        // ============================================================
        System.out.println("\n=== NormalizerMinMaxScaler ===");
        {
            RecordReader reader = new CSVRecordReader(1, ',');
            reader.initialize(new FileSplit(csvFile));
            DataSetIterator iterator = new RecordReaderDataSetIterator(reader, 150, 4, 3);
            DataSet allData = iterator.next();

            // Scale to [0, 1] range
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
            normalizer.fit(allData);

            System.out.println("  Before normalization:");
            System.out.println("    Min values: " + allData.getFeatures().min(0));
            System.out.println("    Max values: " + allData.getFeatures().max(0));

            normalizer.transform(allData);

            System.out.println("  After [0,1] scaling:");
            System.out.println("    Min values: " + allData.getFeatures().min(0) + " (should be ~0)");
            System.out.println("    Max values: " + allData.getFeatures().max(0) + " (should be ~1)");

            // Can also scale to custom range
            NormalizerMinMaxScaler custom = new NormalizerMinMaxScaler(-1, 1);
            normalizer.revert(allData);
            custom.fit(allData);
            custom.transform(allData);
            System.out.println("  After [-1,1] scaling:");
            System.out.println("    Min values: " + allData.getFeatures().min(0) + " (should be ~-1)");
            System.out.println("    Max values: " + allData.getFeatures().max(0) + " (should be ~1)");

            reader.close();
        }

        // ============================================================
        // 7. COMPLETE PIPELINE EXAMPLE
        // ============================================================
        System.out.println("\n=== Complete Pipeline: CSV -> Normalized DataSet ===");
        {
            // Step 1: Define schema
            Schema schema = new Schema.Builder()
                    .addColumnDouble("sepal_length")
                    .addColumnDouble("sepal_width")
                    .addColumnDouble("petal_length")
                    .addColumnDouble("petal_width")
                    .addColumnInteger("species")
                    .build();

            // Step 2: Read CSV
            RecordReader reader = new CSVRecordReader(1, ',');
            reader.initialize(new FileSplit(csvFile));

            // Step 3: Create iterator
            int labelIndex = 4;
            int numClasses = 3;
            int batchSize = 50;
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    reader, batchSize, labelIndex, numClasses);

            // Step 4: Fit normalizer on first pass
            DataSet firstBatch = iterator.next();
            NormalizerStandardize normalizer = new NormalizerStandardize();
            normalizer.fitLabel(false); // don't normalize labels
            normalizer.fit(firstBatch);

            // Step 5: Apply normalizer to iterator (automatic for future batches)
            iterator.reset();
            iterator.setPreProcessor(normalizer);

            // Step 6: Use normalized data
            System.out.println("  Pipeline: CSV -> CSVRecordReader -> RecordReaderDataSetIterator -> NormalizerStandardize");
            System.out.println("  Processing batches:");
            int count = 0;
            while (iterator.hasNext()) {
                DataSet batch = iterator.next();
                count++;
                System.out.println("    Batch " + count + ": features " +
                        java.util.Arrays.toString(batch.getFeatures().shape()) +
                        ", labels " + java.util.Arrays.toString(batch.getLabels().shape()));
            }
            System.out.println("  Total batches processed: " + count);

            reader.close();
        }

        // Cleanup
        for (File f : tempDir.listFiles()) {
            f.delete();
        }
        tempDir.delete();

        System.out.println("\nAll data pipeline operations demonstrated successfully.");
    }

    /**
     * Creates a sample CSV file with Iris-like data for demonstration.
     */
    private static File createSampleCSV(File dir) throws Exception {
        File csvFile = new File(dir, "sample_iris.csv");
        try (PrintWriter pw = new PrintWriter(new FileWriter(csvFile))) {
            pw.println("sepal_length,sepal_width,petal_length,petal_width,species");

            // Generate synthetic Iris-like data (3 classes, 50 samples each)
            java.util.Random rng = new java.util.Random(42);

            // Class 0: Setosa-like
            for (int i = 0; i < 50; i++) {
                pw.printf("%.1f,%.1f,%.1f,%.1f,%d%n",
                        5.0 + rng.nextGaussian() * 0.4,
                        3.4 + rng.nextGaussian() * 0.4,
                        1.5 + rng.nextGaussian() * 0.2,
                        0.2 + rng.nextGaussian() * 0.1,
                        0);
            }
            // Class 1: Versicolor-like
            for (int i = 0; i < 50; i++) {
                pw.printf("%.1f,%.1f,%.1f,%.1f,%d%n",
                        5.9 + rng.nextGaussian() * 0.5,
                        2.8 + rng.nextGaussian() * 0.3,
                        4.3 + rng.nextGaussian() * 0.5,
                        1.3 + rng.nextGaussian() * 0.2,
                        1);
            }
            // Class 2: Virginica-like
            for (int i = 0; i < 50; i++) {
                pw.printf("%.1f,%.1f,%.1f,%.1f,%d%n",
                        6.6 + rng.nextGaussian() * 0.6,
                        3.0 + rng.nextGaussian() * 0.3,
                        5.6 + rng.nextGaussian() * 0.6,
                        2.0 + rng.nextGaussian() * 0.3,
                        2);
            }
        }
        return csvFile;
    }
}

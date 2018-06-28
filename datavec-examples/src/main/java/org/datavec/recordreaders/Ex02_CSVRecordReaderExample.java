package org.datavec.recordreaders;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.split.StringSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.FileInputStream;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link org.datavec.api.records.reader.impl.csv.CSVRecordReader} is an implementation of
 * {@link org.datavec.api.records.reader.impl.LineRecordReader} and is used for reading CSV data line by line.
 * It takes an {@link org.datavec.api.split.InputSplit} along with the delimiter and quote characters to
 * successfully read the csv data.
 */
public class Ex02_CSVRecordReaderExample {
    public static void main(String[] args) throws Exception {
        //=====================================================================
        //     Example 0: CSVRecordReader Constructors and initializations
        //=====================================================================
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader = new CSVRecordReader(5);
        csvRecordReader = new CSVRecordReader(',');
        csvRecordReader = new CSVRecordReader(5, ',');
        csvRecordReader = new CSVRecordReader(5, ',', '"');

        String data = "a, b, c, d, e\nf, g, h, i, j\nk, l, m, n, o, p\nq, r, s, t, u\nv, w, x, y, z";
        StringSplit stringSplit = new StringSplit(data);

        Configuration configuration = new Configuration();
        configuration.set(CSVRecordReader.SKIP_NUM_LINES, "10");
        configuration.set(CSVRecordReader.DELIMITER, ",");
        configuration.set(CSVRecordReader.QUOTE, "\"");

        csvRecordReader.initialize(stringSplit);
        csvRecordReader.initialize(configuration, stringSplit);

        //=====================================================================
        //          Example 1: CSVRecordReader with a single file
        //=====================================================================

        /*
          This example shows the working of a CSVRecordReader with a single file. It will read the file
          line by line till it reaches the end.
         */
        ClassPathResource classPathResource1 = new ClassPathResource("JoinExample/CustomerPurchases.csv");
        FileSplit fileSplit1 = new FileSplit(
            classPathResource1.getFile()
        );

        CSVRecordReader CSVRecordReader1 = new CSVRecordReader();
        CSVRecordReader1.initialize(fileSplit1);

        System.out.println("--------------- Example 1: CSVRecordReader with a single file ---------------");
        int i = 0;
        while (CSVRecordReader1.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, CSVRecordReader1.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 2: CSVRecordReader with a listener
        //=====================================================================

        /*
          You can also add listeners to your record readers to see what record are being read.
          Make sure you have log4j.properties setup in the resource folder prior to using LogRecordListener
         */
        InputStreamInputSplit inputStreamInputSplit = new InputStreamInputSplit(new FileInputStream(classPathResource1.getFile()));
        CSVRecordReader CSVRecordReader2 = new CSVRecordReader();
        //
        CSVRecordReader2.setListeners(new LogRecordListener());
        CSVRecordReader2.initialize(inputStreamInputSplit);

        System.out.println("--------------- Example 2: CSVRecordReader with a listener ---------------");
        while (CSVRecordReader2.hasNext()) {
            CSVRecordReader2.next(); // This will put the logs on the console as a record is read
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 3: CSVRecordReader with multiple files
        //=====================================================================
        /*
          Every file in the FileSplit will be loaded one by one and will be read line by line
         */
        FileSplit fileSplit2 = new FileSplit(
            new ClassPathResource("JoinExample").getFile().getParentFile(), // The parent "resources" folder path
            new String[]{ "csv" }
        );

        CSVRecordReader CSVRecordReader3 = new CSVRecordReader();
        CSVRecordReader3.initialize(fileSplit2);

        System.out.println("--------------- Example 3: CSVRecordReader with multiple files ---------------");
        i = 0;
        while (CSVRecordReader3.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, CSVRecordReader3.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 4: Reading records
        //=====================================================================
        /*
          When you read Record from a RecordReader, it gives you the metadata along with the List of Writables.
          The metadata contains the line number and file URI for the data source in the case of CSVRecordReader
         */

        // This is for the next example when we'll load the Records from a list of RecordMetaData
        List<RecordMetaData> recordMetaDataLineList = new ArrayList<>();

        // Reset the iterators and other internal states for the CSVRecordReader
        if (CSVRecordReader3.resetSupported()) {
            CSVRecordReader3.reset();
        }

        System.out.println("--------------- Example 4: Reading records ---------------");
        i = 1;
        String fileName = null;
        while (CSVRecordReader3.hasNext()) {
            Record record = CSVRecordReader3.nextRecord();
            RecordMetaDataLine recordMetaData = (RecordMetaDataLine) record.getMetaData();

            // This is for the next example when we'll load the Records from a list of RecordMetaData
            recordMetaDataLineList.add(recordMetaData);

            String path = recordMetaData.getURI().getPath();
            if(fileName == null || !fileName.equals(path)) {
                fileName = path;
                System.out.println(MessageFormat.format("\nReading from file {0}\n----", fileName));
            }
            System.out.println(MessageFormat.format("Total Lines Read {0} - Line in file {1} | Record: {2}",
                i++, recordMetaData.getLineNumber(), record.getRecord()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 5: Read records from a list of RecordMetaData
        //=====================================================================
        /*
          If you have a list of RecordMetaData (Preferably, RecordMetaDataLine in this case), then you can
          load all the metadata records by feeding the list to CSVRecordReader
         */
        List<Record> recordList = CSVRecordReader3.loadFromMetaData(recordMetaDataLineList);

        System.out.println("--------------- Example 5: Read records from a list of RecordMetaData ---------------");
        i = 1;
        fileName = null;
        for (Record record : recordList) {
            RecordMetaDataLine recordMetaData = (RecordMetaDataLine) record.getMetaData();

            String path = recordMetaData.getURI().getPath();
            if(fileName == null || !fileName.equals(path)) {
                fileName = path;
                System.out.println(MessageFormat.format("\nReading from file {0}\n----", fileName));
            }
            System.out.println(MessageFormat.format("Total Lines Read {0} - Line in file {1} | Record: {2}",
                i++, recordMetaData.getLineNumber(), record.getRecord()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }
}

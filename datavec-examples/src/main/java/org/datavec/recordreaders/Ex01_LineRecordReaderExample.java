package org.datavec.recordreaders;

import org.datavec.api.records.Record;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.FileInputStream;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link LineRecordReader} in an implementation of {@link org.datavec.api.records.reader.BaseRecordReader}
 * which is implemented from the interface {@link org.datavec.api.records.reader.RecordReader}. It
 * read records line by line, given any type of file. {@link LineRecordReader} takes an
 * implementation of {@link InputSplit} in its {@link LineRecordReader#initialize(InputSplit)}
 * method to read from the data source. In case of multiple files/locations in an {@link InputSplit},
 * {@link LineRecordReader} will go through each file/location one by one and read it line by line.
 */
public class Ex01_LineRecordReaderExample {
    public static void main(String[] args) throws Exception {
        //=====================================================================
        //          Example 1: LineRecordReader with a single file
        //=====================================================================

        /*
          This example shows the working of a LineRecordReader with a single file. It will read the file
          line by line till it reaches the end.
         */
        ClassPathResource classPathResource1 = new ClassPathResource("JoinExample/CustomerPurchases.csv");
        FileSplit fileSplit1 = new FileSplit(
            classPathResource1.getFile()
        );

        LineRecordReader lineRecordReader1 = new LineRecordReader();
        lineRecordReader1.initialize(fileSplit1);

        System.out.println("--------------- Example 1: LineRecordReader with a single file ---------------");
        int i = 0;
        while (lineRecordReader1.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, lineRecordReader1.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 2: LineRecordReader with a listener
        //=====================================================================

        /*
          You can also add listeners to your record readers to see what record are being read.
          Make sure you have log4j.properties setup in the resource folder prior to using LogRecordListener
         */
        InputStreamInputSplit inputStreamInputSplit = new InputStreamInputSplit(new FileInputStream(classPathResource1.getFile()));
        LineRecordReader lineRecordReader2 = new LineRecordReader();
        //
        lineRecordReader2.setListeners(new LogRecordListener());
        lineRecordReader2.initialize(inputStreamInputSplit);

        System.out.println("--------------- Example 2: LineRecordReader with a listener ---------------");
        while (lineRecordReader2.hasNext()) {
            lineRecordReader2.next(); // This will put the logs on the console as a record is read
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 3: LineRecordReader with multiple files
        //=====================================================================
        /*
          Every file in the FileSplit will be loaded one by one and will be read line by line
         */
        FileSplit fileSplit2 = new FileSplit(
            new ClassPathResource("JoinExample").getFile().getParentFile(), // The parent "resources" folder path
            new String[]{ "csv" }
        );

        LineRecordReader lineRecordReader3 = new LineRecordReader();
        lineRecordReader3.initialize(fileSplit2);

        System.out.println("--------------- Example 3: LineRecordReader with multiple files ---------------");
        i = 0;
        while (lineRecordReader3.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, lineRecordReader3.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 4: Reading records
        //=====================================================================
        /*
          When you read Record from a RecordReader, it gives you the metadata along with the List of Writables.
          The metadata contains the line number and file URI for the data source in the case of LineRecordReader
         */

        // This is for the next example when we'll load the Records from a list of RecordMetaData
        List<RecordMetaData> recordMetaDataLineList = new ArrayList<>();

        // Reset the iterators and other internal states for the LineRecordReader
        if (lineRecordReader3.resetSupported()) {
            lineRecordReader3.reset();
        }

        System.out.println("--------------- Example 4: Reading records ---------------");
        i = 1;
        String fileName = null;
        while (lineRecordReader3.hasNext()) {
            Record record = lineRecordReader3.nextRecord();
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
          If you have a list of RecordMetaData, then you can
          load all the metadata records by feeding the list to LineRecordReader
         */
        List<Record> recordList = lineRecordReader3.loadFromMetaData(recordMetaDataLineList);

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

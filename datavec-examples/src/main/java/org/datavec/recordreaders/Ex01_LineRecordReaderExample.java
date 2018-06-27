package org.datavec.recordreaders;

import org.datavec.api.records.Record;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.FileInputStream;
import java.text.MessageFormat;

public class Ex01_LineRecordReaderExample {
    public static void main(String[] args) throws Exception {
        //=====================================================================
        //          Example 1: LineRecordReader with a single file
        //=====================================================================

        /*
          This example shows the working of a LineRecordReader with a single file. It will read the file
          line by line till it reaches the end.
         */
        ClassPathResource classPathResource = new ClassPathResource("JoinExample/CustomerPurchases.csv");
        FileSplit fileSplit = new FileSplit(
            classPathResource.getFile()
        );

        LineRecordReader lineRecordReader = new LineRecordReader();
        lineRecordReader.initialize(fileSplit);

        System.out.println("--------------- Example 1: LineRecordReader with a single file ---------------");
        int i = 0;
        while (lineRecordReader.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, lineRecordReader.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 2: LineRecordReader with a listener
        //=====================================================================

        /*
          You can also add listeners to your record readers to see what record are being read.
          Make sure you have log4j.properties setup in the resource folder prior to using LogRecordListener
         */
        InputStreamInputSplit inputStreamInputSplit = new InputStreamInputSplit(new FileInputStream(classPathResource.getFile()));
        LineRecordReader lineRecordReader1 = new LineRecordReader();
        //
        lineRecordReader1.setListeners(new LogRecordListener());
        lineRecordReader1.initialize(inputStreamInputSplit);

        System.out.println("--------------- Example 2: LineRecordReader with a listener ---------------");
        while (lineRecordReader1.hasNext()) {
            lineRecordReader1.next(); // This will put the logs on the console as a record is read
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }
}

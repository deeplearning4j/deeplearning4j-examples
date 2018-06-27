package org.datavec.recordreaders;

import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
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
    }
}

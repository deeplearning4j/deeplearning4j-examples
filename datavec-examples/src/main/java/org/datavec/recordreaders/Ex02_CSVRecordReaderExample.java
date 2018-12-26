package org.datavec.recordreaders;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRegexRecordReader;
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
        CSVRecordReader csvRecordReader = new CSVRecordReader(); // No arguments
        csvRecordReader = new CSVRecordReader(5); // Number of lines to skip
        csvRecordReader = new CSVRecordReader(','); // Delimiter character
        csvRecordReader = new CSVRecordReader(5, ','); // Number of lines to skip with delimiter character
        csvRecordReader = new CSVRecordReader(5, ',', '"'); // Number of lines to skip with delimiter and quote characters

        // Creating a string split for input
        String data = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z";
        StringSplit stringSplit = new StringSplit(data);

        // CSVRecordReader configurations
        Configuration configuration = new Configuration();
        configuration.set(CSVRecordReader.SKIP_NUM_LINES, "0");
        configuration.set(CSVRecordReader.DELIMITER, ",");
        configuration.set(CSVRecordReader.QUOTE, "\"");

        csvRecordReader.initialize(stringSplit); // Initialize with Input Split
        csvRecordReader.initialize(configuration, stringSplit); // Initialize with Configuration object and Input Split
        //=====================================================================
        //          Example 1: CSVRecordReader with a single file
        //=====================================================================

        /*
          This example shows the working of a CSVRecordReader with a single file. It will read the file
          line by line and parse the CSV data based on the delimiter and quote character specified in the
          constructor. If the no argument constructors are used then the default delimiter and quote characters
          are used, under the variables CSVRecordReader.DEFAULT_DELIMITER and CSVRecordReader.DEFAULT_QUOTE,
          respectively.
         */
        ClassPathResource classPathResource1 = new ClassPathResource("JoinExample/CustomerPurchases.csv");
        FileSplit fileSplit1 = new FileSplit(
            classPathResource1.getFile()
        );

        CSVRecordReader csvRecordReader1 = new CSVRecordReader();
        csvRecordReader1.initialize(fileSplit1);

        System.out.println("--------------- Example 1: CSVRecordReader with a single file ---------------");
        int i = 0;
        while (csvRecordReader1.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, csvRecordReader1.next()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 2: CSVRecordReader with a listener
        //=====================================================================

        /*
          You can also add listeners to your record readers to see what record are being read.
          Make sure you have log4j.properties setup in the resource folder prior to using LogRecordListener.
          The LogRecordListener will print the non CSV parsed raw line data read from the source.
         */
        InputStreamInputSplit inputStreamInputSplit = new InputStreamInputSplit(new FileInputStream(classPathResource1.getFile()));
        CSVRecordReader csvRecordReader2 = new CSVRecordReader();

        csvRecordReader2.setListeners(new LogRecordListener());
        csvRecordReader2.initialize(inputStreamInputSplit);

        System.out.println("--------------- Example 2: CSVRecordReader with a listener ---------------");
        while (csvRecordReader2.hasNext()) {
            csvRecordReader2.next(); // This will put the logs on the console as a record is read
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 3: CSVRecordReader with multiple files
        //=====================================================================
        /*
          Every file in the FileSplit will be loaded one by one and will be read line by line and get parsed.
         */
        FileSplit fileSplit2 = new FileSplit(
            new ClassPathResource("JoinExample").getFile().getParentFile(), // The parent "resources" folder path
            new String[]{ "csv" }
        );

        CSVRecordReader csvRecordReader3 = new CSVRecordReader();
        csvRecordReader3.initialize(fileSplit2);

        System.out.println("--------------- Example 3: CSVRecordReader with multiple files ---------------");
        i = 0;
        while (csvRecordReader3.hasNext()) {
            System.out.println(MessageFormat.format("Line {0}: {1}", i++, csvRecordReader3.next()));
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
        if (csvRecordReader3.resetSupported()) {
            csvRecordReader3.reset();
        }

        System.out.println("--------------- Example 4: Reading records ---------------");
        i = 1;
        String fileName = null;
        while (csvRecordReader3.hasNext()) {
            Record record = csvRecordReader3.nextRecord();
            RecordMetaDataLine recordMetaData = (RecordMetaDataLine) record.getMetaData();

            // This is for the next example when we'll load the Records from a list of RecordMetaData
            recordMetaDataLineList.add(recordMetaData);

            String path = recordMetaData.getURI().getPath();
            if (fileName == null || !fileName.equals(path)) {
                fileName = path;
                System.out.println(MessageFormat.format("\nReading from file {0}\n----", fileName));
            }
            System.out.println(MessageFormat.format("Total Lines Parsed {0} - Line in file {1} | Record: {2}",
                i++, recordMetaData.getLineNumber(), record.getRecord()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 5: Read records from a list of RecordMetaData
        //=====================================================================
        /*
          If you have a list of RecordMetaData, then you can
          load all the metadata records by feeding the list to CSVRecordReader
         */
        List<Record> recordList = csvRecordReader3.loadFromMetaData(recordMetaDataLineList);

        System.out.println("--------------- Example 5: Read records from a list of RecordMetaData ---------------");
        i = 1;
        fileName = null;
        for (Record record : recordList) {
            RecordMetaDataLine recordMetaData = (RecordMetaDataLine) record.getMetaData();

            String path = recordMetaData.getURI().getPath();
            if (fileName == null || !fileName.equals(path)) {
                fileName = path;
                System.out.println(MessageFormat.format("\nReading from file {0}\n----", fileName));
            }
            System.out.println(MessageFormat.format("Total Lines Read {0} - Line in file {1} | Record: {2}",
                i++, recordMetaData.getLineNumber(), record.getRecord()));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //          Example 6: Using CSVRegexRecordReader
        //=====================================================================
        /*
          A CSVRecordReader that can split each column into additional columns using a RegEx pattern for each column.
          A RegEx is a short form for a regular expression. A regular expression is a sequence of characters that define
          a search pattern. You can learn more about it here: https://en.wikipedia.org/wiki/Regular_expression.
          Also, look at the javadocs for the Pattern class here:
          https://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html

          This could be useful for splitting data, such as date/time into their constituting elements.
          For example: A date of pattern [yyyy-MM-dd HH:mm:ss.SSS] can be split apart into it's components from
          [date] -> [year, month, day, hours, minutes, seconds, milliseconds]
          with a regex that defines the group as (\d+)-(\d+)-(\d+)\s+(\d+):(\d+):(\d+)\.(\d+).
          Where:
          - '\d+' represents one or more digits
          - '\s+' represents one or more space characters and
          - '\.' defines a period (.)
          - The parenthesis contains the part of the regex that we want to extract out of the matched string.

          See the example below, in which the date (first column) is split apart into its components. The other
          regex strings are null as they don't required to be split.
         */

        CSVRegexRecordReader csvRegexRecordReader = new CSVRegexRecordReader(
            0, ",", "\"",
            new String[]{
                "(\\d+)-(\\d+)-(\\d+)\\s+(\\d+):(\\d+):(\\d+)\\.(\\d+)",
                null, null, null, null, null, null}
                );

        csvRegexRecordReader.initialize(
            new FileSplit(
                new ClassPathResource("BasicDataVecExample/exampledata.csv").getFile()
            )
        );

        System.out.println("--------------- Example 6: Using CSVRegexRecordReader ---------------");
        while (csvRegexRecordReader.hasNext()) {
            System.out.println(csvRegexRecordReader.next());
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }
}

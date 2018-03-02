package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.records.converter.RecordReaderConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.writer.mapfile.MapFileRecordWriter;

import java.io.File;
import java.util.Random;

/**
 * A simple example on how to convert a CSV (text format) to a Hadoop MapFile
 *
 * Why would you want to do this?
 * 1. Faster: reading from a map file is much faster than reading from a CSV (parse once, plus reading binary format)
 * 2. Randomization: MapFileRecordReader supports randomization of iteration order
 *
 * Note that MapFileRecordReader/Writer are in datavec-hadoop package. You will also likely need hadoop-common.
 *
 * @author Alex Black
 */
public class MapFileConversion {

    public static void main(String[] args) throws Exception {

        //Create CSV reader
        File irisFile = new ClassPathResource("iris.txt").getFile();
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(irisFile));

        //Create map file writer
        File mapFileLoc = new File("mapFileOutputDir");
        MapFileRecordWriter writer = new MapFileRecordWriter(mapFileLoc);

        //Convert to MapFile binary format:
        RecordReaderConverter.convert(recordReader, writer);


        //Read back in from binary MapFile, random order:
        Random rng = new Random(12345);
        RecordReader rr1 = new MapFileRecordReader(rng);
        rr1.initialize(new FileSplit(mapFileLoc));

        //Print out:
        while(rr1.hasNext()){
            System.out.println(rr1.next());
        }

    }

}

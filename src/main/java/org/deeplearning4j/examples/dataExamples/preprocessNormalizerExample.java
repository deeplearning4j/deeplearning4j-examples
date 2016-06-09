package org.deeplearning4j.examples.dataExamples;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

/**
 * This basic example demonstrates how to use the preprocessors available
 * 3.10 release works with only select iterators that have a valid total ex function
 * Later releases and current master will work with all iterators.
 * Run this example with current master
 * Created by susaneraly on 6/8/16.
 */
public class preprocessNormalizerExample {

    private static Logger log = LoggerFactory.getLogger(preprocessNormalizerExample.class);

    public static void main(String[] args) throws  Exception {

        //========= This section is to create a dataset and a dataset iterator from the iris dataset stored in csv =============
        //                               Refer to the csv example for details
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderA = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderB = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderA.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderB.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;
        DataSetIterator iteratorA = new RecordReaderDataSetIterator(recordReaderA,10,labelIndex,numClasses);
        DataSetIterator iteratorB = new RecordReaderDataSetIterator(recordReaderB,10,labelIndex,numClasses);
        DataSetIterator fulliterator = new RecordReaderDataSetIterator(recordReader,150,labelIndex,numClasses);
        DataSet datasetX = fulliterator.next();
        DataSet datasetY = datasetX.copy();

        // We now have datasetX, datasetY, iteratorA, iteratorB all of which have the iris dataset loaded
        // iteratorA and iteratorB have batchsize of 10. So the full dataset is 150/10 = 15 batches
        //=====================================================================================================================

        log.info("INFO: All preprocessors have to be fit to the intended metrics before they can be used to transform");
        log.info("INFO: To have a transformation occur each time next on the iterator is called use the 'setpreprocessor', eg very end here\n");
        log.info("This example demonstrates preprocessor use with the standard normalizer. Refer javadocs for MinMaxScaler");

        log.info("INFO: Standardizing - subtracts the mean and divides by the standard deviation");
        log.info("INFO: Find further information here: https://en.wikipedia.org/wiki/Standard_score");
        log.info("INFO: Instantiating a standardizing preprocessor...\n");

        NormalizerStandardize preProcessor = new NormalizerStandardize();
        log.info("INFO: During 'fit' the preprocessor calculates the metrics (std dev and mean for the standardizer) from the data given");
        log.info("INFO: Fit can take a dataset or a dataset iterator\n");

        //Fitting a preprocessor with a dataset
        log.info("INFO: Fitting with a dataset...............");
        preProcessor.fit(datasetX);
        log.info("INFO: Calculated metrics");
        log.info("INFO: Mean:"+ preProcessor.getMean().toString());
        log.info("INFO: Std dev:" + preProcessor.getStd().toString()+"\n");

        log.info("INFO: Once fit the preprocessor can be used to transform data wrt to the metrics of the dataset it was fit to");
        log.info("INFO: Transform takes a dataset and modifies it in place");

        log.info("INFO:Transforming a dataset, printing only the first ten.....");
        preProcessor.transform(datasetX);
        log.info("\n"+datasetX.getRange(0,9).toString()+"\n");

        log.info("INFO: Transformed datasets can be reverted back as well...");
        log.info("INFO: Note the reverting happens in place.");
        log.info("INFO: Reverting back the dataset, printing only the first ten.....");
        preProcessor.revert(datasetX);
        log.info("\n"+datasetX.getRange(0,9).toString()+"\n\n");

        //Setting a preprocessor in an iterator
        log.info("INFO: Fitting the preprocessor with iteratorB......");
        NormalizerStandardize preProcessorIter = new NormalizerStandardize();
        preProcessorIter.fit(iteratorB);
        log.info("A fitted preprocessor can be set to an iterator so each time next is called the transform step happens automatically");
        log.info("Setting a preprocessor for iteratorA");
        iteratorA.setPreProcessor(preProcessorIter);
        while (iteratorA.hasNext()) {
            log.info("Calling next on iterator A that has a preprocessor on it");
            log.info("\n"+iteratorA.next().toString());
            log.info("Calling next on iterator B that has no preprocessor on it");
            log.info("\n"+iteratorB.next().toString());
            log.info("Note the data is different - iteratorA is standardized, iteratorB is not");
            log.info("Now using transform on the next datset on iteratorB");
            iteratorB.reset();
            preProcessorIter.transform(iteratorB.next());
            log.info("\n"+iteratorB.next().toString());
            log.info("Note that this now gives the same results");
            break;
        }

        log.info("INFO: If you are using batches and an iterator, set the preprocessor on your iterator to transform data automatically when next is called");
        log.info("INFO: Use the .transform function only if you are working with a small dataset and no iterator");

    }
}

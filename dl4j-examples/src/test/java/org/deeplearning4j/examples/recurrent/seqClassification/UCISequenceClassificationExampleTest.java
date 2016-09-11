package org.deeplearning4j.examples.recurrent.seqClassification;

import org.datavec.api.util.ClassPathResource;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

/**
 * Created by claytongraham on 9/11/16.
 */
public class UCISequenceClassificationExampleTest {


    @Before
    public void setUp() throws FileNotFoundException {
        //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
        UCISequenceClassificationExample.baseDir = new ClassPathResource("/recurrent/seqClassification/UCISequence/uci/").getFile();
        UCISequenceClassificationExample.baseTrainDir = new File(UCISequenceClassificationExample.baseDir, "train");
        UCISequenceClassificationExample.featuresDirTrain = new File(UCISequenceClassificationExample.baseTrainDir, "features");
        UCISequenceClassificationExample.labelsDirTrain = new File(UCISequenceClassificationExample.baseTrainDir, "labels");
        UCISequenceClassificationExample.baseTestDir = new File(UCISequenceClassificationExample.baseDir, "test");
        UCISequenceClassificationExample.featuresDirTest = new File(UCISequenceClassificationExample.baseTestDir, "features");
        UCISequenceClassificationExample.labelsDirTest = new File(UCISequenceClassificationExample.baseTestDir, "labels");
    }

    @Test
    public void testSequenceClassification(){
        String[] args = {};
        try {

            Map<Integer,Map<String,Object>>  classifiedTestData =
                    UCISequenceClassificationExample.trainNetworkAndMapTestClassifications(args);

            Assert.assertEquals("Cyclic",classifiedTestData.get(0).get("classification").toString());

        } catch (IOException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } catch (InterruptedException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }


    }

}

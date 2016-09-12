package org.deeplearning4j.examples.recurrent.seqClassification;

import org.datavec.api.util.ClassPathResource;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

            Assert.assertEquals("Cyclic",classifiedTestData.get(0).get("classificationName").toString());
            Assert.assertEquals("Upward shift",classifiedTestData.get(1).get("classificationName").toString());
            Assert.assertEquals("Downward shift",classifiedTestData.get(2).get("classificationName").toString());
            Assert.assertEquals("Downward shift",classifiedTestData.get(3).get("classificationName").toString());
            Assert.assertEquals("Upward shift",classifiedTestData.get(4).get("classificationName").toString());
            Assert.assertEquals("Upward shift",classifiedTestData.get(5).get("classificationName").toString());
            Assert.assertEquals("Increasing trend",classifiedTestData.get(6).get("classificationName").toString());
            Assert.assertEquals("Cyclic",classifiedTestData.get(7).get("classificationName").toString());
            Assert.assertEquals("Increasing trend",classifiedTestData.get(8).get("classificationName").toString());
            Assert.assertEquals("Normal",classifiedTestData.get(9).get("classificationName").toString());


        } catch (IOException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } catch (InterruptedException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }


    }

}

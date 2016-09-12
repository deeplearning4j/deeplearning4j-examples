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
    public void testSimpleN4j(){
        INDArray arr1 = Nd4j.create(new float[]{1,2,4,3}, new int[]{2,2});
        INDArray arr1Max = Nd4j.argMax(arr1, 1);
        Assert.assertTrue(1.0==arr1Max.getFloat(0));
        Assert.assertTrue(0.0==arr1Max.getFloat(1));

        INDArray arr2 = Nd4j.create(new float[]{1,2,4,3,4,6,7,8}, new int[]{2,2,2});
        INDArray arr2Max = Nd4j.argMax(arr2, 2);
        Assert.assertTrue(1.0==arr2Max.getRow(0).getFloat(0));
        Assert.assertTrue(0.0==arr2Max.getRow(0).getFloat(1));
        Assert.assertTrue(1.0==arr2Max.getRow(1).getFloat(0));
        Assert.assertTrue(1.0==arr2Max.getRow(1).getFloat(1));
    }

    @Test
    public void testSequenceClassification(){
        String[] args = {};
        try {

            Map<Integer,Map<String,Object>>  classifiedTestData =
                    UCISequenceClassificationExample.trainNetworkAndMapTestClassifications(args);

            Assert.assertEquals("Cyclic",classifiedTestData.get(0).get("classificationName").toString());

        } catch (IOException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } catch (InterruptedException e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }


    }

}

package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * @Description  a->b b->c c->d ....
 * @Author WangFeng
 */
public class LotteryCharacterSequenceDataSetReader implements BaseDataSetReader {

    private Iterator<String> iter;
    private Path filePath;

    private int totalExamples;
    private int currentCursor;


    public LotteryCharacterSequenceDataSetReader() {}
    public LotteryCharacterSequenceDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
    }

    public void doInitialize(){
        List<String> dataLines;
        try {
            dataLines = Files.readAllLines(filePath, Charset.forName("UTF-8"));
        } catch (Exception e) {
            throw new RuntimeException("loading data failed");
        }
        iter = dataLines.iterator();
        totalExamples = dataLines.size();
        currentCursor = 0;
    }

    public DataSet next(int num) {
        //Though ‘c’ order arrays will also work, performance will be reduced due to the need to copy the arrays to ‘f’ order first, for certain operations.
        INDArray features = Nd4j.create(new int[]{num, 10, 16}, 'f');
        INDArray labels = Nd4j.create(new int[]{num, 10, 16}, 'f');


        INDArray featuresMask = null;
        INDArray labelsMask = null;
        for (int i =0; i < num && iter.hasNext(); i ++) {
            String featureStr = iter.next();
            currentCursor ++;
            featureStr = featureStr.replaceAll(",", "");
            String[] featureAry = featureStr.split("");
            for (int j = 0; j < featureAry.length - 1; j ++) {
                int feature = Integer.parseInt(featureAry[j]);
                int label = Integer.parseInt(featureAry[j + 1]);
                features.putScalar(new int[]{i, feature, j}, 1.0);
                labels.putScalar(new int[]{i, label, j}, 1.0);
            }
        }
        DataSet result = new DataSet(features, labels, featuresMask, labelsMask);
        return result;

    }

    public boolean hasNext() {
        if (iter != null && iter.hasNext()) {
            return true;
        }
        return false;
    }

    public List<String> getLabels() {
        return null;
    }

    public void reset() {
        doInitialize();
    }
    public int totalExamples() {
        return totalExamples;
    }
    public int cursor() {
        return currentCursor;
    }



}


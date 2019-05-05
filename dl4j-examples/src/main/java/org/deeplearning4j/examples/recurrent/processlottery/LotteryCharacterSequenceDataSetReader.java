package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;


/**
 * a->b b->c c->d ....
 * @author WangFeng
 */
public class LotteryCharacterSequenceDataSetReader extends BaseDataSetReader {

    public LotteryCharacterSequenceDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
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
        return new DataSet(features, labels, featuresMask, labelsMask);
    }

}


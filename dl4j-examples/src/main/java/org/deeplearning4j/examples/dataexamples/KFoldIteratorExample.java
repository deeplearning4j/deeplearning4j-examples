package org.deeplearning4j.examples.dataexamples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 *
 * @author Robert Altena
 */
public class KFoldIteratorExample {

    public static void main(String[] args) {

        INDArray x = Nd4j.create(new float[][]{{1,2},{2,3},{3,4},{4,5}});
        INDArray y = Nd4j.create(new float[][]{{1},{2},{3},{4}});
        DataSet ds = new DataSet(x,y);

        System.out.println("Full dataset: ");
        System.out.println(ds);

        KFoldIterator kiter = new KFoldIterator(2, ds);
        while (kiter.hasNext()){
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();
            System.out.println();
            System.out.println("Train: ");
            System.out.println(now);
            System.out.println("Test: ");
            System.out.println(test);

        }
    }
}

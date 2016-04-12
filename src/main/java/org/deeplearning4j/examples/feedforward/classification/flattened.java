package org.deeplearning4j.examples.feedforward.classification;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 4/5/16.
 */
public class flattened {

    public static void main(String[] args){

        int nTests = 1000;
        int size = 1024;

        //2d, c->c, c->f, f->c, f->f

        INDArray c2d = Nd4j.create(new int[]{size,size},'c');
        INDArray f2d = Nd4j.create(new int[]{size,size},'f');


        long start = System.currentTimeMillis();
        for( int i=0; i<nTests; i++ ){
            Nd4j.toFlattened('c',c2d);
        }
        long ccTime = System.currentTimeMillis() - start;
        System.out.println("cc");

        start = System.currentTimeMillis();
        for( int i=0; i<nTests; i++ ){
            Nd4j.toFlattened('f',c2d);
        }
        long cfTime = System.currentTimeMillis() - start;
        System.out.println("cf");

        start = System.currentTimeMillis();
        for( int i=0; i<nTests; i++ ){
            Nd4j.toFlattened('c',f2d);
        }
        long fcTime = System.currentTimeMillis() - start;
        System.out.println("fc");

        start = System.currentTimeMillis();
        for( int i=0; i<nTests; i++ ){
            Nd4j.toFlattened('f',f2d);
        }
        long ffTime = System.currentTimeMillis() - start;
        System.out.println("ff");


        System.out.println("c->c: " + ccTime);
        System.out.println("c->f: " + cfTime);
        System.out.println("f->f: " + ffTime);
        System.out.println("f->c: " + fcTime);


    }

}

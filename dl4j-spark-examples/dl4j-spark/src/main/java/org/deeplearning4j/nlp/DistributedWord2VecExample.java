package org.deeplearning4j.nlp;

/**
 * This example shows how to build Word2Vec model with distributed p2p ParameterServer.
 *
 * PLEASE NOTE: This example is NOT meant to be run on localhost, consider spark-submit ONLY
 *
 * @author raver119@gmail.com
 */
public class DistributedWord2VecExample {


    public void entryPoint(String[] args) {

    }

    public static void main(String[] args) throws Exception {
        new DistributedWord2VecExample().entryPoint(args);
    }
}

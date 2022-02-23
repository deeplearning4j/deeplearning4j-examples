package org.deeplearning4j.examples;

import org.deeplearning4j.examples.quickstart.modeling.convolution.LeNetMNIST;
import org.deeplearning4j.examples.quickstart.modeling.variationalautoencoder.VaeMNISTAnomaly;
import org.junit.jupiter.api.Test;

public class QuickTest {
    @Test
    public void test() throws Exception {
        System.out.println("Beginning LenetMNIST");
        LeNetMNIST.main(new String[]{});
        System.out.println("Ending LenetMNIST");
        System.out.println("Beginning VaeMNISTAnomaly");
        //VaeMNISTAnomaly.main(new String[]{});
        //System.out.println("Beginning VaeMNISTAnomaly");

    }

}

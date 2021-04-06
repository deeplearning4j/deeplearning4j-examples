package org.nd4j.examples.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.examples.samediff.quickstart.basics.Ex1_SameDiff_Basics;
import org.nd4j.examples.samediff.quickstart.basics.Ex2_LinearRegression;
import org.nd4j.examples.samediff.quickstart.basics.Ex3_Variables;
import org.nd4j.examples.samediff.quickstart.modeling.MNISTCNN;

public class QuickTest {

    @Test
    public void test() throws Exception {
        System.out.println("Begin running Ex1_SameDiff_Basics");
        Ex1_SameDiff_Basics.main(new String[]{});
        System.out.println("End running Ex1_SameDiff_Basics");
        System.out.println("Begin running Ex2_LinearRegression");
        Ex2_LinearRegression.main(new String[]{});
        System.out.println("End running Ex2_LinearRegression");
        System.out.println("Begin running Ex3_Variables");
        Ex3_Variables.main(new String[]{});
        System.out.println("End running Ex3_Variables");
        System.out.println("Begin running MNISTCNN");
        MNISTCNN.main(new String[]{});
        System.out.println("End running MNISTCNN");

    }

}

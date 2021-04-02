package org.deeplearning4j.modelimportexamples;

import org.deeplearning4j.modelimportexamples.keras.quickstart.SimpleFunctionalMlpImport;
import org.deeplearning4j.modelimportexamples.keras.quickstart.SimpleSequentialMlpImport;
import org.deeplearning4j.modelimportexamples.tf.advanced.mobilenet.ImportMobileNetExample;
import org.deeplearning4j.modelimportexamples.tf.advanced.tfgraphrunnerinjava.TFGraphRunnerExample;
import org.deeplearning4j.modelimportexamples.tf.quickstart.MNISTMLP;
import org.junit.jupiter.api.Test;

public class QuickTest {


    @Test
    public void runExamples() throws Exception {
        System.out.println("Beginning MNISTMLP");
        MNISTMLP.main(new String[]{});
        System.out.println("Ended MNISTMLP");
        System.out.println("Beginning ImportMobileNetExample");
        ImportMobileNetExample.main(new String[]{});
        System.out.println("Ended ImportMobileNetExample");
        TFGraphRunnerExample.main(new String[]{});
        System.out.println("Beginning ImportMobileNetExample");
        SimpleFunctionalMlpImport.main(new String[]{});
        System.out.println("Ended ImportMobileNetExample");
        System.out.println("Beginning ImportMobileNetExample");
        SimpleSequentialMlpImport.main(new String[]{});
        System.out.println("Ended ImportMobileNetExample");

    }

}

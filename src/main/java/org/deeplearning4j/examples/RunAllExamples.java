package org.deeplearning4j.examples;

import java.lang.reflect.*;


/**
 * @author Atilla Ozgur
 */
public class RunAllExamples {

    private static String[] exampleList =  new String[]{
        "org.deeplearning4j.examples.misc.csv.CSVExample" 
        ,"org.deeplearning4j.examples.feedforward.classification.MLPClassifierLinear"
        ,"org.deeplearning4j.examples.feedforward.classification.MLPClassifierMoon" 
        ,"org.deeplearning4j.examples.feedforward.MLPClassifierSaturn" 
        ,"org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample" 
        ,"org.deeplearning4j.examples.feedforward.MLPMnistTwoLayerExample" 
        ,"org.deeplearning4j.examples.feedforward.regression.RegressionSum" 
        ,"org.deeplearning4j.examples.feedforward.regression.RegressionMathFunctions" 
        ,"org.deeplearning4j.examples.misc.earlystopping.EarlyStoppingMNIST" 
        ,"org.deeplearning4j.examples.recurrent.basic.BasicRNNExample" 
        ,"org.deeplearning4j.examples.unsupervised.StackedAutoEncoderMnistExample" 
        ,"org.deeplearning4j.examples.unsupervised.DBNMnistFullExample" 
        ,"org.deeplearning4j.examples.unsupervised.deepbelief.DeepAutoEncoderExample" 
        ,"org.deeplearning4j.examples.xor.XorExample"
        };

    public static void main(String[] args) throws  Exception {

        System.out.println("Hello Deep Learning4J");

        for(String exampleName : exampleList)
        {
            System.out.println("Running:" + exampleName);

            Class<?> cls = Class.forName(exampleName);
            Method mainMethod = cls.getMethod("main", String[].class);
            String[] params = null; 
            mainMethod.invoke(null, (Object) params);


            
        }

    }


}

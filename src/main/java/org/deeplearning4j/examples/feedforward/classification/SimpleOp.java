package org.deeplearning4j.examples.feedforward.classification;

/**
 * Created by susaneraly on 3/31/16.
 */

import java.io.File;
import java.util.Collections;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SimpleOp {
    public static void main(String[] args) throws Exception {
        int sizeXY = 4;
        INDArray myX = Nd4j.ones(sizeXY, sizeXY);
        INDArray myY = Nd4j.ones(sizeXY, sizeXY);
        int myx = 3;
        int myy = 1;
        myX = myX.mul(myx);
        myY = myY.mul(myy);

        int mya = 2;
        int myb = 1;

        System.out.println("X+Y = X^A + Y^B");
        INDArray myXPow = Nd4j.getExecutioner().execAndReturn(new Pow(myX, mya));
        INDArray myYPow = Nd4j.getExecutioner().execAndReturn(new Pow(myY, myb));
        INDArray myXSumY = myXPow.add(myYPow);
        System.out.printf("%d^%d + %d^%d\n", myx, mya, myx, myb);
        System.out.println(myXSumY);

        System.out.println("Multiplying the above negative by a leaky vector..");
        INDArray leakyVector = Nd4j.linspace(-1, 1, sizeXY);
        System.out.println(leakyVector);
        System.out.println("");

        leakyVector = leakyVector.mul(myXSumY.getDouble(0, 0));
        for (int i = 0; i < sizeXY; i++) {
            myXSumY.putRow(i, leakyVector);
        }
        System.out.println(myXSumY);
        System.out.println("");

        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with 0.5 ..");
        System.out.println("======================");
        INDArray out = Nd4j.getExecutioner().execAndReturn(new Pow(myXSumY.dup(), 2));

        System.out.println(out);

        out = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(myXSumY.dup()));
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with no default..");
        System.out.println("======================");
        System.out.println(out);

        // OPFACTORY STUFF
        //        INDArray ret = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
        //conf.getLayer().getActivationFunction(), z));

        String confActivation = "leakyrelu";
        Object [] confExtra = {0.5};
        out = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(confActivation, myXSumY.dup(),confExtra));
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with a value via getOpFactory");
        System.out.println("======================");
        System.out.println(out);

        out = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(confActivation, myXSumY.dup()));
        System.out.println("======================");
        System.out.println("Exec and Return: Leaky Relu transformation with no default via getOpFactory");
        System.out.println("======================");
        System.out.println(out);
    }
}

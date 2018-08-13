package org.nd4j.examples;

import org.apache.commons.io.FileUtils;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;


public class Imdb{
    private static SameDiff sd;
    private double mean;
    private double std;

    public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new ClassPathResource(filepath).getFile();
        }
        sd = TFGraphMapper.getInstance().importGraph(file);
        if (sd == null){
            throw new Exception("Error loading model : " + file);
        }
    }

    private static void loadStats(){
        File file = new ClassPathResource("Boston/stats.txt").getFile();
        String contents = FileUtils.readFileToString(file);
        String stats[] = contents.split(",");
        mean = stats[0];
        std = stats[1];
    }

    public static INDArray getSampleData(){

        /*
        * The dataset contains 13 different features:

                Per capita crime rate.
                The proportion of residential land zoned for lots over 25,000 square feet.
                The proportion of non-retail business acres per town.
                Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
                Nitric oxides concentration (parts per 10 million).
                The average number of rooms per dwelling.
                The proportion of owner-occupied units built before 1940.
                Weighted distances to five Boston employment centers.
                Index of accessibility to radial highways.
                Full-value property-tax rate per $10,000.
                Pupil-teacher ratio by town.
                1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
                Percentage lower status of the population.

        * */

        double sampleData[] = new double[]{7.8750e-02, 4.5000e+01, 3.4400e+00, 0.0000e+00, 4.3700e-01, 6.7820e+00
        4.1100e+01, 3.7886e+00, 5.0000e+00, 3.9800e+02, 1.5200e+01, 3.9387e+02, 6.6800e+00};

        INDArray arr = Nd4j.create(arr).reshape(13);

        // Normalize
        arr.subi(mean);
        arr.divi(std);
        return arr;
    }


    public static double predict(INDArray arr){
        arr = Nd4j.expandDims(arr, 0);  // add batch dimension
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray outArr = sd.execAndEndResult();
        double pred = outArr.getDouble(0);
        return pred;
    }

    public static void main(String[] args) throws Exception{
        loadModel("Boston/boston.pb");
        loadStats();
        INDArray sampleData = getSampleData();
        double prediction = predict(sampleData); // in $1000
        System.out.println(Strings.format("Predicted price = $%d".format(prediction * 1000)));
    }
}

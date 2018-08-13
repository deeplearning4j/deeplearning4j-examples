package org.nd4j.examples;

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
import java.util.Arrays;

/**
 * Shows tensorflow import using mnist.
 * For the trained graph, please look at the python files under
 * Mnist/
 *
 * mnist_tf.py was used to train the graph
 *
 * @author Fariz Rahman
 */
public class Mnist {
    private static SameDiff sd;

    public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new ClassPathResource(filepath).getFile();
        }

        sd = TFGraphMapper.getInstance().importGraph(file);

        if (sd == null) {
            throw new Exception("Error loading model : " + file);
        }
    }

    public static INDArray predict(INDArray arr){
        INDArray batchedArr = Nd4j.expandDims(arr, 0);
        sd.associateArrayWithVariable(batchedArr, sd.variables().get(0));
        INDArray out = sd.execAndEndResult();
        return Nd4j.squeeze(out, 0);
    }

    public static INDArray predictBatch(INDArray arr){
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        return sd.execAndEndResult();
    }

    public static INDArray predict (String filepath) throws IOException{
        File file = new File(filepath);
        if (!file.exists()){
            file = new ClassPathResource(filepath).getFile();
        }

        BufferedImage img = ImageIO.read(file);
        double data[] = new double[28 * 28];
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                Color color = new Color(img.getRGB(i, j));
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                double greyScale = (r + g + b) / 3;
                greyScale /= 255.;
                data[i * 28 + j] = greyScale;
            }
        }

        INDArray arr = Nd4j.create(data).reshape(1, 28, 28);
        arr = Nd4j.pile(arr, arr);
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray output = sd.execAndEndResult().get(NDArrayIndex.point(0));
        System.out.println(Arrays.toString(output.reshape(10).toDoubleVector()));
        return output;

    }

    public static int predictionToLabel(INDArray prediction){
        return Nd4j.argMax(prediction.reshape(10)).getInt(0);
    }


    public static void main(String[] args) throws Exception{
        loadModel("Mnist/mnist.pb");
        for(int i = 1; i < 11; i++){
            String file = "Mnist/images/img_%d.jpg";
            file = String.format(file, i);
            INDArray prediction = predict(file);
            int label = predictionToLabel(prediction);
            System.out.println(file + "  ===>  " + label);
        }

    }
}

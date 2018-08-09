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
import java.nio.file.Files;
import java.util.Arrays;


public class Imdb{
    private static SameDiff sd;
    private static const int maxlen = 256;
    private static HashMap<String, int> wordIndex;


    private static void loadWordIndex(){
        wordIndex = new HashMap<>();
        File file = new ClassPathResource("Imdb/word_index.txt").getFile();
        String content = FileUtils.readFileToString(file);
        String[] lines = content.split('\\n');
        for(int i=0; i < lines.length - 1; i++){
            String line = lines[i];
            String[] kv = line.split(',');
            String k = kv[0];
            int v = Integer.parseInt(kv[1]);
            wordIndex.put(k, v);

        }
    }

    private static INDArray encodeText(String text){
        String words = text.split(' ');
        double arr[] = new double[maxlen];
        int pads = 256 - words.length;
        for(int i=0; i<pads; i++){
            arr[i] = (double)wordIndex.get("<PAD>");
        }
        for(int i=0; i<words.length; i++){
            if wordIndex.containsKey(words[i]){
                arr[pads + i] = (double)wordIndex.get(words[i]);
            }
            else{
                arr[pads + i] = (double)wordIndex.get("<UNK>");
            }
        }
        INDArray indArr = Nd4j.create(arr).reshape(256);
        return indArr;
    }

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

    public static double predict(INDArray arr){
        arr = Nd4j.expandDims(arr, 0);  // add batch dimension
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray outArr = sd.execAndEndResult();
        double pred = outArr.getDouble(0);
        return pred;
    }

    public static void main(String[] args) throws Exception{
        loadModel("Imdb/imdb.pb");
        loadWordIndex();
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        while(true){
            System.out.println('Enter review : ');
            String review = reader.readLine();
            INDArray arr = encodeText(review);
            double prediction = predict(arr);
            System.out.println(Strings.format("Sentiment prediction : %d", prediction));
        }

    }
}

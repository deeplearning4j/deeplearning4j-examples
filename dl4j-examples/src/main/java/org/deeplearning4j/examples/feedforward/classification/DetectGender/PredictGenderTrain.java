package org.deeplearning4j.examples.feedforward.classification.DetectGender;

/**
 * Created by KIT Solutions on 9/28/2016.
 */

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
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.data.general.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This program does following tasks
 *  - loads male/female name dataset in List
 *  - process data to convert it to binary format
 *  - trains network and save it to PredictGender.net
 *
 *  - Notes:
 *      - Data files are stored at following location
 *      .\dl4j-0.4-examples-master\dl4j-examples\src\main\resources\PredictGender folder
 */

public class PredictGenderTrain
{
    // File contains indian female names
    public File femaleNamesFile;

    // File contains indian male names
    public File maleNamesFile;

    // List to hold female names from file object "femaleNamesFile"
    public List<String> femaleNames;

    // List to hold male names from file object "maleNamesFile"
    public List<String> maleNames;

    // length of longest name, to decide number of input neurons
    public int maxLengthName;


    // String contains possible alphabets from all male/female names
    public String possibleCharacters;

    // number of input neurons decided dynamically based on input data
    public int numInputNeurons;

    // total male/female names
    public int totalRecords;

    public String filePath;


    /**
     *  This is the main function, which
     *  - creates object of PredictGenderTrain class
     *  - initializes some variables
     *  - loads male and female names from files into Lists
     *  - finds possible characters from all names during loading names into above mentioned Lists
     *  - sorts possible characters from space,a to z
     *  - converts alphabets to binary string
     *  - trains network and stores it into a file for later use.
     */
    public static void main(String args[])
    {
        PredictGenderTrain dg = new PredictGenderTrain();
        dg.filePath =  System.getProperty("user.dir") + "\\src\\main\\resources\\PredictGender\\";

        dg.maxLengthName = 0;
        dg.totalRecords = 0;

        dg.loadFemaleNameList();
        dg.loadMaleNameList();

        dg.possibleCharacters = dg.loadPossibleCharacters();

        char[] chars = dg.possibleCharacters.toCharArray();
        Arrays.sort(chars);
        dg.possibleCharacters = new String(chars);

        dg.numInputNeurons = dg.maxLengthName;

        dg.convertToBinary();
        dg.trainNetworkBinary();
    }

    /**
     * This function reads Indian-Female-Names.csv file. This file contains name and 'F' seperated by comma
     * File is read line by line and each name is stored in a List named femaleNames
     * It also keeps track of variable 'maxLengthName' to store length of largest name in this variable to be
     * used later while deciding number of input neurons dynamically.
     */
    public void loadFemaleNameList()
    {
        try
        {
            femaleNames = new ArrayList<String>();
            femaleNamesFile = new File(this.filePath + "Indian-Female-Names.csv");
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(femaleNamesFile.getAbsolutePath())));
            String line;
            while((line = br.readLine()) != null)
            {
                femaleNames.add(line);
                int currNameLength = line.split(",")[0].length();
                if (currNameLength > this.maxLengthName)
                    this.maxLengthName = currNameLength;
            }
            br.close();
            //System.out.println("Total female Names : " + femaleNames.size());
        }
        catch(Exception e)
        {
            System.out.println("Exception while reading female names file : " + e.getMessage());
        }
    }

    /**
     * This function reads Indian-Male-Names.csv file. This file contains name and 'F' seperated by comma
     * File is read line by line and each name is stored in a List named maleNames
     * It also keeps track of variable 'maxLengthName' to store length of largest name in this variable to be
     * used later while deciding number of input neurons dynamically.
     */
    public void loadMaleNameList()
    {
        try
        {
            maleNames = new ArrayList<String>();
            maleNamesFile = new File(this.filePath + "Indian-Male-Names.csv");
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(maleNamesFile.getAbsolutePath())));
            String line;
            while((line = br.readLine()) != null)
            {
                maleNames.add(line);
                int currNameLength = line.split(",")[0].length();
                if (currNameLength > this.maxLengthName)
                    this.maxLengthName = currNameLength;
            }
            br.close();
            //System.out.println("Total male Names : " + maleNames.size());
        }
        catch(Exception e)
        {
            System.out.println("Exception while reading male names file : " + e.getMessage());
        }
    }

    /**
     * This function finds all possible alphabets from all names. Later, it will be used to allocate a position to each alphabet
     * This will help while converting position of any alphabet in name to binary
     * e.g. Suppose all possible alphabets are ' abcdefghijklmnopqrstuvwxyz' (first letter is space),
     *      and name is 'Adam', then position of A is 1, d is 5,a is 1 and m is 13 and so on.
     * So, these possible characters string will be used to find a number which we can convert to 5-digit binary string
     * Complete conversion is explained in convertToBinary function below.
     */
    public String loadPossibleCharacters()
    {
        String possibleCharacters = "";
        for(int i=0;i<this.femaleNames.size();i++)
        {
            String oneName = femaleNames.get(i).split(",")[0];
            for(int j=0;j<oneName.length();j++)
            {
                if (possibleCharacters.indexOf(oneName.charAt(j)) < 0)
                    possibleCharacters = possibleCharacters + oneName.charAt(j);

            }
        }
        for(int i=0;i<this.maleNames.size();i++)
        {
            String oneName = maleNames.get(i).split(",")[0];
            for(int j=0;j<oneName.length();j++)
            {
                if (possibleCharacters.indexOf(oneName.charAt(j)) < 0)
                    possibleCharacters = possibleCharacters + oneName.charAt(j);

            }
        }
        return possibleCharacters;
    }

    /**
     * This function prepares DataSetIterator from binary files created from convertTobinary function and trains network and stores it into file to be used later.
     */
    public void trainNetworkBinary()
    {
        try
        {
            int seed = 123456;
            double learningRate = 0.01;
            int batchSize = 100;
            int nEpochs = 100;

            int numInputs = this.maxLengthName * 5;  // multiplied by 5 as for each letter we use five binary digits like 00000
            int numOutputs = 2;
            int numHiddenNodes = 2 * numInputs + numOutputs;

            //Load the training data:
            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File(this.filePath + "BinaryNameFile.csv")));
            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2);
            //Load the test/evaluation data:
            RecordReader rrTest = new CSVRecordReader();
            rrTest.initialize(new FileSplit(new File(this.filePath + "BinaryNameFile.csv")));
            DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, numInputs, 2);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .biasInit(1)
                .regularization(true).l2(1e-4)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .weightInit(WeightInit.XAVIER)
                    .activation("relu")
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .weightInit(WeightInit.XAVIER)
                    .activation("softmax")
                    .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new HistogramIterationListener(10));


            for ( int n = 0; n < nEpochs; n++)
            {
                while(trainIter.hasNext())
                {
                    model.fit(trainIter.next());
                }
                trainIter.reset();
            }

            ModelSerializer.writeModel(model,this.filePath + "PredictGender.net",true);

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(numOutputs);
            while(testIter.hasNext()){
                DataSet t = testIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray predicted = model.output(features,false);

                eval.eval(lables, predicted);

            }

            //Print the evaluation statistics
            System.out.println(eval.stats());

        }
        catch(Exception e)
        {
            System.out.println("exception while training network : " + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * This function does following job
     * - First it finds which category has less number of names, i.e. male or female
     * - So, that it can iterate upto that number (minsize).
     * - After that, it iterate through minsize, take one name from female and male alternately and convert it to binary string for each alphabet
     * e.g. possible characters ' abcdefghijklmnopqrstuvwxyz'
     * First Name : 'shivani'
     * Positions :
     * - s - 19 - binary - 10011
     *   h - 08 - binary - 01000
     *   i - 09 - binary - 01001
     *   v - 22 - binary - 10110
     *   a - 01 - binary - 00001
     *   n - 14 - binary - 01110
     *   i - 09 - binary - 01001
     *
     *   So, binary string for 'shivani' would be - 1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1
     *   Also, length of largest name is 47 (for current dataset of males and females only), so we are left with remaining 45 - 7 = 38
     *   Hence, we need to append 38 x 5 = 190 remaining digits with all 0's , so final string will be
     *   'shivani' = 1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     *               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     *               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     *               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     *               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
     *
     *   Similarly, all names would be converted to this type of binary strings
     *   Last digit would be added in this string as 0 (for female) or 1 (for male) as labels
     *   If number of names in male or female are unequal, then pending names would also be converted to binary and added to the file.
     *   This function writes BinaryNameFile.csv containing above mentioned data.
     */
    public void convertToBinary()
    {
        try
        {
            PrintWriter writer = new PrintWriter(this.filePath + "BinaryNameFile.csv");
            int minSize = 0;
            boolean femalemore = false;
            if (femaleNames.size() >= maleNames.size())
            {
                minSize = maleNames.size();
                femalemore = true;
            }
            else
            {
                minSize = femaleNames.size();
                femalemore = false;
            }

            for(int m = 0;m<minSize;m++)
            {
                String oneNumericLine = "";
                String oneName = femaleNames.get(m).split(",")[0];
                oneNumericLine = oneNumericLine + this.getBinaryString(oneName);
                oneNumericLine = oneNumericLine + "0";
                writer.println(oneNumericLine);
                this.totalRecords++;

                oneNumericLine = "";
                oneName = maleNames.get(m).split(",")[0];
                oneNumericLine = oneNumericLine + this.getBinaryString(oneName);
                oneNumericLine = oneNumericLine + "1";
                writer.println(oneNumericLine);
                this.totalRecords++;
            }

/*            if(femalemore == true)
            {
                for(int n = minSize;n<femaleNames.size();n++)
                {
                    String oneNumericLine = "";
                    String oneName = femaleNames.get(n).split(",")[0];
                    oneNumericLine = oneNumericLine + this.getBinaryString(oneName);
                    oneNumericLine = oneNumericLine + "0";
                    writer.println(oneNumericLine);
                    this.totalRecords++;
                }
            }
            else
            {
                for(int n=minSize;n<maleNames.size();n++)
                {
                    String oneNumericLine = "";
                    String oneName = maleNames.get(n).split(",")[0];
                    oneNumericLine = oneNumericLine + this.getBinaryString(oneName);
                    oneNumericLine = oneNumericLine + "1";
                    writer.println(oneNumericLine);
                    this.totalRecords++;
                }
            }
*/
            writer.close();
        }
        catch(Exception e)
        {
            System.out.println("file writing error :" + e.getMessage());
        }
    }

    /**
     *
     * @param name - name of the person is passed for binary conversion
     * @return - returns the binary converted string to be stored in file.
     */
    public String getBinaryString(String name)
    {
        String binaryString ="";
        for (int j = 0; j < name.length(); j++)
        {
            String fs = pad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))),5);
            binaryString = binaryString + fs;
        }
        int diff = 0;
        if(name.length() < this.maxLengthName )
        {
            diff = this.maxLengthName - name.length();
            for(int i=0;i<diff;i++)
            {
                binaryString = binaryString + "00000";
            }
        }

        String tempStr = "";
        for(int i=0;i<binaryString.length();i++)
        {
            tempStr = tempStr + binaryString.charAt(i) + ",";
        }
        return tempStr;
    }

    /**
     *
     * @param string - binary converted string
     * @param total_length - total number of characters after padding
     * @return - returns a string after left-padding 0 to the string passed as first parameter
     */
    public String pad(String string,int total_length)
    {
        String str = string;
        int diff = 0;
        if(total_length > string.length())
            diff = total_length - string.length();
        for(int i=0;i<diff;i++)
            str = "0" + str;
        return str;
    }
}

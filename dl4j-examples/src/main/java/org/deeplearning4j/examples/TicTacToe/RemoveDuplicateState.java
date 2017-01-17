package org.deeplearning4j.examples.TicTacToe;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com) on 04-Aug-16.
 * Removes the duplicate move and update the probability.
 */
public class RemoveDuplicateState {

    public List<INDArray> finalInputArray = new ArrayList<INDArray>();
    public List<INDArray> finalProbabilityArray = new ArrayList<INDArray>();

    public static void main(String[] args) throws Exception {

        /*
        Read all states from file and remove the Duplicate State.
        */
        String filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";
        String inputFileDataSet = filePath + "AllMoveWithReward.txt";

        RemoveDuplicateState processDataObject = new RemoveDuplicateState();

        try (BufferedReader br = new BufferedReader(new FileReader(inputFileDataSet))) {

            String line = "";
            while ((line = br.readLine()) != null) {

                INDArray input = Nd4j.zeros(1, 9);
                INDArray label = Nd4j.zeros(1, 1);

                String[] nextLines = line.split(" ");

                String tempLine1 = "";
                String tempLine2 = "";

                tempLine1 = nextLines[0];
                tempLine2 = nextLines[1];

                String testLines[] = tempLine1.split(":");

                for (int i = 0; i < 9; i++) {
                    int number = Integer.parseInt(testLines[i]);
                    input.putScalar(new int[]{0, i}, number);
                }

                double doubleNumber = Double.parseDouble(tempLine2);
                label.putScalar(new int[]{0, 0}, doubleNumber);

                processDataObject.processData(input, label);
            }

            String saveFilePath = filePath + "DuplicateRemoved.txt";
            processDataObject.saveProcessData(saveFilePath);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    /**
     * Using this method,Remove the duplicate state and update the maximum reward(probability).
     */
    public void processData(INDArray inputLabelArray, INDArray outputLabel) {

        int indexPosition = finalInputArray.indexOf(inputLabelArray);

        if (indexPosition != -1) {
            INDArray outputArray = finalProbabilityArray.get(indexPosition);
            INDArray newUpdatedArray = this.getNewArray(outputArray, outputLabel);
            finalProbabilityArray.set(indexPosition, newUpdatedArray);
        } else {
            finalInputArray.add(inputLabelArray);
            finalProbabilityArray.add(outputLabel);
        }
    }

    /**
     * Using this method,store the all unique state of game with its reward.
     */
    public void saveProcessData(String saveFilePath) {

        try (FileWriter writer = new FileWriter(saveFilePath)) {


            for (int index = 0; index < finalInputArray.size(); index++) {

                INDArray arrayFromInputList = finalInputArray.get(index);
                INDArray arrayFromLabelList = finalProbabilityArray.get(index);

                String tempString1 = "";
                int sizeOfInput = arrayFromInputList.length();

                for (int i = 0; i < sizeOfInput; i++) {

                    int number = (int) arrayFromInputList.getDouble(i);

                    tempString1 = tempString1 + String.valueOf(number).trim();
                    if (i != (sizeOfInput - 1)) {
                        tempString1 += ":";
                    }
                }
                String tempString2 = String.valueOf(arrayFromLabelList.getDouble(0));
                String output = tempString1 + " " + tempString2;

                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        } catch (Exception i) {
            System.out.println(i.toString());
        }
    }

    /**
     * Compare two INDArray(s)  and return INDArray with maximum value.
     */
    public INDArray getNewArray(INDArray array1, INDArray array2) {
        INDArray newReturnArray = Nd4j.zeros(1, 1);

        for (int i = 0; i < array1.length(); i++) {

            double a = array1.getDouble(i);
            double b = array2.getDouble(i);

            double max = 0;

            if (a > b) {
                max = a;
            } else {
                max = b;
            }
            newReturnArray.putScalar(new int[]{0, i}, max);
        }
        return newReturnArray;
    }

}

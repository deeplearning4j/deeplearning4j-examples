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
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com) on 08-Aug-16.
 * Please update file path based on your dir.
 * Create a DataSet from move ,create a move with probability  and input sequences
 */
public class RewardGameState {

    FileWriter writer;
    List<INDArray> middleList = new ArrayList<>();

    RewardGameState() {
        try {
            String filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";
            writer = new FileWriter(filePath + "AllMoveWithReward.txt");
        } catch (Exception i) {
            System.out.println(i.toString());
        }
    }
    /*
    * In this method,pass the game sequence  and generate intermediate game state
    * For Odd
    * Game sequence like:-1.00, 2.00, 4.00, 3.00, 0.00, 0.00, 5.00, 0.00, 0.00 (i.e Max number is Odd)from this method will create game state as example
    *
    *  First State
    *   1,0,0,
    *   0,0,0,
    *   0,0,0
    *  Second State -- Intermediate State
    *   1,2,0,
    *   0,0,0,
    *   0,0,0,
    *  Third State
    *   1,2,0,
    *   1,0,0,
    *   0,0,0,
    *  Fourth State -- Intermediate State
    *   1,2,2,
    *   1,0,0,
    *   0,0,0,
    *  Fifth State -- In this State,First player win the game
    *   1,2,2,
    *   1,0,0,
    *   1,0,0,
    *
    *  For Even
    *  Game sequence like:-1.00, 2.00, 3.00, 5.00, 4.00, 0.00, 0.00, 6.00, 0.00(i.e Max number is Even)from this method will create game state as example
    *
    *  First State -- Intermediate State
    *   1,0,0,
    *   0,0,0,
    *   0,0,0
    *  Second State
    *   1,2,0,
    *   0,0,0,
    *   0,0,0,
    *  Third State -- Intermediate State
    *   1,2,1,
    *   0,0,0,
    *   0,0,0,
    *  Fourth State
    *   1,2,1,
    *   0,2,0,
    *   0,0,0,
    *  Fifth State -- Intermediate State
    *   1,2,1,
    *   1,2,0,
    *   0,0,0,
    *  Sixth State -- In this State,Second player win the game
    *   1,2,1,
    *   1,2,0,
    *   0,2,0,
    *
    *   All Game State Store in INDArray
    */

    public static void main(String[] args) throws Exception {

        String filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";

        RewardGameState rewardObject = new RewardGameState();

        rewardObject.processMoveFile(filePath + "OddMove.txt", 0); //Odd Position
        System.out.println("Odd Move Processed");

        rewardObject.processMoveFile(filePath + "EvenMove.txt", 1); //Even Position
        System.out.println("Even Move Processed");

        rewardObject.addExtraMove();
        System.out.println("Intermediate Move Processed"); // Intermediate Move store to the file systeam
    }

    public void generateGameStateAndRewardToIt(INDArray output, int moveType) {

        INDArray maxArray = Nd4j.max(output);
        double maxNumber = maxArray.getDouble(0);

        List<INDArray> sequenceList = new ArrayList<>();
        INDArray sequenceArray = Nd4j.zeros(1, 9);

        int move = 1;

        int positionOfDigit = 0;

        for (int i = 1; i <= maxNumber; i++) {

            INDArray newTempArray = Nd4j.zeros(1, 9);
            positionOfDigit = getPosition(output, i);

            if (i % 2 == moveType) {

                Nd4j.copy(sequenceArray, newTempArray);
                sequenceList.add(newTempArray);
            } else {
                Nd4j.copy(sequenceArray, newTempArray);
                middleList.add(newTempArray);
            }
            sequenceArray.putScalar(new int[]{0, positionOfDigit}, move);
            move = move * (-1);
        }

        move = move * (-1);
        INDArray newTempArray2 = Nd4j.zeros(1, 9);

        sequenceArray.putScalar(new int[]{0, positionOfDigit}, move);
        Nd4j.copy(sequenceArray, newTempArray2);
        sequenceList.add(newTempArray2);

        rewardToState(sequenceList);
    }

    /*
    * Here pass the Game sequences's fileName and Also pass "Game is odd player game or even player game'.
    * Using above argument,genrate the game state and also reward that state base on the win,loose and Draw
    * */
    public void processMoveFile(String fileName, int moveType) {

        try (BufferedReader br = new BufferedReader(new FileReader(fileName));) {

            String line = "";
            while ((line = br.readLine()) != null) {

                INDArray input = Nd4j.zeros(1, 9);
                String[] nextLine = line.split(",");

                for (int i = 0; i < 9; i++) {

                    double number = (Double.parseDouble(nextLine[i]));
                    input.putScalar(new int[]{0, i}, number);
                }
                generateGameStateAndRewardToIt(input, moveType); //0 odd Position  and 1 for Even Position
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }


    /*
    * Using this method ,store the intermediate state in file with probability 0.50
    *
    */

    public void addExtraMove() {

        try {

            for (int index = 0; index < middleList.size(); index++) {

                INDArray arrayFromInputList = middleList.get(index);

                String tempString1 = "";
                int sizeOfInput = arrayFromInputList.length();

                for (int i = 0; i < sizeOfInput; i++) {

                    int number = (int) arrayFromInputList.getDouble(i);
                    tempString1 = tempString1 + String.valueOf(number).trim();

                    if (i != (sizeOfInput - 1)) {
                        tempString1 += ":";
                    }
                }

                String tempString2 = "0.5";
                tempString1 = tempString1.replaceAll("-1", "2");
                String output = tempString1 + " " + tempString2;

                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        } catch (Exception Io) {
            System.out.println(Io.toString());
        }
    }

    public int getPosition(INDArray array, double number) {

        for (int i = 0; i < array.length(); i++) {

            if (array.getDouble(i) == number) {
                return i;
            }
        }
        return 0;
    }

    /* Game sequence like:-1.00, 2.00, 4.00, 3.00, 0.00, 0.00, 5.00, 0.00, 0.00 (i.e Max number is Odd)from this method will be reward as follow
    *
    *  First State (State-1)
    *   1,0,0,
    *   0,0,0,
    *   0,0,0
    *  Second State -- Intermediate State
    *   1,2,0,
    *   0,0,0,
    *   0,0,0,
    *  Third State (State-2)
    *   1,2,0,
    *   1,0,0,
    *   0,0,0,
    *  Fourth State -- Intermediate State
    *   1,2,2,
    *   1,0,0,
    *   0,0,0,
    *  Fifth State -- In this State,First player win the game (State-3)
    *   1,2,2,
    *   1,0,0,
    *   1,0,0,
    *
    * Here first player win the game.in this last State of game reward is 1
    * using this equation calculate the reward
    * probabilityOfPreviousState  = 0.5 + 0.1*(probabilityOfCurrentState-0.5);
    * so the
    * Reward for  State-3 is 1
    * Reward for  State-2 is  ( probabilityOf2ndState=0.5+0.1(1-0.5) = 0.55
    * Reward for  State-1 is  ( probabilityOf1stState=0.5+0.1(0.55-0.5) = 0.505
    * And intermediate State Reward with 0.50
    * Store the State and reward in file.
    */

    public void rewardToState(List<INDArray> arrayList) {

        double probabilityValue = 0;
        int sizeOfArray = arrayList.size();
        INDArray probabilityArray = Nd4j.zeros(sizeOfArray, 1);


        for (int p = (arrayList.size() - 1); p >= 0; p--) {

            if (p == (arrayList.size() - 1)) {
                probabilityValue = 1.0;
            } else {
                probabilityValue = 0.5 + 0.1 * (probabilityValue - 0.5);
            }
            probabilityArray.putScalar(new int[]{p, 0}, probabilityValue);
        }

        try {
            for (int index = 0; index < arrayList.size(); index++) {

                INDArray arrayFromInputList = arrayList.get(index);
                String tempString1 = "";

                int sizeOfInput = arrayFromInputList.length();

                for (int i = 0; i < sizeOfInput; i++) {

                    int number = (int) arrayFromInputList.getDouble(i);
                    tempString1 = tempString1 + String.valueOf(number).trim();

                    if (i != (sizeOfInput - 1)) {
                        tempString1 += ":";
                    }
                }

                String tempString2 = String.valueOf(probabilityArray.getDouble(index));
                tempString1 = tempString1.replaceAll("-1", "2");
                String output = tempString1 + " " + tempString2;

                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        } catch (Exception io) {
            System.out.println(io.toString());
        }
    }
}


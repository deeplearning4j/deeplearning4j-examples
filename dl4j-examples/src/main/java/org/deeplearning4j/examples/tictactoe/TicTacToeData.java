package org.deeplearning4j.examples.tictactoe;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * This program generates basic data to be used in Training Program.
 * It performs following major steps
 * - generates all possible game states
 * - reward all game states generated in above step by finding winning state, assign it to value 1 and goes back upto first step through all steps and
 * calculates probability of each step in the game to make that move win game in the last state.
 * - Writes all states data along with probability of each state to win the game which was calculated in above step.
 * Note :
 * - Used <b>http://www.se16.info/hgb/tictactoe.htm</b> link to understand all possible number of moves in Tic-Tac-Toe game.
 * - Refer ReadMe.txt for detail explanation of each step.
 * <p>
 * <b>Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com), 19-Jan-2017.</b>
 */


public class TicTacToeData {


    //All these private variables are not meant to be used from outside of the class. So, no getter/setter methods are provided.
    private List<INDArray> moveSequenceList = new ArrayList<>();
    private List<INDArray> oddPlayerWiningList = new ArrayList<>();
    private List<INDArray> evenPlayerWiningList = new ArrayList<>();

    private List<INDArray> middleList = new ArrayList<>();
    private List<INDArray> finalOutputArrayList = new ArrayList<>();
    private List<Double> finalProbabilityValueList = new ArrayList<>();

    private int previousMoveNumber = 0;

    /**
     * Main function that calls all major functions one-by-one to generate training data to be used in training program.
     */
    public static void main(String[] args) throws Exception {

        String filePath = new ClassPathResource("TicTacToe").getFile().getAbsolutePath() + File.separator + "AllMoveWithReward.txt";

        TicTacToeData data = new TicTacToeData();


        System.out.println("Data Processing Started : " + (new Date()).toString());
        data.generatePossibleGames();
        System.out.println("All possible game state sequence generated, Finished At : " + (new Date()).toString());

        data.rewardGameState();
        System.out.println("Reward calculation finished : " + (new Date()).toString());

        data.writeFinalData(filePath);
        System.out.println("File generation completed : " + (new Date()).toString());
    }

    /**
     * Initiate generating all possible game states. Refer ReadMe.txt for detailed explanation.
     */
    public void generatePossibleGames() {
        try {
            for (int index = 1; index <= 9; index++) {
                generateStateBasedOnMoveNumber(index);
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        /*Here  process odd and Draw using odd list*/
        oddPlayerWiningList.addAll(moveSequenceList);
    }

    /**
     * This function allocates reward points to each state of the game based on the winning state.
     * For all elements in oddPlayerWiningList, evenPlayerWiningList and middleList (which contains intermediate entries before winning or draw).
     * Refer ReadMe.txt for detailed explanation.
     */
    public void rewardGameState() {
        for (INDArray a : oddPlayerWiningList) {
            generateGameStatesAndRewardToIt(a, 0);//0 odd for position  and 1 for even Position
        }
        for (INDArray a : evenPlayerWiningList) {
            generateGameStatesAndRewardToIt(a, 1);
        }
        for (INDArray element : middleList) {
            addToFinalOutputList(element, 0.5);
        }
    }

    /**
     * This function called by generatePossibleGames. It is the main function that generates all possible game states.
     * Refer ReadMe.txt for detailed explanation.
     */
    private void generateStateBasedOnMoveNumber(int moveNumber) throws Exception {

        int newMoveNumber = previousMoveNumber + 1;

        if (newMoveNumber != moveNumber) {
            throw new Exception("Missing one or more moves between 1 to 9");
        } else if (moveNumber > 9 || moveNumber < 1) {
            throw new Exception("Invalid move number");
        }

        previousMoveNumber = newMoveNumber;

        List<INDArray> tempMoveSequenceList = new ArrayList<>();
        tempMoveSequenceList.addAll(moveSequenceList);
        moveSequenceList.clear();

        if (moveNumber == 1) {
            for (int i = 0; i < 9; i++) {
                INDArray temp2 = Nd4j.zeros(1, 9);
                temp2.putScalar(new int[]{0, i}, 1);
                moveSequenceList.add(temp2);
            }
        } else {
            boolean isOddMoveNumber = ((moveNumber % 2) != 0) ? true : false;
            int lengthOfTempMoveSequenceList = tempMoveSequenceList.size();

            for (int i = 0; i < lengthOfTempMoveSequenceList; i++) {
                INDArray moveArraySequence = tempMoveSequenceList.get(i);
                for (int j = 0; j < 9; j++) {
                    INDArray temp1 = Nd4j.zeros(1, 9);
                    Nd4j.copy(moveArraySequence, temp1);
                    if (moveArraySequence.getInt(j) == 0) {
                        temp1.putScalar(new int[]{0, j}, moveNumber);
                        if (moveNumber > 4) {
                            if (checkWin(temp1, isOddMoveNumber)) {
                                if (isOddMoveNumber == true) {
                                    oddPlayerWiningList.add(temp1);
                                } else {
                                    evenPlayerWiningList.add(temp1);
                                }
                            } else {
                                moveSequenceList.add(temp1);
                            }

                        } else {
                            moveSequenceList.add(temp1);
                        }
                    }
                }
            }
        }
    }


    /**
     * Identify the game state win/Draw.
     */
    private boolean checkWin(INDArray sequence, boolean isOdd) {
        double boardPosition1 = sequence.getDouble(0);
        double boardPosition2 = sequence.getDouble(1);
        double boardPosition3 = sequence.getDouble(2);
        double boardPosition4 = sequence.getDouble(3);
        double boardPosition5 = sequence.getDouble(4);
        double boardPosition6 = sequence.getDouble(5);
        double boardPosition7 = sequence.getDouble(6);
        double boardPosition8 = sequence.getDouble(7);
        double boardPosition9 = sequence.getDouble(8);

        boolean position1 = isOdd ? (sequence.getDouble(0) % 2.0 != 0) : (sequence.getDouble(0) % 2.0 == 0);
        boolean position2 = isOdd ? (sequence.getDouble(1) % 2.0 != 0) : (sequence.getDouble(1) % 2.0 == 0);
        boolean position3 = isOdd ? (sequence.getDouble(2) % 2.0 != 0) : (sequence.getDouble(2) % 2.0 == 0);
        boolean position4 = isOdd ? (sequence.getDouble(3) % 2.0 != 0) : (sequence.getDouble(3) % 2.0 == 0);
        boolean position5 = isOdd ? (sequence.getDouble(4) % 2.0 != 0) : (sequence.getDouble(4) % 2.0 == 0);
        boolean position6 = isOdd ? (sequence.getDouble(5) % 2.0 != 0) : (sequence.getDouble(5) % 2.0 == 0);
        boolean position7 = isOdd ? (sequence.getDouble(6) % 2.0 != 0) : (sequence.getDouble(6) % 2.0 == 0);
        boolean position8 = isOdd ? (sequence.getDouble(7) % 2.0 != 0) : (sequence.getDouble(7) % 2.0 == 0);
        boolean position9 = isOdd ? (sequence.getDouble(8) % 2.0 != 0) : (sequence.getDouble(8) % 2.0 == 0);

        if (((position1 && position2 && position3) && (boardPosition1 != 0 && boardPosition2 != 0 && boardPosition3 != 0)) ||
            ((position4 && position5 && position6) && (boardPosition4 != 0 && boardPosition5 != 0 && boardPosition6 != 0)) ||
            ((position7 && position8 && position9) && (boardPosition7 != 0 && boardPosition8 != 0 && boardPosition9 != 0)) ||
            ((position1 && position4 && position7) && (boardPosition1 != 0 && boardPosition4 != 0 && boardPosition7 != 0)) ||
            ((position2 && position5 && position8) && (boardPosition2 != 0 && boardPosition5 != 0 && boardPosition8 != 0)) ||
            ((position3 && position6 && position9) && (boardPosition3 != 0 && boardPosition6 != 0 && boardPosition9 != 0)) ||
            ((position1 && position5 && position9) && (boardPosition1 != 0 && boardPosition5 != 0 && boardPosition9 != 0)) ||
            ((position3 && position5 && position7) && (boardPosition3 != 0 && boardPosition5 != 0 && boardPosition7 != 0))) {

            return true;
        } else {
            return false;
        }
    }

    /**
     * This function generate all intermediate (including winning) game state from the winning state available oddPlayerWiningList or evenPlayerWiningList
     * and pass it to calculateReward function to calculate probability of all states of winning game.
     * Refer ReadMe.txt for detailed explanation.
     */
    private void generateGameStatesAndRewardToIt(INDArray output, int moveType) {

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
        calculateReward(sequenceList);
    }

    /**
     * This function gives cell number of a particular move
     */
    private int getPosition(INDArray array, double number) {

        for (int i = 0; i < array.length(); i++) {
            if (array.getDouble(i) == number) {
                return i;
            }
        }
        return 0;
    }


    /**
     * Function to calculate Temporal Difference. Refer ReadMe.txt for detailed explanation.
     */
    private void calculateReward(List<INDArray> arrayList) {

        double probabilityValue = 0;
        for (int p = (arrayList.size() - 1); p >= 0; p--) {
            if (p == (arrayList.size() - 1)) {
                probabilityValue = 1.0;
            } else {
                probabilityValue = 0.5 + 0.1 * (probabilityValue - 0.5);
            }
            INDArray stateAsINDArray = arrayList.get(p);
            addToFinalOutputList(stateAsINDArray, probabilityValue);
        }
    }

    /**
     * This function adds game states to final list after calculating reward for each state of a winning game.
     */
    private void addToFinalOutputList(INDArray inputLabelArray, double inputRewardValue) {
        int indexPosition = finalOutputArrayList.indexOf(inputLabelArray);

        if (indexPosition != -1) {
            double rewardValue = finalProbabilityValueList.get(indexPosition);
            double newUpdatedRewardValue = (rewardValue > inputRewardValue) ? rewardValue : inputRewardValue;
            finalProbabilityValueList.set(indexPosition, newUpdatedRewardValue);
        } else {
            finalOutputArrayList.add(inputLabelArray);
            finalProbabilityValueList.add(inputRewardValue);
        }
    }

    /**
     * This function writes all states of all games into file along with their probability values.
     */
    public void writeFinalData(String saveFilePath) {

        try (FileWriter writer = new FileWriter(saveFilePath)) {

            List<String> finalStringListForFile = new ArrayList<>();
            for (int index = 0; index < finalOutputArrayList.size(); index++) {
                INDArray arrayFromInputList = finalOutputArrayList.get(index);
                double rewardValue = finalProbabilityValueList.get(index);

                String tempString = arrayFromInputList.toString().replace('[', ' ').replace(']', ' ').replace(',', ':').replaceAll("\\s", "");
                String tempString2 = tempString;
                tempString = tempString.replaceAll("-1", "2");
                String output = tempString + " " + String.valueOf(rewardValue);

                int indexInList1 = finalStringListForFile.indexOf(output);
                if (indexInList1 == -1) {
                    finalStringListForFile.add(output);
                }
                tempString2 = tempString2.replaceAll("1", "2").replaceAll("-2", "1");
                String output2 = tempString2 + " " + String.valueOf(rewardValue);
                int indexInList2 = finalStringListForFile.indexOf(output2);

                if (indexInList2 == -1) {
                    finalStringListForFile.add(output2);
                }
            }
            for (String s : finalStringListForFile) {
                writer.append(s);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }

        } catch (Exception i) {
            System.out.println(i.toString());
        }
    }
}

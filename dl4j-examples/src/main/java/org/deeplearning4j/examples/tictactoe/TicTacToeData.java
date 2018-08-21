package org.deeplearning4j.examples.tictactoe;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

	private static Log log = LogFactory.getLog(TicTacToeData.class);

    /**
     * Main function that calls all major functions one-by-one to generate training data to be used in training program.
     */
    public static void main(String[] args) throws Exception {
    	long start = System.nanoTime();
    	try {
            TicTacToeData data = new TicTacToeData();
            log.info("Data Processing Started");
            final String allMoves = data.generatePossibleGames();
            log.info("All possible game state sequence generated, Finished");

            final Path dataFile = Paths.get(System.getProperty("user.home") + "/AllMoveWithReward.txt");
            Files.deleteIfExists(dataFile);
            final Path dataFilePath = Files.createFile(dataFile);
            try (BufferedWriter writer = Files.newBufferedWriter(dataFilePath)) {
                writer.write(allMoves);
            }
        } catch (Exception e) {
            log.error(e);
        }
        log.info("Total time = " + (System.nanoTime() - start)/1_000_000);
    }

    /**
     * Initiate generating all possible game states. Refer ReadMe.txt for detailed explanation.
     */
    private String generatePossibleGames() throws Exception {
        List<String> values = new ArrayList<>();
        List<INDArray> moveSequenceList = new ArrayList<>();
        for (int index = 1; index <= 9; index++) {
            generateStateBasedOnMoveNumber(index, moveSequenceList, values);
        }
        return values.stream().distinct().collect(Collectors.joining("\r\n"));
    }

    /**
     * This function called by generatePossibleGames. It is the main function that generates all possible game states.
     * Refer ReadMe.txt for detailed explanation.
     */
    private void generateStateBasedOnMoveNumber(int moveNumber, List<INDArray> moveSequenceList, List<String> values) throws Exception {

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
            int lengthOfTempMoveSequenceList = tempMoveSequenceList.size();

            for (INDArray moveArraySequence : tempMoveSequenceList)
                IntStream.range(0, 9).filter(j -> moveArraySequence.getInt(j) == 0).forEach(j -> {
                    INDArray temp1 = Nd4j.zeros(1, 9);
                    Nd4j.copy(moveArraySequence, temp1);
                    temp1.putScalar(new int[]{0, j}, moveNumber);
                    if (moveNumber > 4) {
                        boolean isOddMoveNumber = (moveNumber % 2) != 0;
                        if (checkWin(temp1, isOddMoveNumber)) {
                            values.addAll(generateGameStatesAndRewardToIt(temp1, isOddMoveNumber ? 0 : 1));
                        } else {
                            moveSequenceList.add(temp1);
                        }

                    } else {
                        moveSequenceList.add(temp1);
                    }
                });
        }
        if (moveNumber == 9) {
            values.addAll(moveSequenceList.stream()
                            .flatMap(temp1 -> generateGameStatesAndRewardToIt(temp1, 0).stream())
                            .collect(Collectors.toList()));
        }
    }


    /**
     * Identify the game state win/Draw.
     */
    private boolean checkWin(INDArray sequence, boolean isOdd) {
        double boardPosition1 = sequence.getDouble(0);
        boolean boardIsOdd = boardPosition1 % 2.0 != 0;
        double boardPosition2 = sequence.getDouble(1);
        double boardPosition3 = sequence.getDouble(2);
        double boardPosition4 = sequence.getDouble(3);
        double boardPosition5 = sequence.getDouble(4);
        double boardPosition6 = sequence.getDouble(5);
        double boardPosition7 = sequence.getDouble(6);
        double boardPosition8 = sequence.getDouble(7);
        double boardPosition9 = sequence.getDouble(8);

        boolean position1 = isOdd && boardIsOdd;
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
    private List<String> generateGameStatesAndRewardToIt(INDArray output, int moveType) {
    	Map<INDArray, Double> valueMap = new HashMap<>();
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
                valueMap.put(newTempArray, 0.5);
            }
            sequenceArray.putScalar(new int[]{0, positionOfDigit}, move);
            move = move * (-1);
        }
        move = move * (-1);
        INDArray newTempArray2 = Nd4j.zeros(1, 9);

        sequenceArray.putScalar(new int[]{0, positionOfDigit}, move);
        Nd4j.copy(sequenceArray, newTempArray2);
        sequenceList.add(newTempArray2);
        calculateReward(sequenceList, valueMap);
        return valueMap.entrySet()
                .parallelStream()
                .map(entry -> generateStringList(entry.getKey(), entry.getValue()))
                .distinct()
                .collect(Collectors.toList());
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
    private void calculateReward(List<INDArray> arrayList, Map<INDArray, Double> valueMap) {

        double probabilityValue = 0;
        for (int p = (arrayList.size() - 1); p >= 0; p--) {
            if (p == (arrayList.size() - 1)) {
                probabilityValue = 1.0;
            } else {
                probabilityValue = 0.5 + 0.1 * (probabilityValue - 0.5);
            }
            INDArray stateAsINDArray = arrayList.get(p);
            valueMap.merge(stateAsINDArray, probabilityValue, (oldValue, newValue) -> oldValue > newValue ? oldValue : newValue);
        }
    }

	private String generateStringList(INDArray arrayFromInputList,
			double rewardValue) {
		List<String> strings = new ArrayList<>();
		StringBuilder stringBuilder = new StringBuilder();
		String tempString = arrayFromInputList.toString().replace('[', ' ').replace(']', ' ').replace(',', ':').replaceAll("\\s", "");
		stringBuilder.append(tempString.replaceAll("-1", "2"));
        stringBuilder.append(" ");
        stringBuilder.append(rewardValue);
        stringBuilder.append("\r\n");
        stringBuilder.append(tempString.replaceAll("1", "2").replaceAll("-2", "1"));
        stringBuilder.append(" ");
        stringBuilder.append(rewardValue);
		return stringBuilder.toString();
	}
}

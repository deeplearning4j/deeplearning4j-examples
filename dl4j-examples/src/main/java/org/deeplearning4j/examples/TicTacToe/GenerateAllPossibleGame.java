package org.deeplearning4j.examples.TicTacToe;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt. Ltd.( www.kitsol.com) on 05-Aug-2016.
 * This program is used for generating possible number of moves for tic-tac-toe game.
 * Here,Odd Number sequence consider as first player's move and Even number sequence consider  as second player's move
 */
public class GenerateAllPossibleGame {
    public static void main(String[] args) throws Exception {
        String filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";

        /*
        TicTacToe board is Empty, i.e First player with first move has 9 possible move.
        */
        List<INDArray> firstMovesSequence = new ArrayList<INDArray>();
        for (int i = 0; i < 9; i++) {
            INDArray temp2 = Nd4j.zeros(1, 9);
            temp2.putScalar(new int[]{0, i}, 1);
            firstMovesSequence.add(temp2);
        }

        /*
        For Second player with second move, 8 possible positions remain, so there will be next (9) * 8 = 72 possible states of game.
        */
        List<INDArray> secondMovesSequence = new ArrayList<INDArray>();
        for (int i = 0; i < 9; i++) {
            INDArray firstMoveArraySeq = firstMovesSequence.get(i);
            for (int j = 0; j < 9; j++) {
                INDArray temp1 = Nd4j.zeros(1, 9);
                temp1.putScalar(new int[]{0, i}, firstMoveArraySeq.getInt(i));
                if (firstMoveArraySeq.getInt(j) != 1) {
                    temp1.putScalar(new int[]{0, j}, 2);
                    secondMovesSequence.add(temp1);
                }
            }
        }

        /*
        For First player with third move, 7 possible positions remain, so there will be next (72) * 7 = 504 possible states of game.
        */
        List<INDArray> thirdMovesSequence = new ArrayList<INDArray>();
        for (int i = 0; i < 72; i++) {
            INDArray secondMoveArraySeq = secondMovesSequence.get(i);
            for (int j = 0; j < 9; j++) {
                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(secondMoveArraySeq, temp1);
                if (secondMoveArraySeq.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 3);
                    thirdMovesSequence.add(temp1);
                }
            }
        }

        /*
        For Second player with fourth move, 6 possible positions remain, so there will be next (504) * 6 = 3024 possible states of game.
        */
        List<INDArray> fourthMovesSequence = new ArrayList<INDArray>();
        for (int i = 0; i < 504; i++) {
            INDArray thirdMoveArraySequence = thirdMovesSequence.get(i);
            for (int j = 0; j < 9; j++) {
                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(thirdMoveArraySequence, temp1);
                if (thirdMoveArraySequence.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 4);
                    fourthMovesSequence.add(temp1);
                }
            }
        }
        List<INDArray> fifthMovesSequence = new ArrayList<INDArray>();
        List<INDArray> fifthMovesWins = new ArrayList<INDArray>();

        /*
        For First player with fifth move, 5 possible positions remain, so there will be next (3024) * 5 = 15120 possible states of game.
        */
        for (int i = 0; i < 3024; i++) {
            INDArray fourthMoveArraySequence = fourthMovesSequence.get(i);
            for (int j = 0; j < 9; j++) {
                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(fourthMoveArraySequence, temp1);
                if (fourthMoveArraySequence.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 5);
                    if (CheckWins(temp1, true)) {
                        fifthMovesWins.add(temp1);
                    } else {
                        fifthMovesSequence.add(temp1);
                    }
                }
            }
        }

        System.out.println("Total Win In 5th Move : " + fifthMovesWins.size());
        WriteFile(filePath + "FifthWiningData.txt", fifthMovesWins);

        // Clear lists for 1 to 4th moves
        firstMovesSequence.clear();
        secondMovesSequence.clear();
        thirdMovesSequence.clear();
        fourthMovesSequence.clear();
        fifthMovesWins.clear();

        /*
        When second player is at the sixth move, possible states of the games are 13680. Below is the description
		In fifth move of fisrt player, possible winning moves are 1440.
		Now, possible positions from previous state are 15120. We need to subtract 1440 from this number will give us 13680 (15120 - 1440)
		We need to multiply this number (13680) with empty cells , i.e 4 (in the sixth move), which will give us total 54720 possible moves
        */
        List<INDArray> sixMovesSequence = new ArrayList<INDArray>();
        List<INDArray> sixthMovesWins = new ArrayList<INDArray>();

        for (int i = 0; i < fifthMovesSequence.size(); i++) {

            INDArray sixthMoveArraySequence = fifthMovesSequence.get(i);

            for (int j = 0; j < 9; j++) {
                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(sixthMoveArraySequence, temp1);
                if (sixthMoveArraySequence.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 6);

                    if (CheckWins(temp1, false)) {
                        sixthMovesWins.add(temp1);
                    } else {
                        sixMovesSequence.add(temp1);
                    }
                }
            }
        }

        System.out.println("Total Win In 6th Move : " + sixthMovesWins.size());
        WriteFile(filePath + "SixthWiningData.txt", sixthMovesWins);

        // we can clear fifth move data here
        fifthMovesSequence.clear();

        List<INDArray> seventhMovesSequence = new ArrayList<INDArray>();
        List<INDArray> seventhMoveWins = new ArrayList<INDArray>();

        /*
        When first player is at the seventh move, possible states of the games are 49392. Below is the description
		In sixth move of second player, possible winning moves are 5328.
		Now, possible positions from previous state are 54720. We need to subtract 5328 from this number will give us 49392 (54720 - 5328)
		We need to multiply this number (13680) with empty cells , i.e 3 (in the seventh move), which will give us total 148176 possible moves
        */
        for (int i = 0; i < sixMovesSequence.size(); i++) {

            INDArray seventhArraySequence = sixMovesSequence.get(i);

            for (int j = 0; j < 9; j++) {

                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(seventhArraySequence, temp1);
                if (seventhArraySequence.getInt(j) == 0) {

                    temp1.putScalar(new int[]{0, j}, 7);

                    if (CheckWins(temp1, true)) {
                        seventhMoveWins.add(temp1);
                    } else {
                        seventhMovesSequence.add(temp1);
                    }
                }
            }
        }

        System.out.println("Total Win In 7th Move : " + seventhMoveWins.size());
        WriteFile(filePath + "SeventhWiningData.txt", seventhMoveWins);

        sixMovesSequence.clear();

        /*
		When second player is at the eighth move, possible states of the games are 100224. Below is the description
		In seventh move of fisrt player, possible winning moves are 47952.
		Now, possible positions from previous state are 148176. We need to subtract 47952 from this number will give us 100224 (148176 - 47952)
		We need to multiply this number (100224) with empty cells , i.e 2 (in the eighth move), which will give us total 200448 possible moves
        */
        List<INDArray> eighthMovesSequence = new ArrayList<INDArray>();
        List<INDArray> eighthMoveWins = new ArrayList<INDArray>();

        for (int i = 0; i < seventhMovesSequence.size(); i++) {
            INDArray eighthArraySequence = seventhMovesSequence.get(i);

            for (int j = 0; j < 9; j++) {

                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(eighthArraySequence, temp1);

                if (eighthArraySequence.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 8);
                    if (CheckWins(temp1, false)) {
                        eighthMoveWins.add(temp1);
                    } else {
                        eighthMovesSequence.add(temp1);
                    }
                }
            }
        }

        System.out.println("Total Win In 8th Move : " + eighthMoveWins.size());
        WriteFile(filePath + "EighthWiningData.txt", eighthMoveWins);
        seventhMovesSequence.clear();


        /*
		When first player is at the nineth move, possible states of the games are 127872. Below is the description
		In eigth move of second player, possible winning moves are 72576.
		Now, possible positions from previous state are 200448. We need to subtract 72576 from this number will give us 127872 (200448-72576)
		We need to multiply this number (127872) with empty cells , i.e 1 (in the nineth move), which will give us total 127872 possible moves
        */
        List<INDArray> nineMovesSequence = new ArrayList<INDArray>();
        List<INDArray> nineMoveWins = new ArrayList<INDArray>();

        for (int i = 0; i < eighthMovesSequence.size(); i++) {

            INDArray nineArraySequence = eighthMovesSequence.get(i);

            for (int j = 0; j < 9; j++) {

                INDArray temp1 = Nd4j.zeros(1, 9);
                Nd4j.copy(nineArraySequence, temp1);
                if (nineArraySequence.getInt(j) == 0) {
                    temp1.putScalar(new int[]{0, j}, 9);
                    if (CheckWins(temp1, true)) {
                        nineMoveWins.add(temp1);
                    } else {
                        nineMovesSequence.add(temp1);
                    }
                }
            }
        }

        /*
        For Ninenth move for first Player, out of 127872 moves, he can win only on 81792 states, if not, game will be drawn for remaining states, i.e. 127872-81792 =46080.
        */
        eighthMovesSequence.clear();
        eighthMoveWins.clear();

        System.out.println("Total win in 9th move : " + nineMoveWins.size());
        WriteFile(filePath + "NineWiningData.txt", nineMoveWins);

        nineMoveWins.clear();

        System.out.println("Draw Games : " + nineMovesSequence.size());
        WriteFile(filePath + "DrawGames.txt", nineMovesSequence);

        nineMovesSequence.clear();
    }

    /**
     * Identify the game state win/Draw.
     */
    public static boolean CheckWins(INDArray sequence, boolean isOdd) {
        double vpos1 = sequence.getDouble(0);
        double vpos2 = sequence.getDouble(1);
        double vpos3 = sequence.getDouble(2);
        double vpos4 = sequence.getDouble(3);
        double vpos5 = sequence.getDouble(4);
        double vpos6 = sequence.getDouble(5);
        double vpos7 = sequence.getDouble(6);
        double vpos8 = sequence.getDouble(7);
        double vpos9 = sequence.getDouble(8);

        boolean pos1 = isOdd ? (sequence.getDouble(0) % 2.0 != 0) : (sequence.getDouble(0) % 2.0 == 0);
        boolean pos2 = isOdd ? (sequence.getDouble(1) % 2.0 != 0) : (sequence.getDouble(1) % 2.0 == 0);
        boolean pos3 = isOdd ? (sequence.getDouble(2) % 2.0 != 0) : (sequence.getDouble(2) % 2.0 == 0);
        boolean pos4 = isOdd ? (sequence.getDouble(3) % 2.0 != 0) : (sequence.getDouble(3) % 2.0 == 0);
        boolean pos5 = isOdd ? (sequence.getDouble(4) % 2.0 != 0) : (sequence.getDouble(4) % 2.0 == 0);
        boolean pos6 = isOdd ? (sequence.getDouble(5) % 2.0 != 0) : (sequence.getDouble(5) % 2.0 == 0);
        boolean pos7 = isOdd ? (sequence.getDouble(6) % 2.0 != 0) : (sequence.getDouble(6) % 2.0 == 0);
        boolean pos8 = isOdd ? (sequence.getDouble(7) % 2.0 != 0) : (sequence.getDouble(7) % 2.0 == 0);
        boolean pos9 = isOdd ? (sequence.getDouble(8) % 2.0 != 0) : (sequence.getDouble(8) % 2.0 == 0);

        if (((pos1 && pos2 && pos3) && (vpos1 != 0 && vpos2 != 0 && vpos3 != 0)) ||
            ((pos4 && pos5 && pos6) && (vpos4 != 0 && vpos5 != 0 && vpos6 != 0)) ||
            ((pos7 && pos8 && pos9) && (vpos7 != 0 && vpos8 != 0 && vpos9 != 0)) ||
            ((pos1 && pos4 && pos7) && (vpos1 != 0 && vpos4 != 0 && vpos7 != 0)) ||
            ((pos2 && pos5 && pos8) && (vpos2 != 0 && vpos5 != 0 && vpos8 != 0)) ||
            ((pos3 && pos6 && pos9) && (vpos3 != 0 && vpos6 != 0 && vpos9 != 0)) ||
            ((pos1 && pos5 && pos9) && (vpos1 != 0 && vpos5 != 0 && vpos9 != 0)) ||
            ((pos3 && pos5 && pos7) && (vpos3 != 0 && vpos5 != 0 && vpos7 != 0))) {

            return true;
        } else {
            return false;
        }
    }

    /**
     * Write file to save all games in file system
     */
    public static void WriteFile(String fileName, List<INDArray> input) {
        //Save all game in file systeam
        try (FileWriter writer = new FileWriter(fileName)) {
            for (int index = 0; index < input.size(); index++) {
                INDArray arrayFromInputlist = input.get(index);
                String tempString = arrayFromInputlist.toString();

                tempString = tempString.replace('[', ' ');
                tempString = tempString.replace(']', ' ');

                String output = tempString;
                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        } catch (Exception Io) {
            System.out.println(Io.toString());
        }
    }
}

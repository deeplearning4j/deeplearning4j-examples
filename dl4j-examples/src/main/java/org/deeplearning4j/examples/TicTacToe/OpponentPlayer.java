package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt. Ltd.( www.kitsol.com) on 24-Aug-16.
 * This is supporting class for AI Player(When Computer plays as second player).
 */

public class OpponentPlayer {

    java.util.List<INDArray> opponentPlayerMoveList = new ArrayList<INDArray>();
    java.util.List<Integer> opponentPlayerMoveIndexList = new ArrayList<Integer>();
    java.util.List<Double> probabilityList = new ArrayList<Double>();
    TicTacToeGameTrainer trainerReferences;

    OpponentPlayer(TicTacToeGameTrainer trainer) {
        trainerReferences = trainer;
    }

    /**
     * This method gives the next possible State for the game using passed board state (TicTacToe Board)
     */
    public List<INDArray> getNextPossibleStateBoards(int[] board) {

        INDArray inputArray = Nd4j.zeros(1, 9);
        List<INDArray> returnList = new ArrayList<INDArray>();


        for (int k = 0; k < 9; k++) {
            inputArray.putScalar(new int[]{0, k}, board[k]);
        }

        for (int i = 0; i < inputArray.length(); i++) {

            INDArray tempArray = Nd4j.zeros(1, 9);
            Nd4j.copy(inputArray, tempArray);
            double digit = inputArray.getDouble(i);

            if (digit == 0) {
                tempArray.putScalar(new int[]{0, i}, 2);
                returnList.add(tempArray);
            }
        }
        return returnList;
    }

    /**
     * This method returns best next move based on the passing board position
     */
    public INDArray getNextBestMove(int[] board) {

        double maxNumber = 0;
        int indexInArray = 0;
        INDArray nextMove = null;

        List<INDArray> listOfNextPossibleMove = getNextPossibleStateBoards(board);

        for (int index = 0; index < listOfNextPossibleMove.size(); index++) {

            INDArray positionArray = listOfNextPossibleMove.get(index);
            Move m = trainerReferences.getNextBestMove(positionArray);
            int indexInMoveList = m.index;
            double Probability = m.probability;

            if (maxNumber <= Probability) {
                maxNumber = Probability;
                indexInArray = indexInMoveList;
                nextMove = positionArray;
            }
        }

        probabilityList.add(maxNumber);
        opponentPlayerMoveList.add(nextMove);
        opponentPlayerMoveIndexList.add(indexInArray);

        return nextMove;
    }

    /*
    * Reward the State base on game loose,win and Draw.
    *
    */
    public void updateProbability(int win) {

        double probabilityValue = 0.0;
        int previousIndex = 0;

        for (int p = (opponentPlayerMoveIndexList.size() - 1); p >= 0; p--) {

            previousIndex = opponentPlayerMoveIndexList.get(p);

            if (p == (opponentPlayerMoveIndexList.size() - 1)) {
                if (win == 1) {
                    probabilityValue = 0.0;  //loass
                } else if (win == 0) {
                    probabilityValue = 1.0;  //Win
                } else {
                    probabilityValue = 0.5; //Draw
                }
            } else {
                double probabilityFromPreviousStep = probabilityList.get(p);
                probabilityValue = probabilityFromPreviousStep + 0.1 * (probabilityValue - probabilityFromPreviousStep);
            }
            trainerReferences.updateStateList(previousIndex, (Double) probabilityValue);
        }
        //Clear the List
        opponentPlayerMoveList.clear();
        opponentPlayerMoveIndexList.clear();
        probabilityList.clear();

    }
}

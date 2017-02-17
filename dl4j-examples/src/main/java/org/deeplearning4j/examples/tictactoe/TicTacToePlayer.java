package org.deeplearning4j.examples.tictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <b>Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com)</b> on 24-Aug-16.
 * This program does following tasks.
 * - loads tictactoe data file
 * - provide next best move depending on the previous passed
 * - reset the board when a game is over.
 * - checks whether game is finished.
 * - update probability of each move made in lost or won game when game is finished
 */
public class TicTacToePlayer implements Runnable {

    // To synchronise access of stateList and stateProbabilityList.
    Lock lock = new ReentrantLock();
    // holds path of data file to load data from.
    private String filePath = "";
    // holds data for state and probability loaded from data file.
    private List<INDArray> stateList = new ArrayList<>();
    private List<Double> stateProbabilityList = new ArrayList<>();

    /**
     * Stores a index position from stateList to hold all states from sateList list
     * e.g. if move made by first player is at the 5th position in stateList, then "indexListForPlayer1" will hold 5
     * This is required to update probability of particular state in stateList List when game is finished.
     * This is stored for both player separately for a single game and will be cleared at the end of the game after
     * updating probability.
     */
    private List<Integer> indexListForPlayer1 = new ArrayList<>();
    private List<Integer> indexListForPlayer2 = new ArrayList<>();
    // flag to control update of probability in a data file.
    private boolean updateAIAutomatic = false;
    //Stores game decision at any time. 0-For continue/Not started, 1-For Player1 wins,2-For Player2 wins,3-game Drawn
    private int gameDecision = 0;
    // controls whether data file is loaded or not. used in run() method.
    private boolean aiLoad = false;
    // class variable to hold number of games after which you want to update probability in data file.
    private int updateLimit = 0;
    // private class variable to control number of games played to allow program to update probability after updateLimit number of games.
    private int gameCounter = 0;
    // allows client class to set a flag whether update probability or not in data file. If this flag is false, updateLimit is of no use.
    private boolean updateAIFile = false;

    /**
     * Thread method to load or save data file asynchronously.
     */
    @Override
    public void run() {
        readStateAndRewardFromFile();
        while (true) {
            try {
                if (updateAIFile == true) {
                    updateAIFile = false;
                    saveToFile();
                }
                Thread.sleep(100);
            } catch (Exception e) {
                System.out.println("Exception in File Updatable" + e.toString());
            }
        }
    }

    /**
     * to check whether data is loaded from data file into stateList and stateProbabilityList.
     */
    public boolean isAILoad() {
        return aiLoad;
    }

    /**
     * To retrieve best next move provided current board and player number (i.e. first or second player)
     */
    public INDArray getNextBestMove(INDArray board, int playerNumber) {
        double maxNumber = 0;
        int indexInArray = 0;
        INDArray nextMove = null;
        boolean boardEmpty = isBoardEmpty(board);
        if (boardEmpty == false) {
            if (playerNumber == 1 && indexListForPlayer2.size() == 0) {
                int indexInList = stateList.indexOf(board);
                if (indexInList != -1) {
                    indexListForPlayer2.add(indexInList);
                }
            } else if (playerNumber == 2 && indexListForPlayer1.size() == 0) {
                int indexInList = stateList.indexOf(board);
                if (indexInList != -1) {
                    indexListForPlayer1.add(indexInList);
                }
            }
        }
        List<INDArray> listOfNextPossibleMove = getPossibleBoards(board, playerNumber);
        try {
            lock.lock();
            for (int index = 0; index < listOfNextPossibleMove.size(); index++) {
                INDArray positionArray = listOfNextPossibleMove.get(index);
                int indexInStateList = stateList.indexOf(positionArray);
                double probability = 0;
                if (indexInStateList != -1) {
                    probability = stateProbabilityList.get(indexInStateList);
                }
                if (maxNumber <= probability) {
                    maxNumber = probability;
                    indexInArray = indexInStateList;
                    nextMove = positionArray;
                }
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        } finally {
            lock.unlock();
        }
        boolean isGameOver = false;
        if (playerNumber == 1) {
            indexListForPlayer1.add(indexInArray);
            isGameOver = isGameFinish(nextMove, true);
        } else {
            indexListForPlayer2.add(indexInArray);
            isGameOver = isGameFinish(nextMove, false);
        }
        if (isGameOver == true) {
            reset();
        }
        return nextMove;
    }

    /**
     * Checks if board is completely empty or not?
     */
    private boolean isBoardEmpty(INDArray board) {
        for (int i = 0; i < board.length(); i++) {
            double digit = board.getDouble(i);
            if (digit > 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * resets index id list for all moves made by both users.
     */
    public void reset() {
        indexListForPlayer1.clear();
        indexListForPlayer2.clear();
    }

    /**
     * Checks whether game is finished or not by checking three horizontal, three vertical and two diagonal moves made by any player.
     */
    private boolean isGameFinish(INDArray board, boolean isOdd) {
        boolean isGameOver = false;
        double boardPosition1 = board.getDouble(0);
        double boardPosition2 = board.getDouble(1);
        double boardPosition3 = board.getDouble(2);
        double boardPosition4 = board.getDouble(3);
        double boardPosition5 = board.getDouble(4);
        double boardPosition6 = board.getDouble(5);
        double boardPosition7 = board.getDouble(6);
        double boardPosition8 = board.getDouble(7);
        double boardPosition9 = board.getDouble(8);

        boolean position1 = isOdd ? (board.getDouble(0) % 2.0 != 0) : (board.getDouble(0) % 2.0 == 0);
        boolean position2 = isOdd ? (board.getDouble(1) % 2.0 != 0) : (board.getDouble(1) % 2.0 == 0);
        boolean position3 = isOdd ? (board.getDouble(2) % 2.0 != 0) : (board.getDouble(2) % 2.0 == 0);
        boolean position4 = isOdd ? (board.getDouble(3) % 2.0 != 0) : (board.getDouble(3) % 2.0 == 0);
        boolean position5 = isOdd ? (board.getDouble(4) % 2.0 != 0) : (board.getDouble(4) % 2.0 == 0);
        boolean position6 = isOdd ? (board.getDouble(5) % 2.0 != 0) : (board.getDouble(5) % 2.0 == 0);
        boolean position7 = isOdd ? (board.getDouble(6) % 2.0 != 0) : (board.getDouble(6) % 2.0 == 0);
        boolean position8 = isOdd ? (board.getDouble(7) % 2.0 != 0) : (board.getDouble(7) % 2.0 == 0);
        boolean position9 = isOdd ? (board.getDouble(8) % 2.0 != 0) : (board.getDouble(8) % 2.0 == 0);

        if (((position1 && position2 && position3) && (boardPosition1 != 0 && boardPosition2 != 0 && boardPosition3 != 0)) ||
            ((position4 && position5 && position6) && (boardPosition4 != 0 && boardPosition5 != 0 && boardPosition6 != 0)) ||
            ((position7 && position8 && position9) && (boardPosition7 != 0 && boardPosition8 != 0 && boardPosition9 != 0)) ||
            ((position1 && position4 && position7) && (boardPosition1 != 0 && boardPosition4 != 0 && boardPosition7 != 0)) ||
            ((position2 && position5 && position8) && (boardPosition2 != 0 && boardPosition5 != 0 && boardPosition8 != 0)) ||
            ((position3 && position6 && position9) && (boardPosition3 != 0 && boardPosition6 != 0 && boardPosition9 != 0)) ||
            ((position1 && position5 && position9) && (boardPosition1 != 0 && boardPosition5 != 0 && boardPosition9 != 0)) ||
            ((position3 && position5 && position7) && (boardPosition3 != 0 && boardPosition5 != 0 && boardPosition7 != 0))) {

            gameCounter++;

            if (isOdd == true) {
                gameDecision = 1;
                updateReward(0, indexListForPlayer1); //Win player_1
                updateReward(1, indexListForPlayer2); //loose player_2
            } else {
                gameDecision = 2;
                updateReward(0, indexListForPlayer2);//Win player_2
                updateReward(1, indexListForPlayer1);//loose player_1
            }
            isGameOver = true;
            reset();
        } else {
            isGameOver = true;
            for (int i = 0; i < 9; i++) {
                if (((int) board.getDouble(i)) == 0) {
                    isGameOver = false;
                    gameDecision = 0;
                    break;
                }
            }
            //Draw for both player
            if (isGameOver == true) {
                gameDecision = 3;
                gameCounter++;
                updateReward(2, indexListForPlayer1);
                updateReward(2, indexListForPlayer2);
                reset();
            }
        }
        return isGameOver;
    }

    /**
     * Calculate probability of any won or lost or draw game at the end of the game and update stateList and stateProbabilityList.
     * It uses "Temporal Difference" formula to calculate probability of each game move.
     */
    private void updateReward(int win, List<Integer> playerMoveIndexList) {
        if (updateAIAutomatic == false) {
            return;
        }
        if ((gameCounter >= updateLimit) && updateAIAutomatic == true) {
            gameCounter = 0;
            updateAIFile = true;
        }
        double probabilityValue = 0.0;
        int previousIndex = 0;
        try {
            lock.lock();
            for (int p = (playerMoveIndexList.size() - 1); p >= 0; p--) {

                previousIndex = playerMoveIndexList.get(p);

                if (p == (playerMoveIndexList.size() - 1)) {
                    if (win == 1) {
                        probabilityValue = 0.0;  //loose
                    } else if (win == 0) {
                        probabilityValue = 1.0;  //Win
                    } else {
                        probabilityValue = 0.5; //Draw
                    }
                } else {
                    double probabilityFromPreviousStep = stateProbabilityList.get(previousIndex);
                    probabilityValue = probabilityFromPreviousStep + 0.1 * (probabilityValue - probabilityFromPreviousStep); //This is temporal difference formula for calculating reward for state
                }
                stateProbabilityList.set(previousIndex, (Double) probabilityValue);
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        } finally {
            lock.unlock();
        }
    }

    /**
     * This function returns list of all possible boards states provided current board
     * This will be used to calculate best move for the next player to play
     */
    private List<INDArray> getPossibleBoards(INDArray board, int playerNumber) {
        List<INDArray> returnList = new ArrayList<>();

        for (int i = 0; i < board.length(); i++) {
            INDArray inputArray = Nd4j.zeros(1, 9);
            Nd4j.copy(board, inputArray);
            double digit = board.getDouble(i);
            if (digit == 0) {
                inputArray.putScalar(new int[]{0, i}, playerNumber);
                returnList.add(inputArray);
            }
        }
        return returnList;
    }

    /**
     * This is the function to load data file into stateList and stateProbabilityList lists
     */
    private void readStateAndRewardFromFile() {

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = "";
            lock.lock();
            while ((line = br.readLine()) != null) {
                INDArray input = Nd4j.zeros(1, 9);
                String[] nextLine = line.split(" ");
                String tempLine1 = nextLine[0];
                String tempLine2 = nextLine[1];
                String testLine[] = tempLine1.split(":");
                for (int i = 0; i < 9; i++) {
                    double number = Double.parseDouble(testLine[i]);
                    input.putScalar(new int[]{0, i}, number);
                }
                double doubleNumber = Double.parseDouble(tempLine2);
                stateList.add(input);
                stateProbabilityList.add(doubleNumber);
                aiLoad = true;
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        } finally {
            lock.unlock();
        }
    }

    /**
     * Function to save current data in stateList and stateProbabilityList into data file.
     */
    private void saveToFile() {
        try (FileWriter writer = new FileWriter(filePath);) {
            lock.lock();
            for (int index = 0; index < stateList.size(); index++) {
                INDArray arrayFromInputList = stateList.get(index);
                double rewardValue = stateProbabilityList.get(index);

                String tempString = arrayFromInputList.toString().replace('[', ' ').replace(']', ' ').replace(',', ':').replaceAll("\\s", "");
                String output = tempString + " " + String.valueOf(rewardValue);
                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        } catch (Exception i) {
            System.out.println(i.toString());
        } finally {
            lock.unlock();
        }
    }

    /**
     * returns current state of the game, i.e. won, lose, draw or in progress.
     */
    public int getGameDecision() {
        int currentResult = gameDecision;
        gameDecision = 0;
        return currentResult;
    }

    /**
     * Sets a file name (with full path) to be used to load data from.
     */
    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    /**
     * This function is used to tell TicTacToePlayer to update probability in data file.
     * data file is not updated if you set this as false.
     * This property is false by default
     */
    public void setAutoUpdate(boolean updateAI) {
        updateAIAutomatic = updateAI;
    }

    /**
     * set a limit of number of games after which user wants to update data file from stateList and stateProbabilityList.
     */
    public void setUpdateLimit(int updateLimit) {
        this.updateLimit = updateLimit;
    }

    public void addBoardToList(INDArray board, int playerNumber) {
        int indexInStateList = stateList.indexOf(board);
        if (indexInStateList != -1) {
            boolean isGameOver = false;
            if (playerNumber == 1) {
                indexListForPlayer1.add(indexInStateList);
                isGameOver = isGameFinish(board, true);
            } else {
                indexListForPlayer2.add(indexInStateList);
                isGameOver = isGameFinish(board, false);
            }
        }
    }
}

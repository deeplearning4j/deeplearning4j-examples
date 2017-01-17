package org.deeplearning4j.examples.TicTacToe;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com) on 24-Aug-16.
 * This program is used for training.(Update the move reward based on the win or loose
 * Here both player are AI and update probability.
 */
public class TicTacToeGameTrainer implements Runnable {
    public OpponentPlayer opponentPlayerObject;
    public String filePath = "";
    public int[] board;
    java.util.List<INDArray> stateList = new ArrayList<INDArray>();
    java.util.List<Double> stateProbabilityList = new ArrayList<Double>();
    java.util.List<INDArray> xPlayerMoveList = new ArrayList<INDArray>();
    java.util.List<Integer> xPlayerMoveProbabilityList = new ArrayList<Integer>();
    boolean isGameUpdate;
    boolean isAIFirstPlayer;
    int xPlayer = 0;
    int oPlayer = 0;
    int draw = 0;
    int noOfGamePlay = 0;
    boolean isFileLoad = false;
    boolean isGamePlayRandom = false; //


    TicTacToeGameTrainer() {
        try {
            filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";
        } catch (Exception e) {
            System.out.println("FilePathException" + e.toString());
        }

        board = new int[9];
        isGameUpdate = false;
        isAIFirstPlayer = true;
        xPlayer = 0;
        oPlayer = 0;
        draw = 0;
        isFileLoad = false;
        opponentPlayerObject = new OpponentPlayer(this);
    }

    public static void main(String[] args) {

        int totalPlayCounter = 1000;

        TicTacToeGameTrainer ticTacToeGameTrainerObject = new TicTacToeGameTrainer();

        Thread aiLoad = new Thread(ticTacToeGameTrainerObject);
        aiLoad.start();
        ticTacToeGameTrainerObject.initializeGameBoard();
        List<INDArray> listOfNextPossibleMove = ticTacToeGameTrainerObject.getOtherBoard();
        try {
            for (int p = 0; p < listOfNextPossibleMove.size(); p++) {

                int kPlay = 0;
                System.out.println("Position Change For X player");
                INDArray nextPosition = listOfNextPossibleMove.get(p);

                while (true) {

                    if (ticTacToeGameTrainerObject.isFileLoad == true) {

                        ticTacToeGameTrainerObject.playFirstStep(nextPosition);
                        // t1.PlayAI();
                        kPlay++;
                    }
                    if (kPlay > totalPlayCounter) {
                        break;
                    }
                    Thread.sleep(10);
                }
                ticTacToeGameTrainerObject.saveToFile();
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
        ticTacToeGameTrainerObject.isGameUpdate = true;
    }


    /*
    * Second Player may be AI or Random Play .
    */

    public void secondPlayerPlay() {

        boolean isPlayUpdate = false;

        //Enable For Random Play

        if (isGamePlayRandom == true) {

            while (isPlayUpdate == false) {

                try {
                    Random rand = new Random();
                    int randomNum = 0 + rand.nextInt((9 - 0) + 1);
                    isPlayUpdate = updateRandomPlay(randomNum, 2);
                    Thread.sleep(1);
                } catch (Exception e) {
                    System.out.println(e.toString());
                }
            }
        } else {

            INDArray nextMove = opponentPlayerObject.getNextBestMove(board);

            if (nextMove != null) {
                updateStateOnBoard(nextMove, 2);
            } else {
                System.out.println("Null Found At O Position");
            }
        }

        boolean isGameOver = gameFinish(2);
        if (isGameOver == false) {
            playAI();
        }
    }

    /*First Player as some move statically using following method.
    * */

    public void playFirstStep(INDArray positionArray) {

        int indexInMoveList = stateList.indexOf(positionArray);

        if (positionArray != null) {
            updateStateOnBoard(positionArray, 1);
            xPlayerMoveList.add(positionArray);
            xPlayerMoveProbabilityList.add(indexInMoveList);
        }

        boolean isGameOver = gameFinish(1);

        if (isGameOver == false) {
            secondPlayerPlay();
        }
    }


    /* Machine(AI) play itself best move using this method.
    *
    */
    public void playAI() {

        List<INDArray> listOfNextPossibleMove = getOtherBoard();
        double maxNumber = 0;
        int indexInArray = 0;
        INDArray nextMove = null;

        for (int index = 0; index < listOfNextPossibleMove.size(); index++) {

            INDArray positionArray = listOfNextPossibleMove.get(index);
            int indexInMoveList = stateList.indexOf(positionArray);
            double probability = stateProbabilityList.get(indexInMoveList);

            if (maxNumber <= probability) {
                maxNumber = probability;
                indexInArray = indexInMoveList;
                nextMove = positionArray;
            }
        }

        if (nextMove != null) {
            updateStateOnBoard(nextMove, 1);
            xPlayerMoveList.add(nextMove);
            xPlayerMoveProbabilityList.add(indexInArray);
        }

        boolean isGameOver = gameFinish(1);
        if (isGameOver == false) {
            secondPlayerPlay();
        }
    }

    /*
    * This function gives the probability of State from stored StateList and Probability List.
    *
    */
    public Move getNextBestMove(INDArray positionArray) {

        Move m = new Move();
        int indexInArray = stateList.indexOf(positionArray);
        double probability = stateProbabilityList.get(indexInArray);
        m.index = indexInArray;
        m.probability = probability;
        return m;
    }

    /*
    * Update the reward against particular State.
    *
    */
    public void updateStateList(int indexPosition, Double probabilityValue) {
        double valueForOpponentPlayerUpdate = this.stateProbabilityList.set(indexPosition, probabilityValue);
    }

    /*
    *  Update the State on TicTacToe board.
    */
    public void updateStateOnBoard(INDArray nextMove, int player) {

        for (int i = 0; i < 9; i++) {

            if (board[i] != ((int) nextMove.getDouble(i))) {
                board[i] = player;
                break;
            }
        }
    }

    /*
    * This method is used for random move update when second player play as random State.
    * */
    public boolean updateRandomPlay(int position, int player) {

        boolean isBoardUpdate = false;

        if (board[position] == 0) {
            board[position] = player;
            isBoardUpdate = true;
        }
        return isBoardUpdate;
    }

    /* Initialize the  game board*/
    public void initializeGameBoard() {

        board = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0};
        noOfGamePlay++;
    }

    /*
    * Using this method,Check whether game finish or not for any player and also reward to that particular state base on the game decision.
    *
    */
    public boolean gameFinish(int player) {
        printBoard();
        boolean isGameFinish = false;

        if (board[0] == player && board[1] == player && board[2] == player ||
            board[3] == player && board[4] == player && board[5] == player ||
            board[6] == player && board[7] == player && board[8] == player ||
            board[0] == player && board[3] == player && board[6] == player ||
            board[1] == player && board[4] == player && board[7] == player ||
            board[2] == player && board[5] == player && board[8] == player ||
            board[0] == player && board[4] == player && board[8] == player ||
            board[2] == player && board[4] == player && board[6] == player) {

            if (player == 1) {
                //Update the Smart Move Table
                updateProbability(1);        // if FirstPlayer win the game,then update the reward for firstPlayer(i.e indicate as 1).

                if (isGamePlayRandom == false) {
                    opponentPlayerObject.updateProbability(1); // if secondPlayer loose the game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                xPlayer++;
            } else {

                updateProbability(0);        // if FirstPlayer loose the game,then update the reward for firstPlayer(i.e indicate as 1).
                if (isGamePlayRandom == false) {
                    opponentPlayerObject.updateProbability(0); // if secondPlayer win the game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                oPlayer++;
            }
            isGameFinish = true;
        } else {

            isGameFinish = true;

            for (int index = 0; index < 9; index++) {

                if (board[index] == 0) {
                    isGameFinish = false;
                    break;
                }
            }
            if (isGameFinish == true) {

                updateProbability(2);  // if FirstPlayer draw game,then update the reward for firstPlayer(i.e indicate as 1).

                if (isGamePlayRandom == false) {
                    opponentPlayerObject.updateProbability(2); // if secondPlayer draw game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                draw++;
            }
        }

        if (isGameFinish == true) {
            System.out.println("    Total Game :" + String.valueOf(noOfGamePlay));
            System.out.println("       X Player:" + String.valueOf(xPlayer));
            System.out.println("       O Player:" + String.valueOf(oPlayer));
            System.out.println("       XXDrawOO:" + String.valueOf(draw));
            initializeGameBoard();
        }
        return isGameFinish;
    }

    /*
    * Print the TicTacToe game board
    * */
    public void printBoard() {

        System.out.println("-------------------------------------------------------");
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print("  " + board[k]);
                k++;
            }
            System.out.println("");
        }
    }

    /**
     * This method gives next possible State of the game based on the current board.
     * For the First player,When current board is blank then it will give 9 possible state of game as follow
     * <p>
     * State
     * 1,0,0,
     * 0,0,0,
     * 0,0,0
     * <p>
     * State
     * 0,1,0,
     * 0,0,0,
     * 0,0,0
     * <p>
     * State
     * 0,0,1,
     * 0,0,0,
     * 0,0,0
     * <p>
     * State
     * 0,0,0,
     * 1,0,0,
     * 0,0,0
     * <p>
     * State
     * 0,0,0,
     * 0,1,0,
     * 0,0,0
     * <p>
     * State
     * 0,0,0,
     * 0,0,1,
     * 0,0,0
     * <p>
     * State
     * 0,0,0,
     * 0,0,0,
     * 1,0,0
     * <p>
     * State
     * 0,0,0,
     * 0,0,0,
     * 0,1,0
     * <p>
     * State
     * 0,0,0,
     * 0,0,0,
     * 0,0,1
     */
    public List<INDArray> getOtherBoard() {

        INDArray inputArray = Nd4j.zeros(1, 9);
        List<INDArray> returnList = new ArrayList<INDArray>();


        for (int k = 0; k < 9; k++) {
            inputArray.putScalar(new int[]{0, k}, board[k]);
        }

        for (int i = 0; i < inputArray.length(); i++) {

            INDArray newTempArray2 = Nd4j.zeros(1, 9);
            Nd4j.copy(inputArray, newTempArray2);
            double digit = inputArray.getDouble(i);

            if (digit == 0) {
                if (isAIFirstPlayer == true) {
                    newTempArray2.putScalar(new int[]{0, i}, 1);
                } else {
                    newTempArray2.putScalar(new int[]{0, i}, 2);
                }
                returnList.add(newTempArray2);
            }
        }
        return returnList;
    }


    /*
    * Update reward base on the player win,loose and Draw the game.
    *
    */
    public void updateProbability(int win) {

        int previousIndex = 0;
        double probabilityValue = 0.0;

        for (int p = (xPlayerMoveList.size() - 1); p >= 0; p--) {

            previousIndex = xPlayerMoveProbabilityList.get(p);

            if (p == (xPlayerMoveList.size() - 1)) {
                if (win == 0) {
                    probabilityValue = 0.0; //Loose
                } else if (win == 1) {
                    probabilityValue = 1.0; //Win
                } else {
                    probabilityValue = 0.5; //Draw
                }
            } else {
                double probabilityFromPreviousStep = stateProbabilityList.get(previousIndex);
                probabilityValue = probabilityFromPreviousStep + 0.1 * (probabilityValue - probabilityFromPreviousStep);
            }
            stateProbabilityList.set(previousIndex, probabilityValue);
        }
        xPlayerMoveList.clear();
        xPlayerMoveProbabilityList.clear();
    }


    /*
    * This method use for the load the move and it reward in memory
    * It populate the statelist and state_probabilitylist
    */
    public void readStateAndRewardFromFile() {

        String inputFileDataSet = filePath + "SmartAIMove.csv"; //First Input the this file and then after use new genrated file "G:\TicTacToe Update\AllMove\SmartAIMove.csv"

        try (BufferedReader br = new BufferedReader(new FileReader(inputFileDataSet))) {

            String line = "";

            while ((line = br.readLine()) != null) {

                INDArray input = Nd4j.zeros(1, 9);
                String[] nextLine = line.split(" ");
                String tempLine1 = nextLine[0];
                String tempLine2 = nextLine[1];

                String testLine[] = tempLine1.split(":");

                for (int i = 0; i < 9; i++) {

                    int number = Integer.parseInt(testLine[i]);
                    input.putScalar(new int[]{0, i}, number);
                }

                double doubleNumber = Double.parseDouble(tempLine2);
                stateList.add(input);
                stateProbabilityList.add(doubleNumber);
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    /*
    * Save updated reward value against state in file system */

    public void saveToFile() {

        try (FileWriter writer = new FileWriter(filePath + "SmartAIMove.csv");) {

            for (int index = 0; index < stateList.size(); index++) {

                String tempString1 = "";
                INDArray arrayFromInputList = stateList.get(index);
                double probabilityNumber = stateProbabilityList.get(index);
                int sizeOfInput = arrayFromInputList.length();

                for (int i = 0; i < sizeOfInput; i++) {

                    int number = (int) arrayFromInputList.getDouble(i);
                    tempString1 = tempString1 + String.valueOf(number).trim();

                    if (i != (sizeOfInput - 1)) {
                        tempString1 += ":";
                    }
                }

                String tempString2 = String.valueOf(probabilityNumber);
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

    /*
    * Using this thread ,Load the State and its reward from file system.
    * */
    @Override
    public void run() {
        //Load the network
        readStateAndRewardFromFile();
        isFileLoad = true;

        while (true) {
            try {
                if (isGameUpdate == true) {
                    isGameUpdate = false;
                    break;
                }
                Thread.sleep(10000);
            } catch (Exception e) {
                System.out.println("Exception in File Updatation");
            }
        }
    }
}


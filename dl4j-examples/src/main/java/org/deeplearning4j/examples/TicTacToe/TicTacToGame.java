package org.deeplearning4j.examples.TicTacToe;

import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt.Ltd. ( www.kitsol.com ) on 24-Aug-16.
 * This is a GUI to allow user to play game against the trained network stored in "SmartAIMove.csv"
 */

public class TicTacToGame extends JFrame implements Runnable {
    public String lastMove = "O";
    public boolean isAIFirstPlayer = true;
    public boolean isAILoad = false;
    public String filePath;
    String playerInformation = "FirstPlayer:X";
    JFrame frame = new JFrame("TicTacToe");                    //Global frame and grid button variables
    JButton[] gridMoveButton = new JButton[9];
    JButton startButton = new JButton("Start");
    JButton switchButton = new JButton("Switch Player");
    JLabel infoLabel = new JLabel(playerInformation);
    java.util.List<INDArray> arrayListForX = new ArrayList<INDArray>();
    java.util.List<INDArray> arrayListForO = new ArrayList<INDArray>();
    java.util.List<INDArray> arrayListForAI = new ArrayList<INDArray>();
    java.util.List<Integer> positionListForX = new ArrayList<Integer>();
    java.util.List<Double> probabilityForAI = new ArrayList<Double>();
    int xWon = 0;
    int oWon = 0;
    int draw = 0;

    public TicTacToGame() {

        super();
        try {
            filePath = new ClassPathResource("TicTacToe").getFile().toString() + "\\";
        } catch (Exception e) {
            System.out.println("FilePathException" + e.toString());
        }
        frame.setSize(350, 450);
        frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
        frame.setVisible(true);
        frame.setResizable(false);
    }

    //main method and instantiating tic tac object and calling initialize function
    public static void main(String[] args) {

        TicTacToGame game = new TicTacToGame();
        Thread t1 = new Thread(game); //Thread for AI Smart move load
        game.initialize(); //Initialize the game
        t1.start(); // Load the AI Smart Move File in thread ny runing thread
    }

    /*
    * Using this method,Create the GUI for TicTacToe board game.
    */
    private void initialize() {

        JPanel mainPanel = new JPanel(new BorderLayout());
        JPanel menu = new JPanel(new BorderLayout());
        JPanel tital = new JPanel(new BorderLayout());
        JPanel game = new JPanel(new GridLayout(3, 3));

        frame.add(mainPanel);

        mainPanel.setPreferredSize(new Dimension(325, 425));
        menu.setPreferredSize(new Dimension(300, 50));
        tital.setPreferredSize(new Dimension(300, 50));
        game.setPreferredSize(new Dimension(300, 300));

        //Create the basic layout for game

        mainPanel.add(menu, BorderLayout.NORTH);
        mainPanel.add(tital, BorderLayout.AFTER_LINE_ENDS);
        mainPanel.add(game, BorderLayout.SOUTH);
        tital.add(infoLabel, BorderLayout.CENTER);
        menu.add(startButton, BorderLayout.WEST);
        menu.add(switchButton, BorderLayout.EAST);

        //Create the 9 Grid button on UI
        for (int i = 0; i < 9; i++) {

            gridMoveButton[i] = new JButton();
            gridMoveButton[i].setText(" ");
            gridMoveButton[i].setVisible(true);
            gridMoveButton[i].setEnabled(false);
            gridMoveButton[i].addActionListener(new MyActionListener(i));
            game.add(gridMoveButton[i]);
        }

        game.setVisible(true);
        startButton.setEnabled(false);
        switchButton.setEnabled(false);

        //Start Button Click Listener.
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Clear all List for Next Game
                arrayListForX.clear();
                positionListForX.clear();
                arrayListForO.clear();
                //Reset Button for Next Game
                for (int i = 0; i < 9; i++) {

                    gridMoveButton[i].setText(" ");
                    gridMoveButton[i].setEnabled(true);
                }

                switchButton.setEnabled(true);

                if (isAIFirstPlayer == true) {

                    INDArray firstMove = Nd4j.zeros(1, 9);
                    playUsingAI(firstMove);
                }
            }
        });

        //switch Button Click Listener
        switchButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                if (isAIFirstPlayer == true) {
                    playerInformation = "FirstPlayer:O";
                    isAIFirstPlayer = false;
                } else {
                    playerInformation = "FirstPlayer:X";
                    isAIFirstPlayer = true;
                }

                String updateInformation = playerInformation + "    X:" + String.valueOf(xWon) + "    O:" + String.valueOf(oWon) + "    Draw:" + String.valueOf(draw);

                infoLabel.setText(updateInformation);

                for (int i = 0; i < 9; i++) {
                    gridMoveButton[i].setText(" ");
                    gridMoveButton[i].setEnabled(false);
                }

                arrayListForX.clear();
                positionListForX.clear();
                arrayListForO.clear();
            }
        });
    }

    /* For AI Player,
    *  By passing currentBoard position.Find the next possible moves.from that find the best move base on probability and play the best move.    *
    */

    /*This method gives the next possible State for the game using passed State (IND Array)
    * */
    public List<INDArray> getNextPossibleStateBoards(INDArray inputArray) {

        List<INDArray> returnList = new ArrayList<INDArray>();

        for (int i = 0; i < inputArray.length(); i++) {

            INDArray tempArray = Nd4j.zeros(1, 9);
            Nd4j.copy(inputArray, tempArray);
            double digit = inputArray.getDouble(i);

            if (digit == 0) {

                if (isAIFirstPlayer == true) {
                    tempArray.putScalar(new int[]{0, i}, 1);   // if AI play as first player  then move position indicate as 1
                } else {
                    tempArray.putScalar(new int[]{0, i}, 2);  // if AI play as second player  then move position indicate as 2
                }
                returnList.add(tempArray);
            }
        }
        return returnList;
    }

    public void playNextMove(INDArray currentBoard) {

        List<INDArray> listOfNextPossibleMove = getNextPossibleStateBoards(currentBoard);
        double maxNumber = 0;
        int indexInArray = 0;
        INDArray nextMove = null;

        for (int index = 0; index < listOfNextPossibleMove.size(); index++) {

            INDArray positionArray = listOfNextPossibleMove.get(index);
            int indexInMoveList = arrayListForAI.indexOf(positionArray);
            double probability = probabilityForAI.get(indexInMoveList);

            if (maxNumber <= probability) {
                maxNumber = probability;
                indexInArray = indexInMoveList;
                nextMove = positionArray;
            }
        }

        if (nextMove != null) {
            updateNextMoveToBoard(currentBoard, nextMove);
            arrayListForX.add(nextMove);
            positionListForX.add(indexInArray);
        }
    }

    /*Update the Current board using best move for the AI Player
    * */
    public void updateNextMoveToBoard(INDArray currentBoard, INDArray nextMove) {

        for (int i = 0; i < currentBoard.length(); i++) {

            if (currentBoard.getDouble(i) != nextMove.getDouble(i)) {
                gridMoveButton[i].setText("X");
                break;
            }
        }
    }

    /*It returns the probability of state base on the index value
    * */
    public double getProbabilityFromArray(int index) {
        return probabilityForAI.get(index);
    }

    /*
    * Reward the State base on game loose,win and Draw.
    *
    */
    public void updateProbability(boolean xWin) {

        double probabilityValue = 0.0;
        int previousIndex = 0;

        for (int p = (positionListForX.size() - 1); p >= 0; p--) {

            previousIndex = positionListForX.get(p);

            if (p == (positionListForX.size() - 1)) {

                if (xWin == false) {
                    probabilityValue = 0.0;
                } else {
                    probabilityValue = 1.0;
                }
            } else {

                double probabilityFromPreviousStep = getProbabilityFromArray(previousIndex);
                probabilityValue = probabilityFromPreviousStep + 0.1 * (probabilityValue - probabilityFromPreviousStep);
            }
            probabilityForAI.set(previousIndex, probabilityValue);
        }
        positionListForX.clear();
        arrayListForX.clear();
    }

    /* Play Game using the AI
    * */
    public void playUsingAI(INDArray inputArray) {

        playNextMove(inputArray);
        isGameFinish("X");
    }

    /*
    * This method is use for game decision and also reward the State sequences
    * */
    private boolean isGameFinish(String winTestStr) {

        boolean isGameWon = false;

        if (gridMoveButton[0].getText().equals(winTestStr) && gridMoveButton[1].getText().equals(winTestStr) && gridMoveButton[2].getText().equals(winTestStr) ||
            gridMoveButton[3].getText().equals(winTestStr) && gridMoveButton[4].getText().equals(winTestStr) && gridMoveButton[5].getText().equals(winTestStr) ||
            gridMoveButton[6].getText().equals(winTestStr) && gridMoveButton[7].getText().equals(winTestStr) && gridMoveButton[8].getText().equals(winTestStr) ||
            gridMoveButton[0].getText().equals(winTestStr) && gridMoveButton[3].getText().equals(winTestStr) && gridMoveButton[6].getText().equals(winTestStr) ||
            gridMoveButton[1].getText().equals(winTestStr) && gridMoveButton[4].getText().equals(winTestStr) && gridMoveButton[7].getText().equals(winTestStr) ||
            gridMoveButton[2].getText().equals(winTestStr) && gridMoveButton[5].getText().equals(winTestStr) && gridMoveButton[8].getText().equals(winTestStr) ||
            gridMoveButton[0].getText().equals(winTestStr) && gridMoveButton[4].getText().equals(winTestStr) && gridMoveButton[8].getText().equals(winTestStr) ||
            gridMoveButton[2].getText().equals(winTestStr) && gridMoveButton[4].getText().equals(winTestStr) && gridMoveButton[6].getText().equals(winTestStr)) {

            for (int i = 0; i < 9; i++) {
                gridMoveButton[i].setEnabled(false);
            }

            if (winTestStr.equals("X")) {
                xWon++;
            } else if (winTestStr.equals("O")) {
                oWon++;
            }

            if (winTestStr.equals("X") && (isAIFirstPlayer == true)) {
                updateProbability(true); //Update the Move reward with X win game.
            } else if (winTestStr.equals("O") && (isAIFirstPlayer == true)) {
                updateProbability(false); //Update the Move reward with X loose game.
            }

            isGameWon = true;
            String updateInformation = playerInformation + "    X:" + String.valueOf(xWon) + "    O:" + String.valueOf(oWon) + "    Draw:" + String.valueOf(draw);
            infoLabel.setText(updateInformation);
        } else {

            boolean isBlankMove = false;

            for (int i = 0; i < 9; i++) {

                if (gridMoveButton[i].getText().equals(" ") == true) {

                    isBlankMove = true;
                    isGameWon = false;
                    break;
                }
            }

            if (isBlankMove == false) {

                draw++;
                String updateInformation = playerInformation + "    X:" + String.valueOf(xWon) + "    O:" + String.valueOf(oWon) + "    Draw:" + String.valueOf(draw);
                infoLabel.setText(updateInformation);

                for (int i = 0; i < 9; i++) {
                    gridMoveButton[i].setEnabled(false);
                }
                isGameWon = true;  //may be require true
            }
        }

        if (isGameWon == true) {
            //enable the following code if you want to save updated AI move
            //saveUpdatedMoveInFile();
        }
        return isGameWon;
    }

    /*This method update the UI(Board) and State of game base on the user click Event.
    * */
    public void userNextMove(int indexPosition) {

        String gridMoveButtonText = gridMoveButton[indexPosition].getText();

        if (gridMoveButtonText.equals(" ")) {

            gridMoveButton[indexPosition].setText(lastMove);
            INDArray testArray = getCurrentStateOfBoard();
            arrayListForO.add(testArray);

            if (isGameFinish(lastMove) == false) {
                playUsingAI(testArray);
            }
        }
    }

    /*
    * This function gives the current state board in INDArray
    * */
    public INDArray getCurrentStateOfBoard() {

        INDArray positionArray = Nd4j.zeros(1, 9);

        for (int i = 0; i < 9; i++) {

            String gridMoveButtonValue = gridMoveButton[i].getText();

            if (isAIFirstPlayer == true) {

                if (gridMoveButtonValue.equals("X")) {
                    positionArray.putScalar(new int[]{0, i}, 1);
                } else if (gridMoveButtonValue.equals("O")) {
                    positionArray.putScalar(new int[]{0, i}, 2);
                }
            } else {

                if (gridMoveButtonValue.equals("O")) {
                    positionArray.putScalar(new int[]{0, i}, 1);
                } else if (gridMoveButtonValue.equals("X")) {
                    positionArray.putScalar(new int[]{0, i}, 2);
                }
            }
        }
        return positionArray;
    }

    //To save Updated AI move  in file systeam

    public void saveUpdatedMoveInFile() {

        try (FileWriter writer = new FileWriter(filePath + "SmartAIMove.csv")) {

            for (int index = 0; index < arrayListForAI.size(); index++) {

                INDArray arrayFromInputList = arrayListForAI.get(index);
                double probabilityNumber = probabilityForAI.get(index);

                String tempString1 = "";
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

        String inputFileDataSet = filePath + "SmartAIMove.csv";

        try (BufferedReader br = new BufferedReader(new FileReader(inputFileDataSet))) {

            String line = "";

            while ((line = br.readLine()) != null) {

                INDArray input = Nd4j.zeros(1, 9);
                String[] nextLine = line.split(" ");
                String tempLine1 = "";
                String tempLine2 = "";

                tempLine1 = nextLine[0];
                tempLine2 = nextLine[1];

                String testLine[] = tempLine1.split(":");

                for (int i = 0; i < 9; i++) {

                    int number = Integer.parseInt(testLine[i]);
                    input.putScalar(new int[]{0, i}, number);
                }

                double doubleNumber = Double.parseDouble(tempLine2);
                arrayListForAI.add(input);
                probabilityForAI.add(doubleNumber);
            }

            isAILoad = true;
            startButton.setEnabled(true);
            switchButton.setEnabled(true);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    /*This is Action listener for move buttons
    * */
    private class MyActionListener implements ActionListener {

        private int index;

        public MyActionListener(int index) {
            this.index = index;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            userNextMove(index);
        }
    }
}

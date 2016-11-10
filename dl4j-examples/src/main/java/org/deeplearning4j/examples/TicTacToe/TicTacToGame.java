package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.io.FileWriter;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com ) on 24-Aug-16.
 * Please update file path based on your dir.
 * This is the final game board for TicTacToe with "SmartAIMove.csv"
 */

public class TicTacToGame extends JFrame implements Runnable
{
    String playerInformation="FirstPlayer:X";
    JFrame frame = new JFrame("TicTacToe");                    //Global frame and grid button variables
    JButton [] gridMoveButton = new  JButton[9];
    JButton startButton = new JButton("Start");
    JButton switchButton = new JButton("Switch Player");
    JLabel infoLabel = new JLabel(playerInformation);

    public String lastmove ="O";
    public boolean isAIFirstPlayer =true;
    public boolean isAILoad=false;

    java.util.List<INDArray> arrayListforX = new ArrayList<INDArray>();
    java.util.List<INDArray> arrayListforO = new ArrayList<INDArray>();
    java.util.List<INDArray> arrayListforAI = new ArrayList<INDArray>();

    java.util.List<Integer> positionListforX = new ArrayList<Integer>();
    java.util.List<Double>  probabilityforAI= new ArrayList<Double>();

    boolean isFirstTime = true;
    int xwon=0;
    int owon=0;
    int draw=0;
    public String filepath ;
    public TicTacToGame()
    {
        super();
        filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\" ;
        frame.setSize(350, 450);
        frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
        frame.setVisible(true);
        frame.setResizable(false);
    }
     /*
     * Using this method,Create the GUI for TicTacToe board game.
     */
    private void initialize()
    {
        JPanel mainPanel = new JPanel(new BorderLayout());
        JPanel menu = new JPanel(new BorderLayout());
        JPanel tital = new JPanel(new BorderLayout());
        JPanel game = new JPanel(new GridLayout(3,3));

        frame.add(mainPanel);

        mainPanel.setPreferredSize(new Dimension(325,425));
        menu.setPreferredSize(new Dimension(300,50));
        tital.setPreferredSize(new Dimension(300,50));
        game.setPreferredSize(new Dimension(300,300));

        //Create the basic layout for game

        mainPanel.add(menu, BorderLayout.NORTH);
        mainPanel.add(tital, BorderLayout.AFTER_LINE_ENDS);
        mainPanel.add(game, BorderLayout.SOUTH);
        tital.add(infoLabel,BorderLayout.CENTER);
        menu.add(startButton, BorderLayout.WEST);
        menu.add(switchButton, BorderLayout.EAST);

        //Create the 9 Grid button on UI
        for(int i = 0; i < 9; i++)
        {
            gridMoveButton[i]  =  new JButton();
            gridMoveButton[i].setText(" ");
            gridMoveButton[i].setVisible(true);
            gridMoveButton[i].setEnabled (false);
            gridMoveButton[i].addActionListener(new MyActionListener(i));
            game.add(gridMoveButton[i]);
        }

        game.setVisible(true);
        startButton.setEnabled(false);
        switchButton.setEnabled(false);

        //Start Button Click Listener.
        startButton.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                //Clear all List for Next Game
                arrayListforX.clear();
                positionListforX.clear();
                arrayListforO.clear();
                //Reset Button for Next Game
                for(int i = 0; i < 9; i++)
                {
                    gridMoveButton[i].setText(" ");
                    gridMoveButton[i].setEnabled(true);
                }
                switchButton.setEnabled(true);
                if(isAIFirstPlayer==true)
                {
                    INDArray firstMove = Nd4j.zeros(1, 9);
                    playUsingAI(firstMove);
                }
            }
        });

        //switch Button Click Listener
        switchButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                if(isAIFirstPlayer==true)
                {
                    playerInformation="FirstPlayer:O";
                    isAIFirstPlayer=false;
                }
                else
                {
                    playerInformation="FirstPlayer:X" ;
                    isAIFirstPlayer=true;
                }
                String updateInformation = playerInformation + "    X:" +  String.valueOf(xwon) + "    O:"+String.valueOf(owon)+ "    Draw:"+String.valueOf(draw);
                infoLabel.setText(updateInformation);
                for(int i = 0; i < 9; i++)
                {
                    gridMoveButton[i].setText(" ");
                    gridMoveButton[i].setEnabled(false);
                }
                arrayListforX.clear();
                positionListforX.clear();
                arrayListforO.clear();
            }
        });
    }
    /*This method gives the next possible State for the game using passed State (IND Array)
    * */

    public List<INDArray>  getNextPossibleStateBoards(INDArray i_inputArray)
    {
        List<INDArray> returnList =  new ArrayList<INDArray>();
        for(int i=0 ;i <i_inputArray.length();i++)
        {
            INDArray tempArray = Nd4j.zeros(1,9);
            Nd4j.copy(i_inputArray,tempArray);
            double digit = i_inputArray.getDouble(i);
            if(digit==0)
            {
                if(isAIFirstPlayer==true)
                    tempArray.putScalar(new int[] {0,i},1);   // if AI play as first player  then move position indicate as 1
                else
                    tempArray.putScalar(new int[] {0,i},2);  // if AI play as second player  then move position indicate as 2

                returnList.add(tempArray);
            }
        }
        return returnList;
    }
    /* For AI Player,
    *  By passing currentBoard position.Find the next possible moves.from that find the best move base on probability and play the best move.    *
    * */

    public void playNextMove(INDArray i_currentBoard)
    {
        List<INDArray> listOfNextPossibalMove = getNextPossibleStateBoards(i_currentBoard);

        double maxNumber=0;
        int indexInArray=0;

        INDArray nextMove= null;

        for(int index=0; index < listOfNextPossibalMove.size();index++)
        {
            INDArray positionArray = listOfNextPossibalMove.get(index);
            int indexInMoveList= arrayListforAI.indexOf(positionArray);
            double Probability = probabilityforAI.get(indexInMoveList);
            if(maxNumber <= Probability)
            {
                maxNumber = Probability;
                indexInArray=indexInMoveList;
                nextMove = positionArray;
            }
        }
        if(nextMove!= null)
        {
            updateNextMoveToBoard(i_currentBoard,nextMove);
            arrayListforX.add(nextMove);
            positionListforX.add(indexInArray);
        }
    }

    /*Update the Current board using best move for the AI Player
    * */
    public void updateNextMoveToBoard(INDArray i_currentBoard,INDArray nextMove)
    {
        for (int i =0 ;i < i_currentBoard.length();i++)
        {
            if (i_currentBoard.getDouble(i) != nextMove.getDouble(i) )
            {
                gridMoveButton[i].setText("X");
                break;
            }
        }
    }
    /*It returns the probability of state base on the index value
    * */
    public double getProbabilityfromArray(int index)
    {
        return probabilityforAI.get(index) ;
    }



    /*
    * Reward the State base on game loose,win and Draw.
    *
    */
    public void updateProbability(boolean i_xwin) //Reward the State base on win or loose
    {
        double probVal=0.0;
        int k=0;
        int PreviousIndex=0;

        for(int p=positionListforX.size()-1; p >=0;p--)
        {
            PreviousIndex = positionListforX.get(p);
            if (p == positionListforX.size() - 1)
            {
                if (i_xwin == false) {
                    probVal = 0.0;
                } else {
                    probVal = 1.0;
                }
            }
            else
            {
                double probabilityfromPreviousStep = getProbabilityfromArray(PreviousIndex);
                probVal = probabilityfromPreviousStep + 0.1 * (probVal - probabilityfromPreviousStep);
            }
            probabilityforAI.set(PreviousIndex, probVal);
        }
        positionListforX.clear();
        arrayListforX.clear();
    }

    /* Play Game using the AI
    * */
    public void playUsingAI(INDArray i_inputArray)
    {
        playNextMove(i_inputArray);
        isGameFinish("X");
    }
    /*
    * This method is use for game decision and also reward the State sequences
    * */
    private boolean isGameFinish(String i_winTestStr)
    {

        boolean isgamewon=false ;
        if (gridMoveButton[0].getText().equals(i_winTestStr) && gridMoveButton[1].getText().equals(i_winTestStr) && gridMoveButton[2].getText().equals(i_winTestStr) ||
            gridMoveButton[3].getText().equals(i_winTestStr) && gridMoveButton[4].getText().equals(i_winTestStr) && gridMoveButton[5].getText().equals(i_winTestStr) ||
            gridMoveButton[6].getText().equals(i_winTestStr) && gridMoveButton[7].getText().equals(i_winTestStr) && gridMoveButton[8].getText().equals(i_winTestStr) ||
            gridMoveButton[0].getText().equals(i_winTestStr) && gridMoveButton[3].getText().equals(i_winTestStr) && gridMoveButton[6].getText().equals(i_winTestStr) ||
            gridMoveButton[1].getText().equals(i_winTestStr) && gridMoveButton[4].getText().equals(i_winTestStr) && gridMoveButton[7].getText().equals(i_winTestStr) ||
            gridMoveButton[2].getText().equals(i_winTestStr) && gridMoveButton[5].getText().equals(i_winTestStr) && gridMoveButton[8].getText().equals(i_winTestStr) ||
            gridMoveButton[0].getText().equals(i_winTestStr) && gridMoveButton[4].getText().equals(i_winTestStr) && gridMoveButton[8].getText().equals(i_winTestStr) ||
            gridMoveButton[2].getText().equals(i_winTestStr) && gridMoveButton[4].getText().equals(i_winTestStr) && gridMoveButton[6].getText().equals(i_winTestStr) )
        {
            for(int i=0;i<9 ;i++) {
                gridMoveButton[i].setEnabled(false);
            }
            if(i_winTestStr.equals("X"))
                xwon++;
            else if(i_winTestStr.equals("O"))
                owon++;

            if (i_winTestStr.equals("X") && isAIFirstPlayer==true)
            {
                updateProbability(true); //Update the Move reward with X win game.
            }
            else if (i_winTestStr.equals("O") && isAIFirstPlayer==true )
            {
                updateProbability(false); //Update the Move reward with X loose game.
            }
            isgamewon=true;
            String updateInformation = playerInformation + "    X:" +  String.valueOf(xwon) + "    O:"+String.valueOf(owon)+ "    Draw:"+String.valueOf(draw);
            infoLabel.setText(updateInformation);
        }
        else
        {
            boolean isblankmove=false;
            for(int i = 0; i < 9; i++)
            {
                if (gridMoveButton[i].getText().equals(" ")==true)
                {
                    isblankmove=true;
                    isgamewon=false;
                    break;
                }
            }
            if(isblankmove==false)
            {
                draw++;
                String updateInformation = playerInformation + "    X:" +  String.valueOf(xwon) + "    O:"+String.valueOf(owon)+ "    Draw:"+String.valueOf(draw);
                infoLabel.setText(updateInformation);

                for(int i=0;i<9 ;i++) {
                    gridMoveButton[i].setEnabled(false);
                }
                isgamewon=true;  //may be require true
            }
        }
        if(isgamewon==true)
        {
             //enable the following code if you want to save updated AI move
             //saveUpdatedMoveInFile();
        }
        return isgamewon;
    }

    /*This method update the UI(Board) and State of game base on the user click Event.
    * */
    public void userNextMove(int i_indexPosition)
    {
        String gridMoveButtonText = gridMoveButton[i_indexPosition].getText() ;
        if(gridMoveButtonText.equals(" "))
        {
            gridMoveButton[i_indexPosition].setText(lastmove);
            INDArray testArray= getCurrentStateBoard();
            arrayListforO.add(testArray);
            if(isGameFinish(lastmove)==false)
            {
                playUsingAI(testArray);
            }
        }
    }
    /*
    * This function gives the current state board in INDArray
    * */
    public INDArray  getCurrentStateBoard()
    {
        INDArray posArray = Nd4j.zeros(1,9);
        for(int i=0;i<9 ;i++)
        {
            String gridMoveButtonVal=gridMoveButton[i].getText();
            if(isAIFirstPlayer==true)
            {
                if(gridMoveButtonVal.equals("X")) {
                    posArray.putScalar(new int[]{0,i},1);
                }
                else if(gridMoveButtonVal.equals("O")) {
                    posArray.putScalar(new int[]{0,i},2);
                }
            }
            else
            {
                if(gridMoveButtonVal.equals("O")) {
                    posArray.putScalar(new int[]{0,i},1);
                }
                else if(gridMoveButtonVal.equals("X")) {
                    posArray.putScalar(new int[]{0,i},2);
                }
            }
        }
        return posArray;
    }

    //main method and instantiating tic tac object and calling initialize function
    public static void main(String[] args)
    {
        TicTacToGame game = new TicTacToGame();
        Thread t1 = new Thread(game); //Thread for AI Smart move load
        game.initialize(); //Initialize the game
        t1.start(); // Load the AI Smart Move File in thread ny runing thread
    }

    //To save Updated AI move  in file systeam
    public void saveUpdatedMoveInFile()
    {
        //arrayListforAI.add(input);
        //probabilityforAI.add(dnumber);

        try(FileWriter  writer = new FileWriter(filepath+ "SmartAIMove.csv"))
        {
            for(int index=0 ;index <arrayListforAI.size();index++)
            {
                INDArray arrfromInputlist = arrayListforAI.get(index);
                double probabiltyNumber = probabilityforAI.get(index);

                String tempstring1="";
                int sizeofInput = arrfromInputlist.length();
                for(int i =0 ; i <sizeofInput;i++)
                {
                    int number =  (int)arrfromInputlist.getDouble(i);
                    tempstring1 =  tempstring1 + String.valueOf(number).trim();
                    if(i!=sizeofInput-1)
                        tempstring1 +=":";
                }
                String tempstring2 = String.valueOf(probabiltyNumber);
                String output= tempstring1 +" " +tempstring2 ;
                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        }
        catch (Exception i)
        {
            System.out.println(i.toString());
        }
    }

    /*
    * Using this thread ,Load the State and its reward from file system.
    * */
    @Override
    public void run()
    {

        String inputfiledataset = filepath+ "SmartAIMove.csv";
        try(BufferedReader br = new BufferedReader(new FileReader(inputfiledataset)))
        {
            String line = "";
            while ((line = br.readLine()) != null)
            {
                INDArray input = Nd4j.zeros(1, 9);
                String[] nextline = line.split(" ");
                String templine1="";
                String templine2="";

                templine1 = nextline[0];
                templine2 = nextline[1];
                String testline[] =  templine1.split(":");
                for (int i = 0; i < 9; i++)
                {
                    int number =Integer.parseInt(testline[i]) ;
                    input.putScalar(new int[]{0, i}, number);
                }
                double dnumber = Double.parseDouble(templine2);
                arrayListforAI.add(input);
                probabilityforAI.add(dnumber);
            }
            isAILoad=true;
            startButton.setEnabled(true);
            switchButton.setEnabled(true);
        }
        catch (Exception e)
        {
            System.out.println(e.toString());
        }
    }

    /*This is Action listener for move buttons
    * */
    private class MyActionListener implements ActionListener
    {
        private int index;

        public MyActionListener(int index)
        {
            this.index = index;
        }
        @Override
        public void actionPerformed(ActionEvent e)
        {
            userNextMove(index);
        }
    }
}

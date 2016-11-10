package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com) on 24-Aug-16.
 * Please update file path based on your dir.
 * This program is used for training.(Update the move reward based on the win or loose
 * Here both player are AI and update probability.
 */

public class TicTacToeGameTrainer implements Runnable
{
    java.util.List<INDArray> statelist = new ArrayList<INDArray>();
    java.util.List<Double>   state_probabilitylist = new ArrayList<Double>();

    java.util.List<INDArray> xplayerMoveList = new ArrayList<INDArray>();
    java.util.List<Integer>  xplayerMoveProbabilityList = new ArrayList<Integer>();

    public OPlayer o1 ;
    boolean isGameupdate ;
    boolean isCompterFirstPlayer;

    int xplayer=0;
    int oplayer=0;
    int draw=0;
    int NoofGamePlay=0;
    boolean isFileLoad=false;

    boolean isGamePlayRandom =false; //

    public  String filepath ="";

    public int[] board ;
    //public int[] Positionboard ;

    TicTacToeGameTrainer()
    {
        filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\" ;
        board = new int[9];
        isGameupdate=false;
        isCompterFirstPlayer=true;
        xplayer=0;
        oplayer=0;
        draw=0;
        isFileLoad=false;
        o1 = new OPlayer(this);
    }
    public static void main(String[] args)
    {
        TicTacToeGameTrainer t1 = new TicTacToeGameTrainer();
        Thread aiLoad = new Thread(t1);
        aiLoad.start();
        int TotalPlayCounter = 1000;
        t1.initializeGameBoard();
        List<INDArray> listOfNextPossibalMove = t1.getOtherBoard();
        try
        {
            for (int p = 0; p < listOfNextPossibalMove.size(); p++)
            {
                int kplay = 0;
                System.out.println("Position Change For X player");
                INDArray nextPosition = listOfNextPossibalMove.get(p);

                    while (true)
                    {
                        if (t1.isFileLoad == true)
                        {
                            t1.PlayFirstStep(nextPosition);
                            // t1.PlayAI();
                            kplay++;
                        }
                        if (kplay > TotalPlayCounter) {
                            break;
                        }
                        Thread.sleep(10);
                    }
                t1.saveToFile();
            }
        }
        catch (Exception e)
        {
            System.out.println(e.toString());
        }
        t1.isGameupdate=true;
    }



    /*
    * Second Player may be AI or Random Play .
    */

    public void secondPlayerPlay()
    {
        boolean isplayUpdate=false;

        //Enable For Random Play

        if(isGamePlayRandom==true)
        {
            while(isplayUpdate==false)
            {
                try
                {
                    Random rand = new Random();
                    int randomNum = 0 + rand.nextInt((9 - 0) + 1);
                    isplayUpdate=updateRandomPlay(randomNum,2);
                    Thread.sleep(1);
                }
                catch (Exception e)
                {
                    System.out.println(e.toString());
                }
            }
        }
        else
        {
            INDArray nextmove  = o1.getNextBestMove(board);
            if(nextmove!=null)
            {
                updateStateOnBoard(nextmove,2);
            }
            else
            {
                System.out.println("Null Found At O Position");
            }
        }
        boolean isgameOver = gameFinish(2);
        if(isgameOver==false)
        {
            PlayAI();
        }
    }

    /*First Player as some move statically using following method.
    * */

    public void PlayFirstStep(INDArray positionArray)
    {
        int indexInMoveList = statelist.indexOf(positionArray);
        if(positionArray!= null)
        {
            updateStateOnBoard(positionArray,1);
            xplayerMoveList.add(positionArray);
            xplayerMoveProbabilityList.add(indexInMoveList);
        }
        boolean isgameOver = gameFinish(1);
        if(isgameOver==false)
        {
            secondPlayerPlay();
        }
    }


    /* Machine(AI) play itself best move using this method.
    *
    */
    public void PlayAI()
    {
        List<INDArray> listOfNextPossibalMove = getOtherBoard();
        double maxNumber=0;
        int indexInArray=0;
        INDArray nextMove= null;

        for(int index=0; index < listOfNextPossibalMove.size();index++)
        {
            INDArray positionArray = listOfNextPossibalMove.get(index);
            int indexInMoveList = statelist.indexOf(positionArray);
            double Probability  = state_probabilitylist.get(indexInMoveList);
            if(maxNumber <= Probability)
            {
                maxNumber = Probability;
                indexInArray=indexInMoveList;
                nextMove = positionArray;
            }
        }
        if(nextMove!= null)
        {
            updateStateOnBoard(nextMove,1);
            xplayerMoveList.add(nextMove);
            xplayerMoveProbabilityList.add(indexInArray);
        }
        boolean isgameOver = gameFinish(1);
        if(isgameOver==false)
        {
            secondPlayerPlay();
        }
    }

    /*
    * This function gives the probability of State from stored StateList and Probability List.
    *
    */
    public Move getNextBestMove(INDArray i_positionArray)
    {
        Move m = new Move();
        int indexInArray = statelist.indexOf(i_positionArray);
        double Probability  = state_probabilitylist.get(indexInArray);
        m.m_index = indexInArray ;
        m.m_probability = Probability;
        return m ;
    }


    /*
    * Update the reward against particular State.
    *
    */
    public void updateStateList(int i_indexPosition,Double i_probabilityValue)
    {
        double valueO = this.state_probabilitylist.set(i_indexPosition,i_probabilityValue);
    }


    /*
    *  Update the State on TicTacToe board.
    * */
    public void updateStateOnBoard(INDArray nextMove,int i_player)
    {
        for (int i=0 ;i<9;i++)
        {
            if ( board[i] != (int) nextMove.getDouble(i) )
            {
                board[i]=i_player;
                break;
            }
        }
    }
    /*
    * This method is used for random move update when second player play as random State.
    * */
    public boolean updateRandomPlay(int i_position,int i_player)
    {
        boolean isboardUpdate = false ;
        if(board[i_position]==0)
        {
            board[i_position]=i_player;
            isboardUpdate=true;
        }
        return isboardUpdate;
    }

    /* Initialize the  game board*/
    public void initializeGameBoard()
    {
        board = new int[]{0,0,0,0,0,0,0,0,0} ;
        NoofGamePlay++;
    }

    /*
    * Using this method,Check whether game finish or not for any player and also reward to that particular state base on the game decision.
    *
    */
    public boolean gameFinish(int player)
    {
        printBoard();
        boolean isGameFinish = false;
        if (board[0]==player && board[1]==player && board[2]==player ||
            board[3]==player && board[4]==player && board[5]==player ||
            board[6]==player && board[7]==player && board[8]==player ||
            board[0]==player && board[3]==player && board[6]==player ||
            board[1]==player && board[4]==player && board[7]==player ||
            board[2]==player && board[5]==player && board[8]==player ||
            board[0]==player && board[4]==player && board[8]==player ||
            board[2]==player && board[4]==player && board[6]==player )
        {
            if(player==1)
            {
                //Update the Smart Move Table
                updateProbability(1);        // if FirstPlayer win the game,then update the reward for firstPlayer(i.e indicate as 1).
                if(isGamePlayRandom==false)
                {
                    o1.updateProbability(1); // if secondPlayer loose the game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                xplayer++;
            }
            else
            {
                updateProbability(0);        // if FirstPlayer loose the game,then update the reward for firstPlayer(i.e indicate as 1).
                if(isGamePlayRandom==false)
                {
                    o1.updateProbability(0); // if secondPlayer win the game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                oplayer++;
            }
            isGameFinish=true;
        }
        else
        {
            isGameFinish=true;
            for(int index=0;index<9;index++)
            {
                if(board[index]==0)
                {
                    isGameFinish=false;
                    break;
                }
            }
            if(isGameFinish==true)
            {
                updateProbability(2);  // if FirstPlayer draw game,then update the reward for firstPlayer(i.e indicate as 1).
                if(isGamePlayRandom==false)
                {
                    o1.updateProbability(2); // if secondPlayer draw game,then update the reward for secondPlayer(i.e indicate as 2).
                }
                draw++;
            }
        }
        if(isGameFinish==true)
        {
            System.out.println("    Total Game :" + String.valueOf(NoofGamePlay));
            System.out.println("       X Player:" + String.valueOf(xplayer));
            System.out.println("       O Player:" + String.valueOf(oplayer));
            System.out.println("       XXDrawOO:" + String.valueOf(draw));
            initializeGameBoard();
        }
        return  isGameFinish;
    }

    /*
    * Print the TicTacToe game board
    * */
    public void printBoard()
    {
        System.out.println("----------------");
        int k=0;
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                System.out.print("  "+board[k]);
                k++;
            }
            System.out.println("");
        }
    }

    /**
     *   This method gives next possible State of the game based on the current board.
     *   For the First player,When current board is blank then it will give 9 possible state of game as follow
     *
     *   State
     *   1,0,0,
     *   0,0,0,
     *   0,0,0
     *
     *   State
     *   0,1,0,
     *   0,0,0,
     *   0,0,0
     *
     *   State
     *   0,0,1,
     *   0,0,0,
     *   0,0,0
     *
     *   State
     *   0,0,0,
     *   1,0,0,
     *   0,0,0
     *
     *   State
     *   0,0,0,
     *   0,1,0,
     *   0,0,0
     *
     *   State
     *   0,0,0,
     *   0,0,1,
     *   0,0,0
     *
     *   State
     *   0,0,0,
     *   0,0,0,
     *   1,0,0
     *
     *   State
     *   0,0,0,
     *   0,0,0,
     *   0,1,0
     *
     *   State
     *   0,0,0,
     *   0,0,0,
     *   0,0,1
     * */
    public List<INDArray> getOtherBoard()
    {
        INDArray inputArray = Nd4j.zeros(1,9);

        for(int k=0;k<9;k++)
        {
            inputArray.putScalar(new int[]{0,k},board[k]);
        }
        List<INDArray> returnList =  new ArrayList<INDArray>();
        for(int i=0 ;i <inputArray.length();i++)
        {
            INDArray newtempArray2=Nd4j.zeros(1,9);
            Nd4j.copy(inputArray,newtempArray2);
            double digit = inputArray.getDouble(i);
            if(digit==0)
            {
                if(isCompterFirstPlayer==true)
                    newtempArray2.putScalar(new int[] {0,i},1);
                else
                    newtempArray2.putScalar(new int[] {0,i},2);

                returnList.add(newtempArray2);
            }
        }
        return returnList;
    }
    /*
    * Update reward base on the player win,loose and Draw the game.
    *
    */
    public void updateProbability(int i_win)
    {
        double probVal=0.0;
        int k=0;
        int PreviousIndex=0;

        for(int p=xplayerMoveList.size()-1; p >=0;p--)
        {
            PreviousIndex = xplayerMoveProbabilityList.get(p);
            if(p==xplayerMoveList.size()-1)
            {
                if(i_win==0)
                    probVal=0.0; //Loose
                else if(i_win==1)
                    probVal=1.0; //Win
                else
                    probVal=0.5; //Draw
            }
            else
            {   double probabilityfromPreviousStep = state_probabilitylist.get(PreviousIndex);
                probVal = probabilityfromPreviousStep  + 0.1*(probVal-probabilityfromPreviousStep);
            }
            state_probabilitylist.set(PreviousIndex,probVal);
        }
        xplayerMoveList.clear();
        xplayerMoveProbabilityList.clear();
    }
    /*
    * This method use for the load the move and it reward in memory
    * It populate the statelist and state_probabilitylist
    */
    public void readStateAndRewardFromFile ()
    {
        String inputfiledataset =  filepath + "SmartAIMove.csv" ; //First Input the this file and then after use new genrated file "G:\TicTacToe Update\AllMove\SmartAIMove.csv"
        try(  BufferedReader br = new BufferedReader(new FileReader(inputfiledataset)))
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
                statelist.add(input);
                state_probabilitylist.add(dnumber);
            }
        }
        catch (Exception e)
        {
            System.out.println(e.toString());
        }
    }

    /*
    * Save updated reward value against state in file system */
    public void saveToFile()
    {
        try(FileWriter  writer = new FileWriter(filepath+"SmartAIMove.csv");)
        {
            for(int index=0 ;index <statelist.size();index++)
            {
                INDArray arrfromInputlist = statelist.get(index);
                double probabiltyNumber = state_probabilitylist.get(index);
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
        //Load the network
        readStateAndRewardFromFile();
        isFileLoad=true;
        while(true)
        {
            try
            {
                if(isGameupdate==true)
                {
                    isGameupdate=false;
                    break;
                }
                Thread.sleep(10000);
            }
            catch (Exception e)
            {
                System.out.println("Exception in File Updatation");
            }
        }
    }
}


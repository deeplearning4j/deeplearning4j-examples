package org.deeplearning4j.examples.TicTacToe;

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
public class RewardGameState
{
    FileWriter writer ;
    List<INDArray>MiddleList = new ArrayList<>();

    RewardGameState()
    {
        try
        {
            String filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\" ;
            writer = new FileWriter(filepath+"AllMoveWithReward.txt");
        }
        catch (Exception i)
        {
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
    public void generateGameStateAndRewardToIt(INDArray i_output,int i_movetype)
    {
        INDArray maxArray = Nd4j.max(i_output );
        double maxNumber = maxArray.getDouble(0);
        List<INDArray>SequenceList = new ArrayList<>() ;
        INDArray sequenceArray = Nd4j.zeros(1,9);

        int move=1;
        int positionofDigit=0;
        for(int i=1;i<=maxNumber;i++)
        {
            INDArray newtempArray =Nd4j.zeros(1,9);
            positionofDigit=getPosition(i_output,i);
            if(i%2==i_movetype)
            {
                Nd4j.copy(sequenceArray, newtempArray);
                SequenceList.add(newtempArray);
            }
            else {
                Nd4j.copy(sequenceArray, newtempArray);
                MiddleList.add(newtempArray);
            }
            sequenceArray.putScalar(new int[] {0,positionofDigit},move);
            move=move*(-1);
        }
        move=move*(-1);
        INDArray newtempArray2 =Nd4j.zeros(1,9);
        sequenceArray.putScalar(new int[] {0,positionofDigit},move);
        Nd4j.copy(sequenceArray, newtempArray2);
        SequenceList.add(newtempArray2);
        rewardToState(SequenceList);
    }
    public static void main(String[] args) throws  Exception
    {
        String filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\" ;

        RewardGameState rewardObject = new RewardGameState();

        rewardObject.processMoveFile(filepath+"OddMove.txt",0); //Odd Position
        System.out.println("Odd Move Processed");

        rewardObject.processMoveFile(filepath+"EvenMove.txt",1); //Even Position
        System.out.println("Even Move Processed");

        rewardObject.AddExtraMove();
        System.out.println("Intermediate Move Processed"); // Intermediate Move store to the file systeam
    }


    /*
    * Here pass the Game sequences's fileName and Also pass "Game is odd player game or even player game'.
    * Using above argument,genrate the game state and also reward that state base on the win,loose and Draw
    * */
    public void processMoveFile(String i_fileName,int i_moveType)
    {
        try(BufferedReader br = new BufferedReader(new FileReader(i_fileName));)
        {
            String line = "";
            while ((line = br.readLine()) != null)
            {
                INDArray input = Nd4j.zeros(1, 9);
                String[] nextline = line.split(",");
                for (int i = 0; i < 9; i++)
                {
                    double number = (Double.parseDouble(nextline[i]));
                    input.putScalar(new int[]{0, i}, number);
                }
                generateGameStateAndRewardToIt(input,i_moveType); //0 odd Position  and 1 for Even Position
            }
        }
        catch(Exception e)
        {
            System.out.println(e.toString());
        }
    }


    /*
    * Using this method ,store the intermediate state in file with probability 0.50
    *
    */

    public void AddExtraMove()
    {
        try
        {
            for(int index=0 ;index <MiddleList.size();index++)
            {
                INDArray arrfromInputlist = MiddleList.get(index);

                String tempstring1="";
                int sizeofInput = arrfromInputlist.length();

                for(int i =0 ; i <sizeofInput;i++)
                {
                    int number =  (int)arrfromInputlist.getDouble(i);
                    tempstring1 =  tempstring1 + String.valueOf(number).trim();
                    if(i!=sizeofInput-1)
                        tempstring1 +=":";
                }
                String tempstring2 = "0.5";
                tempstring1=tempstring1.replaceAll("-1","2");
                String output= tempstring1+ " " +tempstring2 ;
                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        }
        catch (Exception Io)
        {
            System.out.println(Io.toString());
        }
    }
    public int getPosition(INDArray i_array,double i_number)
    {
        for(int i=0;i<i_array.length();i++)
        {
            if( i_array.getDouble(i)==i_number )
            {
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

    public void rewardToState (List<INDArray>arrayList)
    {
        double probVal=0;
        int sizeofArray= arrayList.size();

        INDArray probabilityArray  = Nd4j.zeros(sizeofArray,1);

        for(int p=arrayList.size()-1; p >=0;p--)
        {
            if(p==arrayList.size()-1)
            {
                probVal=1.0;
            }
            else
            {
                probVal = 0.5 + 0.1*(probVal-0.5);
            }
            probabilityArray.putScalar( new int[]{p,0},probVal);
        }
        try
        {
            for(int index=0 ;index <arrayList.size();index++)
            {
                INDArray arrfromInputlist = arrayList.get(index);
                String tempstring1="";
                int sizeofInput = arrfromInputlist.length();

                for(int i =0 ; i <sizeofInput;i++)
                {
                    int number =  (int)arrfromInputlist.getDouble(i);
                    tempstring1 =  tempstring1 + String.valueOf(number).trim();
                    if(i!=sizeofInput-1)
                        tempstring1 +=":";
                }
                String tempstring2 = String.valueOf( probabilityArray.getDouble(index));
                tempstring1=tempstring1.replaceAll("-1","2");
                String output= tempstring1+ " " + tempstring2;

                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        }
        catch (Exception Io)
        {
            System.out.println(Io.toString());
        }
    }
}


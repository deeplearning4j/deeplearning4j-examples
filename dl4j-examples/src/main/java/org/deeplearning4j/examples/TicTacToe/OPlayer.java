package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com) on 24-Aug-16.
 * Please update file path based on your dir.
 * This is supporting class for AI Player(When AI Player play as second player).
 */
public class OPlayer
{

    java.util.List<INDArray> OplayerMoveList = new ArrayList<INDArray>();
    java.util.List<Integer>  OplayerMoveIndexList = new ArrayList<Integer>();
    java.util.List<Double>  probabilityList = new ArrayList<Double>();

    TicTacToeGameTrainer m_trainer;

    /*This method gives the next possible State for the game using passed State (TicTacToe Board)
    * */

    public List<INDArray> getNextPossibleStateBoards(int []board)
    {
        INDArray i_inputArray = Nd4j.zeros(1,9);

        for(int k=0;k<9;k++)
        {
            i_inputArray.putScalar(new int[]{0,k},board[k]);
        }
        List<INDArray> returnList =  new ArrayList<INDArray>();
        for(int i=0 ;i <i_inputArray.length();i++)
        {
            INDArray newtempArray2=Nd4j.zeros(1,9);
            Nd4j.copy(i_inputArray,newtempArray2);
            double digit = i_inputArray.getDouble(i);
            if(digit==0)
            {   newtempArray2.putScalar(new int[] {0,i},2);
                returnList.add(newtempArray2);
            }
        }
        return returnList;
    }
    /*This method returns best next move based on the passing board position
    * */
    public INDArray getNextBestMove(int [] i_board)
    {
        List<INDArray> listOfNextPossibalMove = getNextPossibleStateBoards(i_board);
        double maxNumber=0;
        int indexInArray=0;
        INDArray nextMove= null;
        for(int index=0; index < listOfNextPossibalMove.size();index++)
        {
            INDArray positionArray = listOfNextPossibalMove.get(index);
            Move m = m_trainer.getNextBestMove(positionArray);
            int indexInMoveList = m.m_index ;
            double Probability = m.m_probability;
            if(maxNumber <= Probability)
            {
                maxNumber = Probability;
                indexInArray=indexInMoveList;
                nextMove = positionArray;
            }
        }
        probabilityList.add(maxNumber);
        OplayerMoveList.add(nextMove);
        OplayerMoveIndexList.add(indexInArray);
        return nextMove;
    }
    /*
    * Reward the State base on game loose,win and Draw.
    *
    */
    public void updateProbability(int i_win)
    {
        double probVal=0.0;
        int PreviousIndex=0;
        for(int p=OplayerMoveIndexList.size()-1;p>=0;p--)
        {
            PreviousIndex = OplayerMoveIndexList.get(p);
            if(p==(OplayerMoveIndexList.size()-1) )
            {
                if(i_win==1)
                {
                    probVal = 0.0;  //loass
                }
                else if(i_win==0)
                {
                    probVal=1.0;  //Win
                }
                else
                {
                    probVal = 0.5; //Draw
                }
            }
            else
            {   double probabilityfromPreviousStep = probabilityList.get(p);
                probVal = probabilityfromPreviousStep  + 0.1*(probVal-probabilityfromPreviousStep);
            }
           m_trainer.updateStateList(PreviousIndex,(Double)probVal);
        }
        //Clear the List
        OplayerMoveList.clear();
        OplayerMoveIndexList.clear();
        probabilityList.clear();

    }
    OPlayer(TicTacToeGameTrainer i_trainer)
    {
        m_trainer = i_trainer;
    }
}

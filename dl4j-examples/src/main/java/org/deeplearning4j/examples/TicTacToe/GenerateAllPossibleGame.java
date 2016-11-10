package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;

/**
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com) on 05-Aug-2016.
 * Please update file path based on your dir.
 * This program is use for generating the possible games.
 * Here,Odd Number sequence consider as first player's move and Even number sequence consider  as second player's move
 */
public class GenerateAllPossibleGame
{
    public static void main(String[] args) throws  Exception
    {
         String filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\";

         /*
         * If TicTacToe board is Empty.i.e First player with first move has 9 possible move.
         */
        List<INDArray> firstMovesSequence = new ArrayList<INDArray>();
        for(int i=0;i<9;i++)
        {
            INDArray temp2 = Nd4j.zeros(1,9);
            temp2.putScalar(new int[] {0,i}, 1);
            firstMovesSequence.add(temp2);
        }
        /*
        * For Second player with second move, 8 possible position remain.so there will be next (9)*8 =72 possible state of game.
        */
        List<INDArray> secondMovesSequence = new ArrayList<INDArray>();
        for(int i=0;i<9;i++)
        {
            INDArray fmArraySeq = firstMovesSequence.get(i);
            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                temp1.putScalar(new int[] {0,i},fmArraySeq.getInt(i));
                if(fmArraySeq.getInt(j) != 1)
                {
                    temp1.putScalar(new int[] {0,j},2);
                    secondMovesSequence.add(temp1);
                }
            }
        }
        /*
        * For First player with third move, 7 possible position remain.so there will be next (72)*7 =504 possible state of game.
        */

        List<INDArray> thirdMovesSequence = new ArrayList<INDArray>();
        for(int i=0;i<72;i++)
        {
            INDArray smArraySeq = secondMovesSequence.get(i);
            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(smArraySeq, temp1);

                if(smArraySeq.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j},3);
                    thirdMovesSequence.add(temp1);
                }
            }
        }

        /*
        * For Second player with fourth move, 6 possible position remain.so there will be next (504)*6 =3024 possible state of game.
        */
        List<INDArray> fourthMovesSequence = new ArrayList<INDArray>();
        for(int i=0;i<504;i++)
        {
            INDArray tmArraySequence = thirdMovesSequence.get(i);
            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(tmArraySequence, temp1);
                if(tmArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 4);
                    fourthMovesSequence.add(temp1);
                }
            }
        }
        List<INDArray> fifthMovesSequence = new ArrayList<INDArray>();
        List<INDArray> fifthMovesWins = new ArrayList<INDArray>();


        /*
        * For First player with fifth move, 5 possible position remain.so there will be next (3024)*5 =15120 possible state of game.
        */
        for(int i=0;i<3024;i++)
        {
            INDArray fmArraySequence = fourthMovesSequence.get(i);
            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(fmArraySequence,temp1);
                if(fmArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 5);
                    if (checkWins(temp1, true))
                        fifthMovesWins.add(temp1);
                    else
                        fifthMovesSequence.add(temp1);
                }
            }
        }

        System.out.println("Total win in 5th move : " + fifthMovesWins.size());
        WriteFile(filepath+"FifthWiningData.txt",fifthMovesWins);

        // Clear lists for 1 to 4th moves
        firstMovesSequence.clear();
        secondMovesSequence.clear();
        thirdMovesSequence.clear();
        fourthMovesSequence.clear();
        fifthMovesWins.clear();



        /*
        * For Second player with sixth move, in fifth move first player  plays the move and  will win 1440 state. so remain state is 15120-1440=13680.Possible state of game is = 13680*4= 54720.
        */

        List<INDArray> sixMovesSequence = new ArrayList<INDArray>();
        List<INDArray> sixthMovesWins = new ArrayList<INDArray>();
        for(int i=0;i<fifthMovesSequence.size();i++)
        {
            INDArray smArraySequence = fifthMovesSequence.get(i);

            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(smArraySequence,temp1);
                if(smArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 6);
                    if (checkWins(temp1, false))
                        sixthMovesWins.add(temp1);
                    else
                        sixMovesSequence.add(temp1);
                }
            }
        }

        System.out.println("Total win in 6th move : " + sixthMovesWins.size());
        WriteFile(filepath+"SixthWinningdata.txt",sixthMovesWins);

        // we can clear fifth move data here
        fifthMovesSequence.clear();

        List<INDArray> seventhMovesSequence = new ArrayList<INDArray>();
        List<INDArray> seventhMoveWins = new ArrayList<INDArray>();

        /*
        * For First player with seventh move, in sixth move second player  plays the move and  will win 5328 state. so remain state is 54720-5328=49392.Possible state of game is = 49392*3=148176.
        */

        for(int i=0;i<sixMovesSequence.size();i++)
        {
            INDArray sevArraySequence = sixMovesSequence.get(i);

            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(sevArraySequence,temp1);
                if(sevArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 7);
                    if (checkWins(temp1, true))
                        seventhMoveWins.add(temp1);
                    else
                        seventhMovesSequence.add(temp1);
                }
            }
        }
        System.out.println("Total win in 7th move : " + seventhMoveWins.size());
        WriteFile(filepath+"SevenWinningdata.txt",seventhMoveWins);

        sixMovesSequence.clear();
        /*
        * For second player with eighth move, in seventh move First player  plays the move and  will win 47952 state. so remain state is 148176-47952=100224.Possible state of game is = 100224*2=200448.
        */
        List<INDArray> eigthMovesSequence = new ArrayList<INDArray>();
        List<INDArray> eigthMoveWins = new ArrayList<INDArray>();
        for(int i=0;i<seventhMovesSequence.size();i++)
        {
            INDArray eigArraySequence = seventhMovesSequence.get(i);

            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(eigArraySequence,temp1);
                if(eigArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 8);
                    if (checkWins(temp1, false))
                        eigthMoveWins.add(temp1);
                    else
                        eigthMovesSequence.add(temp1);
                }
            }
        }


        System.out.println("Total win in 8th move : " + eigthMoveWins.size());
        WriteFile(filepath+"eightWinningdata.txt",eigthMoveWins);
        seventhMovesSequence.clear();

          /*
        * For First player with ninth move, in eighth  move second player  plays the move and  will win 72576 state. so remain state is 200448-72576=127872.Possible state of game is = 127872*1=127872.
        */

        List<INDArray> NineMovesSequence = new ArrayList<INDArray>();
        List<INDArray> NineMoveWins = new ArrayList<INDArray>();
        for(int i=0;i<eigthMovesSequence.size();i++)
        {
            INDArray nineArraySequence = eigthMovesSequence.get(i);

            for(int j=0;j<9;j++)
            {
                INDArray temp1 = Nd4j.zeros(1,9);
                Nd4j.copy(nineArraySequence,temp1);
                if(nineArraySequence.getInt(j) == 0)
                {
                    temp1.putScalar(new int[] {0,j}, 9);
                    if (checkWins(temp1, true))
                        NineMoveWins.add(temp1);
                    else
                        NineMovesSequence.add(temp1);
                }
            }
        }

         /*
        * For First Player will  win the  game  state 81792  and Draw Game state are :127872-81792 =46080.
        */

        eigthMovesSequence.clear();
        eigthMoveWins.clear();

        System.out.println("Total win in 9th move : " + NineMoveWins.size());
        WriteFile(filepath+"NineWinningdata.txt",NineMoveWins);

        NineMoveWins.clear();

        System.out.println("Draw Games : " + NineMovesSequence.size());
        WriteFile(filepath+"DrawGames.txt",NineMovesSequence);

        NineMovesSequence.clear();
    }

    /*
    * Identify the game state win/Draw.
    * */
    public static boolean checkWins(INDArray sequence, boolean isOdd)
    {
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
        if(((pos1 && pos2 && pos3) && (vpos1 != 0 && vpos2 != 0 && vpos3 != 0 )) ||
            ((pos4 && pos5 && pos6) && (vpos4 != 0 && vpos5 != 0 && vpos6 != 0 )) ||
            ((pos7 && pos8 && pos9) && (vpos7 != 0 && vpos8 != 0 && vpos9 != 0 )) ||
            ((pos1 && pos4 && pos7) && (vpos1 != 0 && vpos4 != 0 && vpos7 != 0 )) ||
            ((pos2 && pos5 && pos8) && (vpos2 != 0 && vpos5 != 0 && vpos8 != 0 )) ||
            ((pos3 && pos6 && pos9) && (vpos3 != 0 && vpos6 != 0 && vpos9 != 0 )) ||
            ((pos1 && pos5 && pos9) && (vpos1 != 0 && vpos5 != 0 && vpos9 != 0 )) ||
            ((pos3 && pos5 && pos7) && (vpos3 != 0 && vpos5 != 0 && vpos7 != 0 )))
        {
            return true;
        }
        else
            return false;
    }

    public static void WriteFile(String fileName, List<INDArray> input)
    {
        //Save all game in file systeam
        try(FileWriter  writer = new FileWriter(fileName))
        {
            for(int index=0 ;index <input.size();index++)
            {
                INDArray arrfromInputlist = input.get(index);
                String tempstring1=  arrfromInputlist.toString() ;

                tempstring1=tempstring1.replace('[',' ');
                tempstring1=tempstring1.replace(']',' ');

                String output= tempstring1;
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

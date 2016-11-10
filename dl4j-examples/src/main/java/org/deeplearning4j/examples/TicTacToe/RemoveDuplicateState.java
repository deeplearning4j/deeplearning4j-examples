package org.deeplearning4j.examples.TicTacToe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Developed by KIT Solutions Pvt,Ltd( www.kitsol.com) on 04-Aug-16.
 * Please update file path based on your dir.
 * Remove the duplicate move and update the probability.
 */
public class RemoveDuplicateState
{

    public static void main(String[] args) throws  Exception
    {
        /*
        * Read all state from file and remove the Duplicate State.
        * */
        String filepath = System.getProperty("user.dir") + "\\src\\main\\resources\\TicTacToe\\" ;

        String inputfileDataset = filepath+"AllMoveWithReward.txt";
        RemoveDuplicateState pdata = new RemoveDuplicateState();

        try(BufferedReader br = new BufferedReader(new FileReader(inputfileDataset)))
        {
            String line = "";
            while ((line = br.readLine()) != null)
            {
                INDArray input = Nd4j.zeros(1, 9);
                INDArray label = Nd4j.zeros(1, 1);

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
                label.putScalar(new int[]{0,0}, dnumber);
                pdata.processData(input,label);
            }
            String savefilepath = filepath+"DuplicateRemoved.txt";
            pdata.saveProcessdata(savefilepath);
        }
        catch(Exception e)
        {
            System.out.println(e.toString());
        }
    }
    /*
    * Using this method,Remove the duplicate state and update the maximum reward(probability).
    * */
    public void processData(INDArray i_inputlabelArray,INDArray i_outputlabel)
    {
        int indexPosition=  finalInputArray.indexOf(i_inputlabelArray) ;
        if(indexPosition!=-1)
        {
            INDArray outputArray = finalProbabilityArray.get(indexPosition);
            INDArray newUpdatedArray = this.getNewArray(outputArray,i_outputlabel);
            finalProbabilityArray.set(indexPosition,newUpdatedArray);
        }
        else
        {
            finalInputArray.add(i_inputlabelArray);
            finalProbabilityArray.add(i_outputlabel);
        }
    }
    /*
   *  Using this method,store the all unique state of game with its reward.
   * */
    public void saveProcessdata(String i_savefilepath)
    {
        try(FileWriter writer = new FileWriter(i_savefilepath) )
        {

            for(int index=0 ;index <finalInputArray.size();index++)
            {
                INDArray arrfromInputlist = finalInputArray.get(index);
                INDArray arrfromlabellist = finalProbabilityArray.get(index);

                String tempstring1="";
                int sizeofInput = arrfromInputlist.length();

                for(int i =0 ; i <sizeofInput;i++)
                {
                    int number =  (int)arrfromInputlist.getDouble(i);
                    tempstring1 =  tempstring1 + String.valueOf(number).trim();
                    if(i!=sizeofInput-1)
                        tempstring1 +=":";
                }
                String tempstring2 = String.valueOf( arrfromlabellist.getDouble(0));
                String output= tempstring1 +" " +tempstring2 ;

                writer.append(output);
                writer.append('\r');
                writer.append('\n');
                writer.flush();
            }
        }
        catch (Exception i) {
            System.out.println(i.toString());
        }
    }
    /*
    * Compare the two INDArray  and return INDArray with maximum value.
    * */
    public INDArray getNewArray(INDArray a1, INDArray b1)
    {
        INDArray newreturnArray = Nd4j.zeros(1,1);
        for(int i=0; i<a1.length();i++)
        {
            double a =a1.getDouble(i) ;
            double b =b1.getDouble(i) ;
            double max=0;

            if(a>b)
            {
                max=a;
            }
            else
            {
                max=b;
            }
            newreturnArray.putScalar(new int[]{0,i},max);
        }
        return newreturnArray;
    }

    public List<INDArray>finalInputArray  = new ArrayList<INDArray>() ;
    public List<INDArray>finalProbabilityArray = new ArrayList<INDArray>() ;

}

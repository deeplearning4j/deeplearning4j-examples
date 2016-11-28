package org.deeplearning4j.examples.feedforward.classification.detectgender;

/**
 * Created by KITS on 9/14/2016.
 */


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;


/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class PredictGenderTest implements Runnable {
    private int row=0;
    private JDialog jd;
    private JTextField jtf;
    private JLabel jlbl;
    private String possibleCharacters;
    private JLabel gender;
    private String filePath;
    private JButton btnNext;
    private JLabel genderLabel;
    private MultiLayerNetwork model;

    public static void main(String[] args) throws Exception
    {
        PredictGenderTest pgt = new PredictGenderTest();
        Thread t = new Thread(pgt);
        t.start();
        pgt.prepareInterface();
    }

    public void prepareInterface()
    {
        this.jd = new JDialog();
        this.jd.getContentPane().setLayout(null);
        this.jd.setBounds(100,100,300,250);
        this.jd.setLocationRelativeTo(null);
        this.jd.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        this.jd.setTitle("Predict Gender By Name");
        this.jd.setVisible(true);

        //jd.add(jp);

        this.jlbl = new JLabel();
        this.jlbl.setBounds(5,10,100,20);
        this.jlbl.setText("Enter Name : ");
        this.jd.add(jlbl);

        this.jtf = new JTextField();
        this.jtf.setBounds(105,10,150,20);
        this.jd.add(jtf);

        this.genderLabel = new JLabel();
        this.genderLabel.setBounds(5,12,70,170);
        this.genderLabel.setText("Gender : ");
        this.jd.add(genderLabel);

        this.gender = new JLabel();
        this.gender.setBounds(75,12,75,170);
        this.jd.add(gender);

        this.btnNext = new JButton();
        this.btnNext.setBounds(5,150,150,20);
        this.btnNext.setText("Predict");

        this.btnNext.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                if (!jtf.getText().isEmpty()) {
                    String binaryData = getBinaryString(jtf.getText().toLowerCase());
                    //System.out.println("binaryData : " + binaryData);
                    String[] arr = binaryData.split(",");
                    int[] db = new int[arr.length];
                    INDArray features = Nd4j.zeros(1, 235);
                    for (int i = 0; i < arr.length; i++) {
                        features.putScalar(new int[]{0, i}, Integer.parseInt(arr[i]));
                    }
                    INDArray predicted = model.output(features);
                    //System.out.println("output : " + predicted);
                    if (predicted.getDouble(0) > predicted.getDouble(1))
                        gender.setText("Female");
                    else if (predicted.getDouble(0) < predicted.getDouble(1))
                        gender.setText("Male");
                    else
                        gender.setText("Both male and female can have this name");
                }
                else
                    gender.setText("Enter name please..");
            }
        });

        this.jd.add(this.btnNext);
    }

    private String getBinaryString(String name)
    {
        String binaryString ="";
        for (int j = 0; j < name.length(); j++)
        {
            String fs = pad(Integer.toBinaryString(possibleCharacters.indexOf(name.charAt(j))),5);
            binaryString = binaryString + fs;
        }
        int diff = 0;
        if(name.length() < 47 )
        {
            diff = 47 - name.length();
            for(int i=0;i<diff;i++)
            {
                binaryString = binaryString + "00000";
            }
        }

        String tempStr = "";
        for(int i=0;i<binaryString.length();i++)
        {
            tempStr = tempStr + binaryString.charAt(i) + ",";
        }
        return tempStr;
    }

    private String pad(String string,int total_length)
    {
        String str = string;
        int diff = 0;
        if(total_length > string.length())
            diff = total_length - string.length();
        for(int i=0;i<diff;i++)
            str = "0" + str;
        return str;
    }

    public void run()
    {
        try
        {
            this.filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\PredictGender\\Data\\";
            this.possibleCharacters = " abcdefghijklmnopqrstuvwxyz";
            this.model = ModelSerializer.restoreMultiLayerNetwork(this.filePath + "PredictGender.net");
        }
        catch(Exception e)
        {
            System.out.println("Exception : " + e.getMessage());
        }
    }
}

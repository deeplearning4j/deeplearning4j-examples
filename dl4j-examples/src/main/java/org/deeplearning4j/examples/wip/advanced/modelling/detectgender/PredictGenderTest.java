/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.wip.advanced.modelling.detectgender;

/**
 * Created by KITS on 9/14/2016.
 */


import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import static org.deeplearning4j.examples.wip.advanced.modelling.detectgender.GenderRecordReader.nameToBinary;


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
                    String binaryData = nameToBinary(jtf.getText().toLowerCase());
                    //System.out.println("binaryData : " + binaryData);
                    String[] arr = binaryData.split(",");
                    INDArray features = Nd4j.zeros(1, 440);
                    for (int i = 0; i < arr.length; i++) {
                        features.putScalar(new int[]{0, i}, Integer.parseInt(arr[i]));
                    }
                    INDArray predicted = model.output(features);
                    System.out.println("output : " + predicted);
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

        this.jd.setVisible(true);
    }

    public void run()
    {
        try
        {
            this.filePath = DownloaderUtility.PREDICTGENDERDATA.Download() + "/Data";
            this.model = MultiLayerNetwork.load(new File(this.filePath + "PredictGender.net"), true);
        }
        catch(Exception e)
        {
            System.out.println("Exception : " + e.getMessage());
        }
    }
}

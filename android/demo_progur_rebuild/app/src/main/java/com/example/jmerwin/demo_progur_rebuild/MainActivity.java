package com.example.jmerwin.demo_progur_rebuild;

import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                createAndUseNetwork();
            }
        });
    }

    private void createAndUseNetwork() {

        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(2)
                .nOut(3)
                .name("Input")
                .build();
        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(3)
                .nOut(2)
                .name("Hidden")
                .build();
        OutputLayer outputLayer = new OutputLayer.Builder()
                .nIn(2)
                .nOut(1)
                .name("Output")
                .build();
        NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
        nncBuilder.updater(Updater.ADAM);

        NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
        listBuilder.layer(0, inputLayer);
        listBuilder.layer(1, hiddenLayer);
        listBuilder.layer(2, outputLayer);

        listBuilder.backprop(true);

        MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
        myNetwork.init();

        final int NUM_SAMPLES = 4;

        INDArray trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.getNIn());
        INDArray trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.getNOut());


        // If 0,0 show 0
        trainingInputs.putScalar(new int[]{0, 0}, 0);
        trainingInputs.putScalar(new int[]{0, 1}, 0);
        trainingOutputs.putScalar(new int[]{0, 0}, 0);
        // If 0,1 show 1
        trainingInputs.putScalar(new int[]{1, 0}, 0);
        trainingInputs.putScalar(new int[]{1, 1}, 1);
        trainingOutputs.putScalar(new int[]{1, 0}, 1);
        // If 1,0 show 1
        trainingInputs.putScalar(new int[]{2, 0}, 1);
        trainingInputs.putScalar(new int[]{2, 1}, 0);
        trainingOutputs.putScalar(new int[]{2, 0}, 1);
        // If 1,1 show 0
        trainingInputs.putScalar(new int[]{3, 0}, 1);
        trainingInputs.putScalar(new int[]{3, 1}, 1);
        trainingOutputs.putScalar(new int[]{3, 0}, 0);

        DataSet myData = new DataSet(trainingInputs, trainingOutputs);

        for(int l=0; l<=5; l++) {
            myNetwork.fit(myData);
            Log.d("MainActivity","for loop l = " + l );
        }
        Log.d("MainActivity","output = " + myData );
    }
}

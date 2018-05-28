package org.deeplearning4j.examples.iris_classifier;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.text.DecimalFormat;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    //Global variables to accept the classification results from the background thread.
    double first;
    double second;
    double third;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //onclick to capture the input
        final EditText PL = (EditText) findViewById(R.id.editText);
        final EditText PW = (EditText) findViewById(R.id.editText2);
        final EditText SL = (EditText) findViewById(R.id.editText3);
        final EditText SW = (EditText) findViewById(R.id.editText4);

        Button button = (Button) findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final double pl = Double.parseDouble(PL.getText().toString());
                final double pw = Double.parseDouble(PW.getText().toString());
                final double sl = Double.parseDouble(SL.getText().toString());
                final double sw = Double.parseDouble(SW.getText().toString());

                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute(pl,pw,sl,sw);
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
    }

    private class AsyncTaskRunner extends AsyncTask<Double, Integer, String> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
        }

        // This is our main background thread for the neural net
        @Override
        protected String doInBackground(Double... params) {
            //Get the doubles from params, which is an array so they will be 0,1,2,3
            double pld = params[0];
            double pwd = params[1];
            double sld = params[2];
            double swd = params[3];

            //Write them in the log
            Log.d("myNetwork Output ", "do in background string pl = " + pld);
            Log.d("myNetwork Output ", "do in background string pw = " + pwd);
            Log.d("myNetwork Output ", "do in background string sl = " + sld);
            Log.d("myNetwork Output ", "do in background string sw = " + swd);

            //Create input
            INDArray actualInput = Nd4j.zeros(1,4);
            actualInput.putScalar(new int[]{0,0}, pld);
            actualInput.putScalar(new int[]{0,1}, pwd);
            actualInput.putScalar(new int[]{0,2}, sld);
            actualInput.putScalar(new int[]{0,3}, swd);


            //Convert the iris data into 150x4 matrix
            int row=150;
            int col=4;

            double[][] irisMatrix=new double[row][col];
            int i = 0;
            for(int r=0; r<row; r++){

                for( int c=0; c<col; c++){
                    irisMatrix[r][c]= org.deeplearning.examples.iris_classifier.DataSet.irisData[i++];
                }

            }

            //Check the array by printing it in the log
            System.out.println(Arrays.deepToString(irisMatrix).replace("], ", "]\n"));

            //Now do the same for the label data
            int rowLabel=150;
            int colLabel=3;

            double[][] twodimLabel=new double[rowLabel][colLabel];
            int ii = 0;
            for(int r=0; r<rowLabel; r++){

                for( int c=0; c<colLabel; c++){
                    twodimLabel[r][c]= org.deeplearning.examples.iris_classifier.DataSet.irisData[ii++];
                }

            }

            System.out.println(Arrays.deepToString(twodimLabel).replace("], ", "]\n"));

            //Convert the data matrices into training INDArrays
            INDArray trainingIn = Nd4j.create(irisMatrix);
            INDArray trainingOut = Nd4j.create(twodimLabel);

            //build the layers of the network
            DenseLayer inputLayer = new DenseLayer.Builder()
                    .nIn(4)
                    .nOut(3)
                    .name("Input")
                    .build();

            DenseLayer hiddenLayer = new DenseLayer.Builder()
                    .nIn(3)
                    .nOut(3)
                    .name("Hidden")
                    .build();

            OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(3)
                    .nOut(3)
                    .name("Output")
                    .activation(Activation.SOFTMAX)
                    .build();


            NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
            long seed = 6;
            nncBuilder.seed(seed);
//            nncBuilder.iterations(1000);
//            nncBuilder.learningRate(0.1);
            nncBuilder.activation(Activation.TANH);
            nncBuilder.weightInit(WeightInit.XAVIER);
//            nncBuilder.regularization(true).l2(1e-4);

            NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
            listBuilder.layer(0, inputLayer);
            listBuilder.layer(1, hiddenLayer);
            listBuilder.layer(2, outputLayer);

            listBuilder.backprop(true);

            MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
            myNetwork.init();

            //Create a data set from the INDArrays and train the network
            DataSet myData = new DataSet(trainingIn, trainingOut);

            for(int l=0; l<=1000; l++) {
                myNetwork.fit(myData);
            }

            //Evaluate the input data against the model
            INDArray actualOutput = myNetwork.output(actualInput);
            Log.d("myNetwork Output ", actualOutput.toString());

            //Retrieve the three probabilities
            first = actualOutput.getDouble(0,0);
            second = actualOutput.getDouble(0,1);
            third = actualOutput.getDouble(0,2);

            //Since we used global variables to store the classification results, no need to return
            //a results string. If the results were returned here they would be passed to onPostExecute.
            return "";
        }

        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

            //Update the UI with output
            TextView setosa = (TextView) findViewById(R.id.textView11);
            TextView versicolor = (TextView) findViewById(R.id.textView12);
            TextView virginica = (TextView) findViewById(R.id.textView13);

            //Limit the double to values to two decimals using DecimalFormat
            DecimalFormat df2 = new DecimalFormat(".##");

            setosa.setText(String.valueOf(df2.format(first)));
            versicolor.setText(String.valueOf(df2.format(second)));
            virginica.setText(String.valueOf(df2.format(third)));
        }
    }
}

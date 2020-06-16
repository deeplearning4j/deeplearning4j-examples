package com.example.androidDl4jClassifier;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

interface OnTrainingUpdateEventListener{
    void OnTrainingUpdate(INDArray modelOut);
}

// All the work is done in a worker thread which we encapsulate here.
// The android docs suggest we use AsynchTask, however that has been deprecated.
//  But is unclear how a worker task that needs to update the UI should be implemented.
// Hence we put everything related to the network training and threading in a separate class for now.
class TrainingTask {

    private Thread thread;
    private INDArray xyGrid = null; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.
    private DataSet ds;
    private float[][] data;

    private final int nPointsPerAxis = 100;

    private OnTrainingUpdateEventListener listener;

    TrainingTask() {
        this.thread = null;
        this.data = null;
    }

    void setListener(OnTrainingUpdateEventListener listener) {
        this.listener = listener;
    }

    int getnPointsPerAxis() {
        return nPointsPerAxis;
    }

    float[][] getData() {
        return data;
    }

    INDArray getXyGrid() { // As this is set only once, it can be freely accessed across threads.
        return xyGrid;
    }

    void executeTask(String filename) {
        if (this.thread != null){
            if (this.thread.isAlive()){
                this.thread.interrupt();

                try {
                    this.thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }
        }
        this.thread = new Thread(() -> {

            try {
                calcGrid();
                ReadCSV(filename);
                BuildNN();

            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        thread.start();
    }

    /**
     * this is not the regular way to read a csv file into a data set with DL4j.
     * In this example we have put the data in the assets folder so that the demo works offline.
     */
    private void ReadCSV(String filename) throws IOException {
        InputStreamReader is = new InputStreamReader(MainActivity.getInstance().getApplicationContext().getAssets()
                .open(filename));

        BufferedReader reader = new BufferedReader(is);
        ArrayList<String> rawSVC = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            rawSVC.add(line);
        }

        float[][] tmpData = new float[rawSVC.size()][3];

        int index = 0;
        for(String l : rawSVC){
            String[] values = l.split(",");
            for(int col = 0; col< 3L; col++){
                tmpData[index][col] = Float.parseFloat(values[col]);
            }

            index++;
        }

        normalizeColumn(1, tmpData);
        normalizeColumn(2, tmpData);

        this.data = tmpData;
        INDArray arrData = Nd4j.createFromArray(tmpData);
        INDArray arrFeatures = arrData.getColumns(1, 2);
        INDArray c1 = arrData.getColumns(0);
        INDArray c2 = c1.mul(-1).addi(1.0);
        INDArray labels = Nd4j.hstack(c1, c2);
        ds = new DataSet(arrFeatures, labels);
    }

    /**
     *  Normalize the data in a given column. Normally one would use datavec.
     * @param c column to normalise.
     * @param tmpData java float array.
     */
    private void normalizeColumn(int c, float[][] tmpData){
        int numPoints = tmpData.length;
        float min= tmpData[0][c];
        float max= tmpData[0][c];
        for (float[] tmpDatum : tmpData) {
            float x = tmpDatum[c];
            if (x < min) {
                min = x;
            }
            if (x > max) {
                max = x;
            }
        }

        for (int i=0; i<numPoints; i++){
            float x = tmpData[i][c];
            tmpData[i][c] = (x - min)  / (max - min);
        }
    }


    /**
     * The x,y grid to calculate the NN output. Only needs to be calculated once.
     */
    private void calcGrid(){
        // x coordinates of the pixels for the NN.
        INDArray xPixels = Nd4j.linspace(0, 1.0, nPointsPerAxis, DataType.DOUBLE);
        // y coordinates of the pixels for the NN.
        INDArray yPixels = Nd4j.linspace(0, 1.0, nPointsPerAxis, DataType.DOUBLE);
        //create the mesh:
        INDArray [] mesh = Nd4j.meshgrid(xPixels, yPixels);
        xyGrid = Nd4j.vstack(mesh[0].ravel(), mesh[1].ravel()).transpose();
    }

    private void BuildNN(){
        int seed = 123;
        double learningRate = 0.1;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 50;
        int nEpochs = 2000;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(this.ds);
            INDArray tmp = model.output(this.xyGrid);
            if (listener != null) {
                listener.OnTrainingUpdate(tmp);
            }
            if (thread.isInterrupted()) {
                return;
            }
        }
    }
}

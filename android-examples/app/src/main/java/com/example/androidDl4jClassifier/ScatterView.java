package com.example.androidDl4jClassifier;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.AsyncTask;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
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

public class ScatterView extends View {

    private final Paint redPaint;
    private final Paint greenPaint;
    private final Paint lightGreenPaint;
    private final Paint lightRedPaint;
    private float[][] data;
    private DataSet ds;

    private final int nPointsPerAxis = 100;
    private INDArray xyGrid; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.
    private INDArray modelOut = null;

    public ScatterView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        data = null;
        redPaint = new Paint();
        redPaint.setColor(Color.RED);
        greenPaint = new Paint();
        greenPaint.setColor(Color.GREEN);

        lightGreenPaint = new Paint();
        lightGreenPaint.setColor(Color.rgb(225, 255, 225));
        lightRedPaint = new Paint();
        lightRedPaint.setColor(Color.rgb(255, 153, 152));

        AsyncTask.execute(() -> {
            try {
                calcGrid();
                ReadCSV();
                BuildNN();

            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    public void onDraw(Canvas canvas) {
        int h = this.getHeight();
        int w = this.getWidth();

        //draw the nn predictions:
        if ((modelOut != null) && (null != xyGrid )){
            int halfRectHeight = h / nPointsPerAxis;
            int halfRectWidth = w / nPointsPerAxis;
            int nRows = xyGrid.rows();

            for (int i = 0; i< nRows; i++){
                int  x =  (int)(xyGrid.getFloat(i, 0) * w);
                int y = (int) (xyGrid.getFloat(i, 1)  * h);
                float z = modelOut.getFloat(i, 0);
                Paint p = (z >= 0.5f) ? lightGreenPaint : lightRedPaint;
                canvas.drawRect(x-halfRectWidth, y-halfRectHeight, x+halfRectWidth, y+halfRectHeight, p);
                //  }
            }
        }

        //draw the data set if we have one.
        if (null != data) {

            for (float[] datum : data) {
                int x = (int) (datum[1] * w);
                int y = (int) (datum[2] * h);
                Paint p = (datum[0] == 0.0f) ? redPaint : greenPaint;
                canvas.drawCircle(x, y, 10, p);
            }
        }
    }

    /**
     * this is not the regular way to read a csv file into a data set with DL4j.
     * In this example we have put the data in the assets folder so that the demo works offline.
     */
    private void ReadCSV() throws IOException {
        InputStreamReader is = new InputStreamReader(MainActivity.getInstance().getApplicationContext().getAssets()
                .open("linear_data_train.csv"));

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

    private void BuildNN(){
        int seed = 123;
        double learningRate = 0.005;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;
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
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for(int i = 0; i<nEpochs; i++){
            model.fit(ds);
            INDArray tmp = model.output(xyGrid);

            this.post(() -> {
                this.modelOut =  tmp; // update from within the UI thread.
                this.invalidate(); // have the UI thread redraw at its own convenience.
            });
        }

        Evaluation eval = new Evaluation(numOutputs);
        INDArray features = ds.getFeatures();
        INDArray labels = ds.getLabels();
        INDArray predicted = model.output(features,false);
        eval.eval(labels, predicted);
        System.out.println(eval.stats());

        this.invalidate();
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
}

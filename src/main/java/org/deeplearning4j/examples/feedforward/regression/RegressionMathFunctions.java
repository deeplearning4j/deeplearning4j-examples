package org.deeplearning4j.examples.feedforward.regression;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
 * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
 * predictions as training progresses.
 * A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for regression
 *
 * @author Alex Black
 */
public class RegressionMathFunctions {

    public enum Function {Sin, SinXDivX, SquareWave, TriangleWave, Sawtooth};

    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of iterations per minibatch
    public static final int iterations = 1;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 2000;
    //How frequently should we plot the network output?
    public static final int plotFrequency = 500;
    //Number of data points
    public static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 100;
    //Network learning rate
    public static final double learningRate = 0.01;
    public static final Random rng = new Random(seed);


    public static void main(String[] args){

        //Switch these two options to do different functions with different networks
        Function fn = Function.SinXDivX;
        boolean useSimpleNetwork = false;   //If true: Network with 1 hidden layer of size 20. False: 2 hidden layers of size 50

        //Generate the training data
        INDArray x = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1);
        DataSetIterator iterator = getTrainingData(x,fn,batchSize,rng);

        //Create the network
        MultiLayerNetwork net = new MultiLayerNetwork(getNetworkConfiguration(useSimpleNetwork));
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        //Train the network on the full data set, and evaluate in periodically
        INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
            if((i+1) % plotFrequency == 0) networkPredictions[i/ plotFrequency] = net.output(x, false);
        }

        //Plot the target data and the network predictions
        plot(fn,x,getFunctionValues(x,fn),networkPredictions);
    }

    /**Returns the network configuration
     * @param simple If true: return a simple network (1 hidden layer of size 20). If false: 2 hidden layers of size 50
     */
    public static MultiLayerConfiguration getNetworkConfiguration(boolean simple){
        int numInputs = 1;
        int numOutputs = 1;

        if(simple) {
            int numHiddenNodes = 20;
            return new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(2)
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .weightInit(WeightInit.XAVIER)
                            .activation("identity").weightInit(WeightInit.XAVIER)
                            .nIn(numHiddenNodes).nOut(numOutputs).build())
                    .pretrain(false).backprop(true).build();
        } else {
            int numHiddenNodes = 50;
            return new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(3)
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation("tanh")
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .weightInit(WeightInit.XAVIER)
                            .activation("identity").weightInit(WeightInit.XAVIER)
                            .nIn(numHiddenNodes).nOut(numOutputs).build())
                    .pretrain(false).backprop(true).build();
        }
    }

    //Calculate the function values (sin(x), etc) for a given function and x values
    public static INDArray getFunctionValues(INDArray x, Function function){
        switch (function){
            case Sin:
                return Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
            case SinXDivX:
                return Nd4j.getExecutioner().execAndReturn(new Sin(x.dup())).div(x);
            case SquareWave:
                INDArray sin = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
                return Nd4j.getExecutioner().execAndReturn(new Sign(sin));
            case TriangleWave:
                double period = 6.0;
                double[] xd = x.data().asDouble();
                double[] yd = new double[xd.length];
                for(int i=0; i<xd.length; i++ ){
                    yd[i] = Math.abs(2*(xd[i]/period-Math.floor(xd[i]/period+0.5)));
                }
                return Nd4j.create(yd,new int[]{xd.length,1});  //Column vector
            case Sawtooth:
                double sawtoothPeriod = 4.0;
                double[] xd2 = x.data().asDouble();
                double[] yd2 = new double[xd2.length];
                for(int i=0; i<xd2.length; i++ ){
                    yd2[i] = 2*(xd2[i]/sawtoothPeriod-Math.floor(xd2[i]/sawtoothPeriod+0.5));
                }
                return Nd4j.create(yd2,new int[]{xd2.length,1});  //Column vector
            default:
                throw new RuntimeException();
        }
    }

    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    public static DataSetIterator getTrainingData(INDArray x, Function function, int batchSize, Random rng){
        INDArray y = getFunctionValues(x,function);
        DataSet allData = new DataSet(x,y);

        List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        return new ListDataSetIterator(list,batchSize);
    }

    //Plot the data
    public static void plot(Function function, INDArray x, INDArray y, INDArray... predicted){
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function,      // chart title
                "X",                      // x axis label
                function + "(X)",         // y axis label
                dataSet,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        ChartPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    public static void addSeries(XYSeriesCollection dataSet, INDArray x, INDArray y, String label){
        double[] xd = x.data().asDouble();
        double[] yd = y.data().asDouble();
        XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}

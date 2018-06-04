package org.deeplearning4j.examples.feedforward.mnist

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**A Simple Multi Layered Perceptron (MLP) applied to digit classification for
 * the MNIST Dataset (http://yann.lecun.com/exdb/mnist/).

 * This file builds one input layer and one hidden layer.

 * The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden layer.

 * The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.

 */
object MLPMnistSingleLayerExample {

    private val log = LoggerFactory.getLogger(MLPMnistSingleLayerExample::class.java)

    @Throws(Exception::class)
    @JvmStatic fun main(args: Array<String>) {
        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 128 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15 // number of epochs to perform

        //Get the DataSetIterators:
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

        log.info("Build model....")
        val conf = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong()) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(Nesterovs(0.006, 0.9)) //specify the rate of change of the learning rate.
                .l2(1e-4)
                .list()
                .layer(0, DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build()

        val model = MultiLayerNetwork(conf)
        model.init()
        //print the score with every 1 iteration
        model.setListeners(ScoreIterationListener(1))

        log.info("Train model....")
        for (i in 0..numEpochs - 1) {
            model.fit(mnistTrain)
        }

        log.info("Evaluate model....")
        val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            val next = mnistTest.next()
            val output = model.output(next.features) //get the networks prediction
            eval.eval(next.getLabels(), output) //check the prediction against the true class
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")
    }

}

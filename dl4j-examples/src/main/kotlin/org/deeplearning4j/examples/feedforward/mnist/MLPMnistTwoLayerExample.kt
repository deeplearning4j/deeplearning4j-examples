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
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory


/** A slightly more involved multilayered (MLP) applied to digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).

 * This example uses two input layers and one hidden layer.

 * The first input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 500 output signals to the second layer.

 * The second input layer has input dimension of 500. This layer also uses a rectified linear unit
 * (relu) activation function. The weights for this layer are also initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 100 output signals to the hidden layer.

 * The hidden layer has input dimensions of 100. These are fed from the second input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.

 */
object MLPMnistTwoLayerExample {

    private val log = LoggerFactory.getLogger(MLPMnistTwoLayerExample::class.java)

    @Throws(Exception::class)
    @JvmStatic fun main(args: Array<String>) {
        //number of rows and columns in the input pictures
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val batchSize = 64 // batch size for each epoch
        val rngSeed = 123 // random number seed for reproducibility
        val numEpochs = 15 // number of epochs to perform
        val rate = 0.0015 // learning rate

        //Get the DataSetIterators:
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)


        log.info("Build model....")
        val conf = NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
                .iterations(1)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(rate) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.98) //specify the rate of change of the learning rate.
                .regularization(true).l2(rate * 0.005) // regularize learning model
                .list()
                .layer(0, DenseLayer.Builder() //create the first input layer.
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .build())
                .layer(1, DenseLayer.Builder() //create the second input layer
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(2, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nIn(100)
                        .nOut(outputNum)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build()

        val model = MultiLayerNetwork(conf)
        model.init()
        model.setListeners(ScoreIterationListener(5))  //print the score with every iteration

        log.info("Train model....")
        for (i in 0..numEpochs - 1) {
            log.info("Epoch " + i)
            model.fit(mnistTrain)
        }


        log.info("Evaluate model....")
        val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            val next = mnistTest.next()
            val output = model.output(next.getFeatureMatrix()) //get the networks prediction
            eval.eval(next.getLabels(), output) //check the prediction against the true class
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}

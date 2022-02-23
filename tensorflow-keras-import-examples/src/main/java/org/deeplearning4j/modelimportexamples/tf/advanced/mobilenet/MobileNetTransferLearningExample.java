package org.deeplearning4j.modelimportexamples.tf.advanced.mobilenet;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.NameScope;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.transform.GraphTransformUtil;
import org.nd4j.autodiff.samediff.transform.OpPredicate;
import org.nd4j.autodiff.samediff.transform.SubGraphPredicate;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * This is an example of doing transfer learning by importing a tensorflow model of mobilenet and replacing the last layer.
 *
 * It turns the original ImageNet model into a model for CIFAR 10.
 * This example will only run for one epoch. You will need to run it for many more and tune it to get good results.
 *
 * See {@link ImportMobileNetExample} for the model import example.
 * See MNISTFeedforward class in the samediff-examples project for a basic SameDiff training example.
 * See CustomListenerExample class in the samediff-examples project for an example of to use custom listeners (we use one here to find the shapes of an activation).
 *
 */
@SuppressWarnings("unused") //
public class MobileNetTransferLearningExample {

    public static void main(String[] args) throws Exception {
        File modelFile = ImportMobileNetExample.downloadModel();

        // import the frozen model into a SameDiff instance
        SameDiff sd = SameDiff.importFrozenTF(modelFile);

        System.out.println("\n\n------------------- Initial Graph -------------------");

        System.out.println(sd.summary());

        System.out.println("\n\n");


        // We want to replace the last convolution layer and the output layer with our own ops, so we can fine tune the network
        // These are the MobilenetV2/Logits and MobilenetV2/Predictions sections, respectively.  See the printed summary.


        // Print shapes for each activation.
        // We need to know the shape (especially the channels) of the convolution op's input, so we know what shape to make the weight.
        // We use a custom listener for this, see SameDiffCustomListenerExample

//        INDArray test = new Cifar10DataSetIterator(10).next().getFeatures();
//        test = batchInceptionPreprocessing(test, 224, 224);
//
//        sd.batchOutput()
//            .input("input", test)
//            .output("MobilenetV2/Predictions/Reshape_1")
//            .listeners(new ShapeListener())
//            .execSingle();

        // get info for the last convolution layer (MobilenetV2/Logits).  We want to use an equivalent config.
        Conv2D convOp = (Conv2D) sd.getOpById("MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D");
        System.out.println("Conv config: " + convOp.getConfig());

        /*
        The MobilenetV2/Logits section looks like:
            MobilenetV2/Logits/AvgPool
            MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D
            MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd
            MobilenetV2/Logits/Squeeze

        We want to replace the convolution layer (Conv2D and BiasAdd) with our own, so we can fine tune it.


        The SubGraphPredicate will select a subset of the graph by starting at the root node,
            and then optionally applying SubGraphPredicate's for inputs.
        Those SubGraphPredicate's can also add their inputs, etc.

        The predicate will only accept a subgraph if it passes all the filters.
         */

        // Create a predicate for selecting the BiasAdd and Conv2D ops we want
        SubGraphPredicate pred1 =
            // Select the subgraph with root MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd
            SubGraphPredicate.withRoot(OpPredicate.nameMatches("MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd"))
            // Select (and require) the BiasAdd's 0th input to be MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D
            .withInputSubgraph(0, OpPredicate.nameMatches("MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D"));


        /*
        Replace any subgraphs matching the predicate with our own subgraph
        There will only be one match, but you can use SubGraphPredicate and GraphTransformUtil to replace many occurrences of the same subgraph.

        The number of outputs from the replacement subgraph must match the number of outputs of the subgraph it is replacing.

        Note that the graph isn't actually modified, a copy is made, modified, and then returned.
         */
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd,
            pred1,
            (sd1, subGraph) -> {
                NameScope logits = sd1.withNameScope("Logits/Conv2D");

                // get the output of the AveragePooling op
                SDVariable input = subGraph.inputs().get(1);

                // we know the sizes from using the ShapeListener earlier

                // We know what shape the weight needs to be from the input's channels and the config's kernel height and width.
                // This is why we printed the shapes.
                SDVariable w = sd1.var("W", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT,
                    1, 1, 1280, 10);

                SDVariable b = sd1.var("b", new XavierInitScheme('c', 10 * 1280, 10 * 10), DataType.FLOAT,
                    10);

                // We know the needed config by getting and printing the convolution config earlier
                SDVariable output = sd1.cnn().conv2d(input, w, b, Conv2DConfig.builder()
                    .kH(1).kW(1).paddingMode(PaddingMode.SAME).dataFormat("NHWC").build());

                logits.close();

                return Collections.singletonList(output);
            });

        /*
        The MobilenetV2/Predictions section looks like:
            MobilenetV2/Predictions/Reshape/shape
            MobilenetV2/Predictions/Reshape
            MobilenetV2/Predictions/Softmax
            MobilenetV2/Predictions/Shape
            MobilenetV2/Predictions/Reshape_1

        We want to replace the reshapes (unneeded and the wrong shape) and the softmax (we need a loss function and an output function).
        You could keep the softmax, but there is no reason to.

        We also need to add a labels input.

        Note that this subgraph has no outputs, so neither should the replacement subgraph.
         */

        // create SubGraphPredicate for selecting the MobilenetV2/Predictions ops
        SubGraphPredicate pred2 =
            // Select a subgraph starting with the Reshape_1 op
            SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Reshape_1"))
                // Add the 0th input to the subgraph if it is the specified Softmax Op
                .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Softmax"))
                    // Add the 0th input of the Softmax op to the subgraph, as long as it is the specified Reshape op
                    .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Reshape"))))
                // Add the 1st input to the subgraph if it is the specified Shape Op
                .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Shape")));

        // Replace any subgraphs matching the predicate with our own subgraph
        // There will only be one match, but you can use SubGraphPredicate and GraphTransformUtil to replace many occurrences of the same subgraph
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd,
            pred2,
            (sd1, subGraph) -> {
                // placeholder for labels (needed for training)
                SDVariable labels = sd1.placeHolder("label", DataType.FLOAT, -1, 10);

                NameScope logits = sd1.withNameScope("Predictions");

                // get the output of the preceding squeeze op
                SDVariable input = subGraph.inputs().get(0);

                // dimension 1 by default
                SDVariable outputs = sd1.nn().softmax("Output", input);

                // we need a loss to train on, the tensorflow model doesn't come with one
                SDVariable loss = sd1.loss().softmaxCrossEntropy("Loss", labels, input, null);

                logits.close();

                return Collections.emptyList();
            });


        // Add inception preprocessing to the input (except for resizing, which is done as part of the record reader)
        // Can't do this with GraphTransformUtil as it can't replace variables or re-use ops

        SDVariable input = sd.getVariable("input");

        // change input to channels last (because this is a tensorflow import)
        SDVariable channelsLast = input.permute(0, 2, 3, 1);

        // normalize to 0-1
        SDVariable normalized = channelsLast.div(256);

        // change range to -1 - 1
        SDVariable processed = normalized.sub(0.5).mul(2);

        // The 0th arg was input, replace it with the preprocessed input
        sd.getOpById("MobilenetV2/Conv/Conv2D").replaceArg(0, processed);



        System.out.println("\n\n------------------- Final Graph -------------------");

        System.out.println(sd.summary());

        SDVariable output = sd.getVariable("Predictions/Output");
        SDVariable loss = sd.getVariable("Predictions/Loss");

        // we reshape to the proper size as part of the data set iterator, rather than doing it as part of the inception preprocessing
        INDArray test2 = new Cifar10DataSetIterator(10, new int[]{224, 224}, DataSetType.TRAIN, null, 12345).next().getFeatures();
        System.out.println("CIFAR10 Shape: " + Arrays.toString(test2.shape()));

        // Test run
        sd.batchOutput()
            .input("input", test2)
            .output(output)
//            .listeners(new ShapeListener())
            .execSingle();

        // need to set loss for training
        sd.setLossVariables(loss);

        // the tensorflow model doesn't come with placeholder shapes, but we need to set them for training
        sd.getVariable("input").setShape(new long[]{-1, 3, 224, 224});

        // Training.  See SameDiffMNISTTrainingExample for more details
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
            .l2(1e-4)                               //L2 regularization
            .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
            .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
            .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
            .trainEvaluation(output, 0, new Evaluation())  // add a training evaluation
            .build();

        sd.setTrainingConfig(config);
        sd.addListeners(new ScoreListener(20));

        // again, we reshape to the proper size as part of the data set iterator
        DataSetIterator trainData = new Cifar10DataSetIterator(32, new int[]{224, 224}, DataSetType.TRAIN, null, 12345);

        //Perform fine tuning for 1 epochs.  The pre-trained weights are imported as constants, and thus not trained.
        //Note that this may take a long time, especially if you try to use the CPU backend.
        int numEpochs = 1;
        History hist = sd.fit()
            .train(trainData, numEpochs)
            .exec();
        List<Double> acc = hist.trainingEval(Metric.ACCURACY);

        System.out.println("Accuracy: " + acc);
    }

    /**
     * Used to figure out the shapes of variables, needed to figure out how many channels are going into our added Conv layer
     *
     * See {@link SameDiffCustomListenerExample}
     */
    static class ShapeListener extends BaseListener{

        @Override
        public boolean isActive(Operation operation) {
            return true;
        }

        @Override
        public void activationAvailable(SameDiff sd, At at,
            MultiDataSet batch, SameDiffOp op,
            String varName, INDArray activation) {
            System.out.println(varName + ": \t\t\t" + Arrays.toString(activation.shape()));

            if(varName.endsWith("Shape")){
                System.out.println("Shape value: " + activation);
            }

        }
    }

    /**
     * Does inception preprocessing on a batch of images.  Takes an image with shape [batchSize, c, h, w]
     * and returns an image with shape [batchSize, height, width, c].
     *
     * @param height the height to resize to
     * @param width the width to resize to
     */
    public static INDArray batchInceptionPreprocessing(INDArray img, int height, int width){
        // change to channels-last
        img = img.permute(0, 2, 3, 1);

        // normalize to 0-1
        img = img.div(256);

        // resize
        INDArray preprocessedImage = Nd4j.createUninitialized(img.size(0), height, width, img.size(3));

        DynamicCustomOp op = DynamicCustomOp.builder("resize_bilinear")
            .addInputs(img)
            .addOutputs(preprocessedImage)
            .addIntegerArguments(height, width).build();
        Nd4j.exec(op);

        // finish preprocessing
        preprocessedImage = preprocessedImage.sub(0.5);
        preprocessedImage = preprocessedImage.mul(2);
        return preprocessedImage;
    }
}

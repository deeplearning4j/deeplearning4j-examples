package org.deeplearning4j.examples.samediff.tfimport;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.samediff.training.SameDiffMNISTTrainingExample;
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
import org.nd4j.autodiff.samediff.transform.SubGraph;
import org.nd4j.autodiff.samediff.transform.SubGraphPredicate;
import org.nd4j.autodiff.samediff.transform.SubGraphProcessor;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 * This is an example of doing transfer learning by importing a tensorflow model of mobilenet and replacing the last layer.
 *
 * It turns the original imagenet model into a model for CIFAR 10.
 *
 * See {@link SameDiffTFImportMobileNetExample} for the model import example.
 * See {@link SameDiffMNISTTrainingExample} for the SameDiff training example.
 *
 */
public class SameDiffTransferLearningExample {

    // Used to figure out the shapes of variables, needed to figure out how many channels are going into our added Conv layer
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

    public static void main(String[] args) throws Exception {
        File modelFile = SameDiffTFImportMobileNetExample.downloadModel();

        // import the frozen model into a SameDiff instance
        SameDiff sd = SameDiff.importFrozenTF(modelFile);

        System.out.println("\n\n------------------- Initial Graph -------------------");

        System.out.println(sd.summary());

        System.out.println("\n\n");

        // Print shapes for each activation

//        INDArray test = new Cifar10DataSetIterator(10).next().getFeatures();
//        test = batchInceptionPreprocessing(test, 224, 224);
//
//        sd.batchOutput()
//            .input("input", test)
//            .output("MobilenetV2/Predictions/Reshape_1")
//            .listeners(new ShapeListener())
//            .execSingle();

        // get info for the last convolution layer (MobilenetV2/Logits)
        Conv2D convOp = (Conv2D) sd.getOpById("MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D");
        System.out.println("Conv config: " + convOp.getConfig());

        // replace last convolution layer (MobilenetV2/Logits)
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd,
            SubGraphPredicate.withRoot(OpPredicate.nameMatches("MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd"))
            .withInputSubgraph(0, OpPredicate.nameMatches("MobilenetV2/Logits/Conv2d_1c_1x1/Conv2D")),
            (sd1, subGraph) -> {

                NameScope logits = sd1.withNameScope("Logits/Conv2D");

                // get the output of the AveragePooling op
                SDVariable input = subGraph.inputs().get(1);

                // we know the sizes from using the ShapeListener earlier

                SDVariable w = sd1.var("W", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT,
                    1, 1, 1280, 10);

                SDVariable b = sd1.var("b", new XavierInitScheme('c', 10 * 1280, 10 * 10), DataType.FLOAT,
                    10);

                // We know the needed config by getting and printing the convolution config earlier
                SDVariable output = sd1.cnn().conv2d(input, w, b, Conv2DConfig.builder()
                    .kH(1).kW(1).isSameMode(true).dataFormat("NHWC").build());

                logits.close();

                return Collections.singletonList(output);
            });

        // create SubGraphPredicate for selecting the MobilenetV2/Predictions ops
        SubGraphPredicate graphPred = SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Reshape_1"))
            .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Softmax"))
                .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Reshape"))))
            .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameEquals("MobilenetV2/Predictions/Shape")));

        // replace the MobilenetV2/Predictions with our own softmax and loss
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd,
            graphPred,
            (sd1, subGraph) -> {

                // placeholder for labels (needed for training)
                SDVariable labels = sd1.placeHolder("label", DataType.FLOAT, -1, 10);

                NameScope logits = sd1.withNameScope("Predictions");

                // get the output of the preceding squeeze op
                SDVariable input = subGraph.inputs().get(0);

                // dimension 1 by default
                SDVariable outputs = sd1.nn().softmax("Output", input);

                // we need a loss to train on, the tensorflow model doesn't come with one
                SDVariable loss = sd1.loss().softmaxCrossEntropy("Loss", labels, input);

                logits.close();

                return Collections.emptyList();
            });


        // replace the input with input and inception preprocessing (except for resizing, which is done as part of the record reader)
        // can't do this with GraphTransformUtil as it can't replace variables or re-use ops

        SDVariable input = sd.getVariable("input");

        // change input to channels last (because this is a tensorflow import)
        SDVariable channelsLast = input.permute(0, 2, 3, 1);

        // normalize to 0-1
        SDVariable normalized = channelsLast.div(256);

        // change range to -1 - 1
        SDVariable processed = normalized.sub(0.5).mul(2);

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

        //Perform fine tuning for 20 epochs.  The pre-trained weights are imported as constants, and thus not trained.
        // Note that this may take a long time, especially if you try to use the CPU backend.
        int numEpochs = 20;
        History hist = sd.fit()
            .train(trainData, numEpochs)
            .exec();
        List<Double> acc = hist.trainingEval(Metric.ACCURACY);

        System.out.println("Accuracy: " + acc);
    }
}

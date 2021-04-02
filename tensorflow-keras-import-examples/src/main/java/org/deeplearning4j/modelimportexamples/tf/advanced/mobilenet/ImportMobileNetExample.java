package org.deeplearning4j.modelimportexamples.tf.advanced.mobilenet;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.resources.Downloader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.image.ResizeBilinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Collections;

/**
 * This example shows the ability to import and use Tensorflow models, specifically mobilenet, and use them for inference.
 */
public class ImportMobileNetExample {

    public static void main(String[] args) throws Exception {

        // download and extract a tensorflow frozen model file (usually a .pb file)
        File modelFile = downloadModel();

        //new tensorflow import api
        TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();

        // import the frozen model into a SameDiff instance
        //note: this uses the new tensorflow import api
        SameDiff sd = tensorflowFrameworkImporter.runImport(modelFile.getAbsolutePath(), Collections.emptyMap());

        System.out.println(sd.summary());

        System.out.println("\n\n");

        // get the image from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg for testing
        INDArray testImage = getTestImage();

        // preprocess image with inception preprocessing
        INDArray preprocessedImage = inceptionPreprocessing(testImage, 224, 224);

        // Input and output names are found by looking at sd.summary() (printed earlyer).
        // The input variable is the output of no ops, and the output variable is the input of no ops.

        // Alternatively, you can use sd.outputs() and sd.inputs().

        System.out.println("Input: " + sd.inputs());
        System.out.println("Output: " + sd.outputs());

        // Do inference for a single batch.
        INDArray out = sd.batchOutput()
            .input("input", preprocessedImage)
            .output("MobilenetV2/Predictions/Reshape_1")
            .outputSingle();

        // ignore label 0 (the background label)
        out = out.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 1001));

        // get the readable label for the classes
        String label = new ImageNetLabels().decodePredictions(out);

        System.out.println("Predictions: " + label);

    }


    // download and extract the model file in the ~/dl4j-examples-data directory used by other examples
    static File downloadModel() throws Exception{
        String dataDir = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/tf_resnet");
        String modelFile = FilenameUtils.concat(dataDir, "mobilenet_v2_1.0_224.tgz");

        File frozenFile = new File(FilenameUtils.concat(dataDir, "mobilenet_v2_1.0_224_frozen.pb"));

        if(frozenFile.exists()){
            return frozenFile;
        }

        String MODEL_URL = "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz";
        Downloader.downloadAndExtract("tf_resnet", new URL(MODEL_URL), new File(modelFile), new File(dataDir), "519bba7052fd279c66d2a28dc3f51f46", 5);

        return frozenFile;
    }

    // gets the image we use to test the network.
    // This isn't a single class ImageNet image, so it won't do very well, but it will at least classify it as a dog or a cat.
    private static INDArray getTestImage() throws IOException {
        URL url = new URL("https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image2.jpg?raw=true");
        return new ImageLoader(358, 500, 3).asMatrix(url.openStream());
    }

    /**
     * Does inception preprocessing.  Takes an image with shape [c, h, w]
     * and returns an image with shape [1, height, width, c].
     *
     * Eventually this will be made part of DL4J.
     *
     * @param height the height to resize to
     * @param width the width to resize to
     */
    @SuppressWarnings("SameParameterValue")
    private static INDArray inceptionPreprocessing(INDArray img, int height, int width){
        // add batch dimension
        img = Nd4j.expandDims(img, 0);

        // change to channels-last
        img = img.permute(0, 2, 3, 1);

        // normalize to 0-1
        img = img.div(256);

        // array to store resized image
        INDArray preprocessedImage = Nd4j.createUninitialized(1, height, width, 3);

        Nd4j.exec(new ResizeBilinear(img,preprocessedImage,height,width,false,false));

        // finish preprocessing
        preprocessedImage = preprocessedImage.sub(0.5);
        preprocessedImage = preprocessedImage.mul(2);
        return preprocessedImage;
    }

}

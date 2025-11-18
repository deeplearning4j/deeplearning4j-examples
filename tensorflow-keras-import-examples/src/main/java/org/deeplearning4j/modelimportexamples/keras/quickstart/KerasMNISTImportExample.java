/**
 * KerasMNISTImportExample
 *
 * Demonstrates how to import a trained Keras (.h5) model into DL4J
 * and run inference on the MNIST handwritten digits dataset.
 *
 * Requirements:
 *  - The Keras model must be trained using TensorFlow backend
 *  - The model must be saved using model.save("model.h5") with no custom layers
 *
 * This example loads:
 *   1. MNIST test data using DL4J's built-in dataset iterators
 *   2. A pre-trained Keras model (.h5)
 *   3. Converts the Keras model into a DL4J ComputationGraph
 *   4. Evaluates accuracy on MNIST test set
 */

public class KerasMNISTImportExample {

    public static void main(String[] args) throws Exception {

        // ------------------------------------------------------------
        // 1. Load MNIST Test Data
        // ------------------------------------------------------------
        int batchSize = 128;

        /*
         * MnistDataSetIterator automatically downloads MNIST if needed
         * and loads normalized test images (28x28 grayscale).
         *
         * The iterator returns DataSet objects with:
         *   - features:  [batchSize, 1, 28, 28]
         *   - labels:    one-hot encoded (10 classes)
         */
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        // ------------------------------------------------------------
        // 2. Load Pretrained Keras Model (.h5 file)
        // ------------------------------------------------------------
        String modelPath = "keras_mnist_cnn.h5";   // example file name

        /*
         * KerasModelImport.importKerasModelAndWeights():
         *   - Reads .h5 model architecture + weights
         *   - Converts it to a DL4J ComputationGraph
         *   - Handles TensorFlow-compatible layers automatically
         */
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelPath);

        System.out.println("Imported Keras model summary:");
        System.out.println(model.summary());

        // ------------------------------------------------------------
        // 3. Run Evaluation
        // ------------------------------------------------------------

        /*
         * The Evaluation class computes:
         *   - Accuracy
         *   - Precision / Recall / F1
         *   - Confusion matrix
         */
        Evaluation eval = new Evaluation(10);

        while (mnistTest.hasNext()) {
            DataSet ds = mnistTest.next();
            INDArray output = model.outputSingle(ds.getFeatures());
            eval.eval(ds.getLabels(), output);
        }

        System.out.println("\nModel Evaluation Results:");
        System.out.println(eval.stats());
    }
}

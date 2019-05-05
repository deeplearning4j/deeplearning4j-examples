package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * Multiclass logistic regression.
 * To successfully apply this algorithm the classes must be linearly separable.
 * Unlike Naive Bayes it doesn't assume strong independence on features.
 *
 * This example can be useful to introduce machine learning.
 * Neural nets can be seen as a non-linear extension of this model.
 *
 * @author fvaleri
 */
public class MultiClassLogit {

  private static final Logger log = LoggerFactory.getLogger(MultiClassLogit.class);

  public static void main(String[] args) {
    DataSet irisDataSet = getIrisDataSet();

    //dataset split: 80% training, 20% test
    SplitTestAndTrain trainAndTest = irisDataSet.splitTestAndTrain(120, new Random(42));
    DataSet trainingData = trainAndTest.getTrain();
    DataSet testData = trainAndTest.getTest();

    //hyperparameters
    long maxIterations = 10000;
    double learningRate = 0.01;
    double minLearningRate = 0.0001;

    INDArray model = trainModel(trainingData, maxIterations, learningRate, minLearningRate);
    testModel(testData, model);
  }

  private static DataSet getIrisDataSet() {
    DataSet irisDataSet = null;
    try {

      File irisData = new ClassPathResource("iris.txt").getFile();
      BufferedReader reader = new BufferedReader(new FileReader(irisData));

      List<DataSet> data = reader.lines()
          .filter(l -> !l.trim().isEmpty())
          .map(mapRowToDataSet)
          .collect(Collectors.toList());

      if (reader != null)
        reader.close();

      DataSetIterator iter = new IteratorDataSetIterator(data.iterator(), 150);
      irisDataSet = iter.next();

    } catch (IOException e) {
      log.error("IO error", e);
    }
    return irisDataSet;
  }

  /* Note that applications can use datavec for this type of transform, especially with big datasets. */
  private static Function<String, DataSet> mapRowToDataSet = (String line) -> {
    //sepalLengthCm,sepalWidthCm,petalLengthCm,petalWidthCm,species
    double[] parsedRows = Arrays.stream(line.split(","))
        .mapToDouble(v -> {
          switch (v) {
          case "0":
            return 0.0;
          case "1":
            return 1.0;
          case "2":
            return 2.0;
          default:
            return Double.parseDouble(v);
          }
        }).toArray();
    int columns = parsedRows.length;
    return new DataSet(
        Nd4j.create(Arrays.copyOfRange(parsedRows, 0, columns - 1)),
        Nd4j.create(Arrays.copyOfRange(parsedRows, columns - 1, columns)));
  };

  public static INDArray trainModel(DataSet trainDataSet, long maxIterations, double learningRate,
      double minLearningRate) {
    log.info("Training the model...");
    long start = System.currentTimeMillis();
    INDArray trainFeatures = prependConstant(trainDataSet);
    INDArray trainLabels = trainDataSet.getLabels();

    //to work with multiple classes we build a model for each class that can predict
    //if an example belongs to it, then we'll pick the class with the highest probability
    //to give the final class prediction
    INDArray class1Labels = getClassLabels(trainLabels, 0);
    INDArray class2Labels = getClassLabels(trainLabels, 1);
    INDArray class3Labels = getClassLabels(trainLabels, 2);
    INDArray model1 = training(trainFeatures, class1Labels, maxIterations, learningRate, minLearningRate);
    INDArray model2 = training(trainFeatures, class2Labels, maxIterations, learningRate, minLearningRate);
    INDArray model3 = training(trainFeatures, class3Labels, maxIterations, learningRate, minLearningRate);

    INDArray finalModel = Nd4j.hstack(model1, model2, model3);
    log.debug("Model's parameters:\n{}", finalModel);
    log.info("Took {} ms", (System.currentTimeMillis() - start));
    return finalModel;
  }

  public static void testModel(DataSet testDataSet, INDArray params) {
    log.info("Testing the model...");
    INDArray testFeatures = prependConstant(testDataSet);
    INDArray testLabels = testDataSet.getLabels();
    INDArray predictedLabels = predictLabels(testFeatures, params);

    long numOfSamples = testLabels.size(0);
    double correctSamples = countCorrectPred(testLabels, predictedLabels);
    double accuracy = correctSamples / numOfSamples;
    log.info("Correct samples: {}/{}", (int) correctSamples, numOfSamples);
    log.info("Accuracy: {}", accuracy);
  }

  /**
   * Prepend the linear regression's constant term (ones column).
   * This avoids the case in which all the features are zero thar produce
   * a zero prediction, which means 50% probability (i.e. max uncertainty).
   *
   * @param dataset dataset
   * @return features
   */
  public static INDArray prependConstant(DataSet dataset) {
    INDArray features = Nd4j.hstack(
        Nd4j.ones(dataset.getFeatures().size(0), 1),
        dataset.getFeatures());
    return features;
  }

  /**
   * Logistic function.
   *
   * Computes a number between 0 and 1 for each element.
   * Note that ND4J comes with its own sigmoid function.
   *
   * @param y input values
   * @return probabilities
   */
  private static INDArray sigmoid(INDArray y) {
    y = y.mul(-1.0);
    y = exp(y, false);
    y = y.add(1.0);
    y = y.rdiv(1.0);
    return y;
  }

  /**
   * Binary logistic regression.
   *
   * Computes the probability that one example is a certain type of flower.
   * Can compute a batch of examples at a time, i.e. a matrix with samples
   * as rows and columns as features (this is normally done by DL4J internals).
   *
   * @param x features
   * @param p parameters
   * @return class probability
   */
  private static INDArray predict(INDArray x, INDArray p) {
    INDArray y = x.mmul(p); //linear regression
    return sigmoid(y);
  }

  /**
   * Gradient function.
   *
   * Compute the gradient of the cost function and
   * how much error each parameter puts into the result.
   *
   * @param x features
   * @param y labels
   * @param p parameters
   * @return paramters gradients
   */
  private static INDArray gradient(INDArray x, INDArray y, INDArray p) {
    long m = x.size(0); //number of examples
    INDArray pred = predict(x, p);
    INDArray diff = pred.dup().sub(y); //diff between predicted and actual class
    return x.dup()
        .transpose()
        .mmul(diff)
        .mul(1.0 / m);
  }

  /**
   * Training algorithm.
   *
   * Gradient descent optimization.
   * The logistic cost function (or max-entropy) is convex,
   * and thus we are guaranteed to find the global minimum.
   *
   * @param x input features
   * @param y output classes
   * @param maxIterations max number of learning iterations
   * @param learningRate how fast parameters change
   * @param minLearningRate lower bound on learning rate
   * @return optimal parameters
   */
  private static INDArray training(INDArray x, INDArray y, long maxIterations, double learningRate,
      double minLearningRate) {
    Nd4j.getRandom().setSeed(1234);
    INDArray params = Nd4j.rand((int)x.size(1), 1); //random guess

    INDArray newParams = params.dup();
    INDArray optimalParams = params.dup();

    for (int i = 0; i < maxIterations; i++) {
      INDArray gradients = gradient(x, y, params);
      gradients = gradients.mul(learningRate);
      newParams = params.sub(gradients);

      if (hasConverged(params, newParams, minLearningRate)) {
        log.debug("Completed iterations: {}", i + 1);
        break;
      }
      params = newParams;
    }

    optimalParams = newParams;
    return optimalParams;
  }

  private static boolean hasConverged(INDArray oldParams, INDArray newParams, double threshold) {
    double diffSum = abs(oldParams.sub(newParams)).sumNumber().doubleValue();
    return diffSum / oldParams.size(0) < threshold;
  }

  private static INDArray getClassLabels(INDArray labels, double label) {
    INDArray binaryLabels = labels.dup();
    for (int i = 0; i < binaryLabels.rows(); i++) {
      double v = binaryLabels.getDouble(i);
      if (v == label)
        binaryLabels.putScalar(i, 1.0);
      else
        binaryLabels.putScalar(i, 0.0);
    }
    return binaryLabels;
  }

  /**
   * Label prediction.
   *
   * Maximum a posteriori probability estimate.
   * For each example: run N independent predictions (one for each class)
   * and return the one with the highest value (argmax).
   *
   * @param features input features
   * @param params model's parameters
   * @return predicted labels
   */
  private static INDArray predictLabels(INDArray features, INDArray params) {
    INDArray predictions = features.mmul(params).argMax(1);
    return predictions;
  }

  private static double countCorrectPred(INDArray labels, INDArray predictions) {
    double counter = 0;
    for (int i = 0; i < labels.size(0); i++) {
      if (labels.getDouble(new int[] { i }) == predictions.getDouble(new int[] { i })) {
        counter++;
      }
    }
    return counter;
  }

}

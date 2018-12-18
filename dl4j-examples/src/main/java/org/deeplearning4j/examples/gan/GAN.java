package org.deeplearning4j.examples.gan;


import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Implementation of vanilla Generative Adversarial Networks as introduced in https://arxiv.org/pdf/1406.2661.pdf.
 *
 * A DL4J GAN is built from two networks: a generator and a discriminator and will build a third network,
 * the GAN network, from the first two. The GAN networks essentially chains the discriminator to the generator, i.e.
 * first images are generated, then the discriminator tries to decide if they are real or fake.
 *
 * In the training step of a GAN, we start off by training the discriminator. For each batch of real images, we generate
 * a batch of fake images using the generator. All real images are labeled "1", all fake ones "0". With this data the
 * discriminator gets better at distinguishing the two. Next, we create so called adversarial or "misleading" examples,
 * that is we let the generator create images and we falsely label them as real ("1"). Now it's time to train the GAN with
 * this batch of data. To do so, we freeze the weights of the discriminator and only update the generator. This way,
 * the generator will get better at creating adversarial example, thereby fooling the discriminator.
 *
 * These two training steps get repeated all over. The discriminator will slowly get better at separating real from fake,
 * and the generator will get better at creating better fake images. This leads to strong discriminators, but first and
 * foremost to really powerful generators. If you train a GAN on MNIST handwritten digits, for instance, the GAN will
 * learn to generate realistic handwritten data.
 *
 * @author Max Pumperla
 */
public class GAN {

    protected MultiLayerNetwork generator;
    protected MultiLayerNetwork discriminator;
    protected MultiLayerNetwork gan;
    protected int latentDim;

    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected OptimizationAlgorithm optimizer;
    protected GradientNormalization gradientNormalizer;
    protected double gradientNormalizationThreshold;
    protected WorkspaceMode trainingWorkSpaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;
    protected CacheMode cacheMode;
    protected long seed;


    public GAN(Builder builder) {
        this.generator = builder.generator;
        this.discriminator = builder.discriminator;
        this.latentDim = builder.latentDimension;
        this.updater = builder.iUpdater;
        this.biasUpdater = builder.biasUpdater;
        this.optimizer = builder.optimizationAlgo;
        this.gradientNormalizer = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.trainingWorkSpaceMode = builder.trainingWorkspaceMode;
        this.inferenceWorkspaceMode = builder.inferenceWorkspaceMode;
        this.cacheMode = builder.cacheMode;
        this.seed = builder.seed;

        defineGan();
    }

    public MultiLayerNetwork getGenerator() {
        return generator;
    }

    public MultiLayerNetwork getDiscriminator() {
        return discriminator;
    }

    public Evaluation evaluateGan(DataSetIterator data) {
        return gan.evaluate(data);
    }

    public Evaluation evaluateGan(DataSetIterator data, List<String> labelsList) {
        return gan.evaluate(data, labelsList);
    }


    public void setGeneratorListeners(BaseTrainingListener[] listeners) {
        generator.setListeners(listeners);
    }

    public void setDiscriminatorListeners(BaseTrainingListener[] listeners) {
        discriminator.setListeners(listeners);
    }

    public void setGanListeners(BaseTrainingListener[] listeners) {
        gan.setListeners(listeners);
    }

    public void fit(DataSetIterator realData, int numEpochs) {

        int batchSize;

        for(int i=0; i < numEpochs; i++) {
            while (realData.hasNext()) {

                // Get real images as features
                INDArray realImages = realData.next().getFeatures();
                batchSize = (int) realImages.shape()[0];

                // Sample from latent space and let the generate create fake images.
                INDArray randomLatentData = Nd4j.rand(new int[]{batchSize, latentDim});
                INDArray fakeImages = generator.output(randomLatentData);

                // Real images are marked as "1", fake images at "0".
                DataSet realSet = new DataSet(realImages, Nd4j.zeros(batchSize, 1));
                DataSet fakeSet = new DataSet(fakeImages, Nd4j.ones(batchSize, 1));

                // Fit the discriminator on a combined batch of real and fake images.
                DataSet combined = DataSet.merge(Arrays.asList(realSet, fakeSet));
                discriminator.fit(combined);

                // Update the discriminator in the GAN network
                updateGanWithDiscriminator();

                // Generate a new set of adversarial examples and try to mislead the GAN by labeling them
                // as real images
                INDArray adversarialExamples = Nd4j.rand(new int[]{batchSize, latentDim});
                INDArray misleadingLabels = Nd4j.ones(batchSize, 1);
                DataSet adversarialSet = new DataSet(adversarialExamples, misleadingLabels);

                // Fit the GAN on the adversarial set, trying to fool the discriminator by generating
                // better fake images.
                gan.fit(adversarialSet);

                // Copy the GANs generator part to "generator".
                updateGeneratorFromGan();


            }
            realData.reset();
        }
    }

    private void defineGan() {
        Layer[] genLayers = generator.getLayers();
        int numGenLayers = genLayers.length;
        Layer[] disLayers = discriminator.getLayers();
        Layer[] layers = ArrayUtils.addAll(genLayers, disLayers);
        MultiLayerConfiguration genConf = generator.getLayerWiseConfigurations();
        MultiLayerConfiguration disConf = discriminator.getLayerWiseConfigurations();
        org.deeplearning4j.nn.conf.layers.Layer[] confLayers = new org.deeplearning4j.nn.conf.layers.Layer[layers.length];

        Map<Integer, InputPreProcessor> preProcessors = new HashMap<>();
        for (int i = 0; i < layers.length; i++) {
            confLayers[i] = layers[i].conf().getLayer();
            if (i < numGenLayers) {
                preProcessors.put(i, genConf.getInputPreProcess(i));
            } else {
                preProcessors.put(i, disConf.getInputPreProcess(i - numGenLayers));
            }
        }

        MultiLayerConfiguration ganConf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(updater)
            .biasUpdater(biasUpdater)
            .optimizationAlgo(optimizer)
            .gradientNormalization(gradientNormalizer)
            .gradientNormalizationThreshold(gradientNormalizationThreshold)
            .activation(Activation.IDENTITY)
            .trainingWorkspaceMode(trainingWorkSpaceMode)
            .inferenceWorkspaceMode(inferenceWorkspaceMode)
            .cacheMode(cacheMode)
            .list(confLayers)
            .inputPreProcessors(preProcessors)
            .build();

        gan = new MultiLayerNetwork(ganConf);
        gan.init();

        // Set learning rate of discriminator part of gan to zero.
        for (int i = genLayers.length; i < layers.length; i++) {
            gan.setLearningRate(i, 0.0);
        }

        // we lose proper init here, need to copy weights after
        copyParamsToGan();
    }

    private  void copyParamsToGan() {
        int genLayerCount = generator.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++) {
            if (i < genLayerCount) {
                generator.getLayer(i).setParams(gan.getLayer(i).params());
            } else {
                discriminator.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }

    /**
     * After the GAN has been trained on misleading images, we update the generator the
     * new weights (we don't have to update the discriminator, as it is frozen in the GAN).
     */
    private void updateGeneratorFromGan() {
        for (int i = 0; i < generator.getLayers().length; i++) {
            generator.getLayer(i).setParams(gan.getLayer(i).params());
        }
    }

    /**
     * After the discriminator has been trained, we update the respective parts of the GAN network
     * as well.
     */
    private  void updateGanWithDiscriminator() {
        int genLayerCount = generator.getLayers().length;
        for (int i = genLayerCount; i < gan.getLayers().length; i++) {
            gan.getLayer(i).setParams(discriminator.getLayer(i - genLayerCount).params());
        }
    }

    /**
     * GAN builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     */
    public static class Builder implements Cloneable {
        protected MultiLayerNetwork generator;
        protected MultiLayerNetwork discriminator;
        protected int latentDimension;

        protected IUpdater iUpdater = new Sgd();
        protected IUpdater biasUpdater = null;
        protected long seed = System.currentTimeMillis();
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
        protected CacheMode cacheMode = CacheMode.NONE;


        public Builder() { }


        /**
         * Set the (fake) image generator of the GAN.
         *
         * @param generator MultilayerNetwork
         * @return Builder
         */
        public GAN.Builder generator(MultiLayerNetwork generator) {
            this.generator = generator;
            return this;
        }

        /**
         * Set the image discriminator of the GAN.
         *
         * @param discriminator MultilayerNetwork
         * @return Builder
         */
        public GAN.Builder discriminator(MultiLayerNetwork discriminator) {
            this.discriminator = discriminator;
            return this;
        }

        /**
         * Set the latent dimension, i.e. the input vector space dimension of the generator.
         *
         * @param latentDimension latent space input dimension.
         * @return Builder
         */
        public GAN.Builder latentDimension(int latentDimension) {
            this.latentDimension = latentDimension;
            return this;
        }


        /**
         * Random number generator seed. Used for reproducibility between runs
         */
        public GAN.Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public GAN.Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }


        /**
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use
         */
        public GAN.Builder updater(IUpdater updater) {
            this.iUpdater = updater;
            return this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use for bias parameters
         */
        public GAN.Builder biasUpdater(IUpdater updater){
            this.biasUpdater = updater;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public GAN.Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public GAN.Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        public GAN build() {
            return new GAN(this);
        }

    }

}

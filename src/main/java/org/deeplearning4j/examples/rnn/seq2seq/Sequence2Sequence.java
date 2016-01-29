package org.deeplearning4j.examples.rnn.seq2seq;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.FilteredSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 28/01/2016.
 */
public class Sequence2Sequence {

    //    public static String enPath = "D:\\Data\\training-giga-fren\\giga-fren.release2.en\\giga-fren.release2.en";
//    public static String frPath = "D:\\Data\\training-giga-fren\\giga-fren-release2.fr";
    public static String enPath = "D:\\Data\\dev\\newstest2010.en";
    public static String frPath = "D:\\Data\\dev\\newstest2010.fr";

    public static final int VOCAB_SIZE = 4096;

    public static void main(String[] args) throws Exception {

        SequenceToSequenceIterator train = getTrainData(32, VOCAB_SIZE);
        AbstractCache<VocabWord> enCache = train.getVocabCache1();
        AbstractCache<VocabWord> frCache = train.getVocabCache2();


//        MultiDataSet mds = train.next();
//        INDArray en = mds.getFeatures(0);
//        INDArray fr = mds.getFeatures(1);
//        INDArray out = mds.getLabels(0);
//        for (int i = 0; i < en.size(2); i++) {
//            System.out.println(en.getDouble(0, 0, i) + "\t" + enCache.wordAtIndex((int) en.getDouble(0, 0, i)));
//        }
//
//        System.out.println("-----------------");
//        for (int i = 0; i < fr.size(2); i++) {
//            int idx = (int) fr.getDouble(0, 0, i);
//            String word = (idx == VOCAB_SIZE ? "<GO>" : frCache.wordAtIndex(idx));
//            System.out.println(fr.getDouble(0, 0, i) + "\t" + word);
//        }
//
//        INDArray argmax = Nd4j.argMax(out,1);
//        System.out.println(argmax);

        //Next: set up network architecture, get training working

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .regularization(true).l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.0001)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("inEn", "inFr")
                .setInputTypes(InputType.recurrent(), InputType.recurrent())
                .addLayer("embeddingEn", new EmbeddingLayer.Builder().nIn(VOCAB_SIZE+1).nOut(128).activation("identity").build(),"inEn")
                .addLayer("encoder", new GravesLSTM.Builder().nIn(128).nOut(256).activation("softsign").build(),"embeddingEn")
                .addVertex("lastTimeStep", new LastTimeStepVertex("inEn"),"encoder")
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("inFr"), "lastTimeStep")
                .addLayer("embeddingFr", new EmbeddingLayer.Builder().nIn(VOCAB_SIZE+1).nOut(128).activation("identity").build(),"inFr")
                .addVertex("embeddingFrSeq", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "embeddingFr")
                .addLayer("decoder", new GravesLSTM.Builder().nIn(128 + 256).nOut(256).activation("softsign").build(), "embeddingFrSeq", "duplicateTimeStep")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(256).nOut(VOCAB_SIZE+1).activation("softmax").build(), "decoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();

//        Map<String,INDArray> mapTemp = net.paramTable();
//        System.out.println(mapTemp.keySet());
        System.out.println("Number of parameters:");
        for( int i=0; i<net.getNumLayers(); i++ ){
            System.out.println(i + "\t" + net.getLayer(i).conf().getLayer().getLayerName() + "\t" + net.getLayer(i).numParams());
        }

        net.setListeners(new ScoreIterationListener(1), new HistogramIterationListener(1));

        net.fit(train);

        System.out.println("DONE");
    }

    private static SequenceToSequenceIterator getTrainData(int batchSize, int vocabSize) throws Exception {

        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> en = getIterator(true, vocabSize);
        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> fr = getIterator(false, vocabSize);

        return new SequenceToSequenceIterator(batchSize, vocabSize, en.getFirst(), fr.getFirst(), en.getSecond(), fr.getSecond());
    }

    private static Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> getIterator(boolean english, int maxVocabSize) throws Exception {
        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
        File file;
        if (english) file = new File(enPath);
        else file = new File(frPath);
        BasicLineIterator lineIter = new BasicLineIterator(file);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(lineIter)
                .tokenizerFactory(t)
                .build();

        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(transformer)
                .build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 1)
                .setTargetVocabCache(vocabCache)
                .setEntriesLimit(maxVocabSize)
                .build();
        constructor.buildJointVocabulary(false, true);

        sequenceIterator.reset();
        SequenceIterator<VocabWord> filteredIterator = new FilteredSequenceIterator<>(sequenceIterator, vocabCache);
        return new Pair<>(filteredIterator, vocabCache);
    }

    private static class SequenceToSequenceIterator implements MultiDataSetIterator {

        private int batchSize;
        private int vocabSize;
        private SequenceIterator<VocabWord> iter1;
        private SequenceIterator<VocabWord> iter2;
        private AbstractCache<VocabWord> vocabCache1;
        private AbstractCache<VocabWord> vocabCache2;

        public SequenceToSequenceIterator(int batchSize, int vocabSize, SequenceIterator<VocabWord> iter1, SequenceIterator<VocabWord> iter2,
                                          AbstractCache<VocabWord> vocabCache1, AbstractCache<VocabWord> vocabCache2) {
            this.batchSize = batchSize;
            this.vocabSize = vocabSize;
            this.iter1 = iter1;
            this.iter2 = iter2;
            this.vocabCache1 = vocabCache1;
            this.vocabCache2 = vocabCache2;
        }

        public AbstractCache<VocabWord> getVocabCache1() {
            return vocabCache1;
        }

        public AbstractCache<VocabWord> getVocabCache2() {
            return vocabCache2;
        }

        @Override
        public MultiDataSet next(int num) {
            List<List<VocabWord>> iter1List = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && iter1.hasMoreSequences(); i++) {
                iter1List.add(iter1.nextSequence().getElements());
            }

            List<List<VocabWord>> iter2List = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && iter2.hasMoreSequences(); i++) {
                iter2List.add(iter2.nextSequence().getElements());
            }

            int numExamples = Math.min(iter1List.size(), iter2List.size());
            int in1Length = 0;
            int in2Length = 0;

            for (int i = 0; i < numExamples; i++) {
                in1Length = Math.max(in1Length, iter1List.get(i).size());
            }
            for (int i = 0; i < numExamples; i++) {
                in2Length = Math.max(in2Length, iter2List.get(i).size());
            }

            //2 inputs here, and 1 output
            //First input: a sequence of word indexes for iter1 words
            //Second input: a sequence of word indexes for iter2 words (shifted by 1, with an additional 'go' class as first time step)
            //Output: sequence of word indexes for iter2 words (with an additional 'stop' class as the last time step)
            //Also need mask arrays

            INDArray in1 = Nd4j.create(numExamples, 1, in1Length);
            INDArray in1Mask = Nd4j.ones(numExamples, in1Length);
            int[] arr1 = new int[3];
            int[] arr2 = new int[2];
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter1List.get(i);
                arr1[0] = i;
                arr2[0] = i;

                int j = 0;
                for (VocabWord vw : list) {
                    arr1[2] = j++;
                    in1.putScalar(arr1, vw.getIndex());
                }
                for (; j < in1Length; j++) {
                    arr2[1] = j;
                    in1Mask.putScalar(arr2, 0.0);
                }
            }

            INDArray in2 = Nd4j.create(numExamples, 1, in2Length + 1);
            INDArray in2Mask = Nd4j.ones(numExamples, in2Length + 1);
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter2List.get(i);
                arr1[0] = i;
                arr2[0] = i;

                //First time step: "go" index = vocab size (as word indexes are 0 to vocabSize-1 inclusive)
                arr1[2] = 0;
                in2.putScalar(arr1, vocabSize);

                int j = 1;
                for (VocabWord vw : list) {
                    arr1[2] = j++;
                    in2.putScalar(arr1, vw.getIndex());
                }
                for (; j < in1Length; j++) {
                    arr2[1] = j;
                    in2Mask.putScalar(arr2, 0.0);
                }
            }

            //Using a one-hot representation here. Can't use indexes line for input
            INDArray out = Nd4j.create(numExamples, vocabSize + 1, in2Length + 1);
            INDArray outMask = Nd4j.ones(numExamples, in2Length + 1);

            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter2List.get(i);
                arr1[0] = i;
                arr2[0] = i;

                int j = 0;
                for (VocabWord vw : list) {
                    arr1[1] = vw.getIndex();
                    arr1[2] = j++;
                    out.putScalar(arr1, 1.0);
                }

                //Last time step: "stop" index = vocab size (as word indexes are 0 to vocabSize-1 inclusive)
                arr1[1] = vocabSize;
                arr1[2] = j++;
                out.putScalar(arr1, 1.0);

                for (; j < in1Length; j++) {
                    arr2[1] = j;
                    outMask.putScalar(arr2, 0.0);
                }
            }

            INDArray[] inputs = new INDArray[]{in1, in2};
            INDArray[] inputMasks = new INDArray[]{in1Mask, in2Mask};
            INDArray[] labels = new INDArray[]{out};
            INDArray[] labelMasks = new INDArray[]{outMask};

            return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

        }

        @Override
        public void reset() {
            iter1.reset();
            iter2.reset();
        }

        @Override
        public boolean hasNext() {
            return iter1.hasMoreSequences() && iter2.hasMoreSequences();
        }

        @Override
        public MultiDataSet next() {
            return next(batchSize);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported");
        }
    }
}

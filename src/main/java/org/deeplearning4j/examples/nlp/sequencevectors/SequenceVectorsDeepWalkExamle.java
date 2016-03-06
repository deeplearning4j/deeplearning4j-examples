package org.deeplearning4j.examples.nlp.sequencevectors;

import org.deeplearning4j.examples.nlp.sequencevectors.classes.Blogger;
import org.deeplearning4j.examples.nlp.sequencevectors.classes.GraphBuilder;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.GraphTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;

/**
 *
 * * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.9 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class SequenceVectorsDeepWalkExamle {

    public static void main(String[] args) throws Exception {

        AbstractCache<Blogger> vocabCache = new AbstractCache.Builder<Blogger>().build();

        Graph<Blogger, Void> graph = GraphBuilder.buildGraph();

        GraphTransformer<Blogger> graphTransformer = new GraphTransformer.Builder<Blogger>(graph)
                .setWalkMode(WalkMode.RANDOM)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(10)
                .setWalkDirection(WalkDirection.RANDOM)
                .shuffleOnReset(true)
                .build();


        AbstractSequenceIterator<Blogger> sequenceIterator = new AbstractSequenceIterator.Builder<Blogger>(graphTransformer).build();


        WeightLookupTable<Blogger> lookupTable = new InMemoryLookupTable.Builder<Blogger>()
                .lr(0.025)
                .vectorLength(150)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();



        SequenceVectors<Blogger> vectors = new SequenceVectors.Builder<Blogger>(new VectorsConfiguration())
                // minimum number of occurencies for each element in training corpus. All elements below this value will be ignored
                // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
                .minWordFrequency(5)

                // WeightLookupTable
                .lookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .vocabCache(vocabCache)

                // batchSize is the number of sequences being processed by 1 thread at once
                // this value actually matters if you have iterations > 1
                .batchSize(250)

                // number of iterations over batch
                .iterations(1)

                // number of iterations over whole training corpus
                .epochs(3)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)


                /*
                    These two methods define our training goals. At least one goal should be set to TRUE.
                 */
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                /*
                    Specifies elements learning algorithms. SkipGram, for example.
                 */
                .elementsLearningAlgorithm(new SkipGram<Blogger>())

                .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();
    }
}

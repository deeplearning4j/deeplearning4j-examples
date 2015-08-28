package org.deeplearning4j.examples.tsne;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/20/14.
 *
 * Dimensionality reduction for high-dimension datasets
 */
public class TSNEStandardExample {

    private static Logger log = LoggerFactory.getLogger(TSNEStandardExample.class);

    public static void main(String[] args) throws Exception  {
        int iterations = 1000;
        List<String> cacheList = new ArrayList<>();

        log.info("Load & Vectorize data....");
        File wordFile = new ClassPathResource("words.txt").getFile();
        Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(wordFile);
        VocabCache cache = vectors.getSecond();
        INDArray weights = vectors.getFirst().getSyn0();

        for(int i = 0; i < cache.numWords(); i++)
            cacheList.add(cache.wordAtIndex(i));

        log.info("Build model....");
        Tsne tsne = new Tsne.Builder()
                .setMaxIter(iterations)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
                .usePca(false)
                .build();

        log.info("Store TSNE Coordinates for Plotting....");
        String outputFile = "target/archive-tmp/tsne-standard-coords.csv";
        (new File(outputFile)).getParentFile().mkdirs();
        tsne.plot(weights,2,cacheList,outputFile);
    }



}

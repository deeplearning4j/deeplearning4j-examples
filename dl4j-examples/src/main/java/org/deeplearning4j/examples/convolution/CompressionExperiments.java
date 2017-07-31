package org.deeplearning4j.examples.convolution;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CompressionExperiments {

    protected static void query(@NonNull Word2Vec model, @NonNull String word) {

        log.info("Fetching nearest words for '{}' ---------------", word);
        Collection<String> result = model.wordsNearest(word, 10);
        for (String res:result) {
            double sim = model.similarity(word, res);
            log.info("{} -> {} : {}%", word, res, sim);
        }
        log.info("------------------");
    }

    protected static void testOutput(Word2Vec model) {
        query(model, "water");
        query(model, "fire");
        query(model, "car");
    }

    public static void main(String[] args) throws Exception {

        Word2Vec model = WordVectorSerializer.loadGoogleModel(new File("/home/raver119/develop/GoogleNews-vectors-negative300.bin.gz"), true);
        testOutput(model);
/*
        INDArray initial = model.getLookupTable().getWeights();
//        INDArray intact = initial.dup(initial.ordering());

        double initialMean = initial.meanNumber().doubleValue();
        double initialDev =  initial.stdNumber().doubleValue();

        INDArray amax = initial.amax(1);

        INDArray logmax = Transforms.log(amax,2.0, true);

//        log.info("LogMax: {}", logmax);

        INDArray scale = Transforms.pow(logmax.subi(14), 2);
        initial.muliColumnVector(scale.rdiv(1.0));

        Nd4j.getCompressor().setDefaultCompression("FLOAT16");
        INDArray compressed = Nd4j.getCompressor().compress(initial);
        INDArray decompressed = Nd4j.getCompressor().decompress(compressed);

        model.getLookupTable().getWeights().assign(decompressed);


        testOutput(model);
        */
    }
}

package org.deeplearning4j.patent.utils;

import net.openhft.chronicle.map.ChronicleMap;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.StaticWord2Vec;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.AbstractStorage;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Field;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class provides word vectors to all Spark workers, where the vectors are shared for all workers.
 * It works by copying vectors from a remote source to a local temorary folder, before loading them.
 * A better implementation would cache the file locally and only re-download if not found.
 *
 * It is designed to be used the context of Spark, as a workaround for Spark's 2GB broadcast limit
 * (work vectors are much larger than this)
 *
 * @author Alex Black
 */
public class WordVectorProvider {
    private static final Logger log = LoggerFactory.getLogger(WordVectorProvider.class);

    public static Map<Integer,INDArray> map;
    public static WordVectors wordVectors;

    public static synchronized WordVectors getWordVectors(Configuration config, String path) throws IOException {
        if (wordVectors != null) {
            return wordVectors;
        }

        //Assume file in azure, copy to local before loading
        String baseName = FilenameUtils.getBaseName(path);
        String ext = FilenameUtils.getExtension(path);
        File tempFile = Files.createTempFile(baseName, "." + ext).toFile();

        URI u;
        try{
            u = new URI(path);
        } catch (URISyntaxException e){
            throw new RuntimeException(e);
        }

        String scheme = u.getScheme();
        if(scheme == null || scheme.length() <= 1){
            throw new IllegalStateException("Could not determine URI scheme for path \"" + path + "\". For file paths, prefix path with \"file:///\"," +
                " for Azure prefix with wasbs://");
        }
        FileSystem fs = FileSystem.get(u, config);

        try {
            log.info("Copying word vectors");
            long start = System.currentTimeMillis();
            try (InputStream in = new BufferedInputStream(fs.open(new Path(u))); OutputStream os = new BufferedOutputStream(new FileOutputStream(tempFile))) {
                IOUtils.copy(in, os);
            }
            log.info("Finished copying word vectors - duration {} sec", (System.currentTimeMillis()-start)/1000);

            log.info("Loading word vectors");
            start = System.currentTimeMillis();
            wordVectors = WordVectorSerializer.loadStaticModel(tempFile);
            log.info("Finished loading word vectors - duration {} sec", (System.currentTimeMillis()-start)/1000);
            return wordVectors;
        } finally {
            tempFile.delete();
        }
    }

    private static AbstractStorage<Integer> storage;
    public static Map<Integer,INDArray> getLookupMap(Configuration configuration, String path) throws Exception{
        if(map != null){
            return map;
        }

        synchronized (WordVectorProvider.class) {
            if(map != null){
                return map;
            }
            WordVectors wv = getWordVectors(configuration, path);
            try {
                StaticWord2Vec sw2v = (StaticWord2Vec) wv;
                Field field = StaticWord2Vec.class.getDeclaredField("storage");
                field.setAccessible(true);
                storage = (AbstractStorage<Integer>) field.get(sw2v);
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }

            ChronicleMap<Integer, INDArray> map = ChronicleMap.of(Integer.class, INDArray.class)
                .entries(3_000_000)
                .averageValue(Nd4j.rand(new int[]{1, 300}))  //Used to estimate size of objects
                .create();


            Nd4j.getMemoryManager().togglePeriodicGc(false);
            System.out.println("Periodic GC disabled... inserting values:");

            AtomicInteger counter = new AtomicInteger(0);
            final int nThreads = 6;
            Thread[] threads = new Thread[nThreads];
            for (int t = 0; t < nThreads; t++) {
                threads[t] = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        while (true) {
                            int i = counter.getAndIncrement();
                            if (i < 3_000_000) {
                                map.put(i, storage.get(i));
                            } else {
                                break;
                            }
                        }
                    }
                });
                threads[t].start();
            }
            for (int i = 0; i < nThreads; i++) {
                threads[i].join();
            }

            wordVectors = null;
            wv = null;
            storage = null;
            System.gc();

            WordVectorProvider.map = map;
            return map;
        }
    }
}

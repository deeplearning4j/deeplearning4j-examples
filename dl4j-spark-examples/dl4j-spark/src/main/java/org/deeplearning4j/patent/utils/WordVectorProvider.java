package org.deeplearning4j.patent.utils;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;

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

    private static WordVectors wordVectors;

    public static synchronized WordVectors getWordVectors(Configuration config, String path) throws IOException {
        if (wordVectors != null) {
            return wordVectors;
        }
        Nd4j.getMemoryManager().setAutoGcWindow(30000);

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

        File localFile = new File(System.getProperty("user.home"), ".deeplearning4j/GoogleNews-vectors-negative300.bin.gz");

        if(localFile.exists() && localFile.length() > 0){
            try{
                log.info("Loading word vectors from saved file: {}", localFile.getAbsolutePath());
                long start = System.currentTimeMillis();
                wordVectors = WordVectorSerializer.loadStaticModel(localFile);
                log.info("Finished loading word vectors - duration {} sec", (System.currentTimeMillis()-start)/1000);
                return wordVectors;
            } catch (Exception e){
                localFile.delete();
                log.error("Error loading existing word vectors");
            }
        }

        try {
            log.info("Copying word vectors");
            long start = System.currentTimeMillis();
            try (InputStream in = new BufferedInputStream(fs.open(new Path(u))); OutputStream os = new BufferedOutputStream(new FileOutputStream(localFile))) {
                IOUtils.copy(in, os);
            }
            log.info("Finished copying word vectors - duration {} sec", (System.currentTimeMillis()-start)/1000);

            log.info("Loading word vectors");
            start = System.currentTimeMillis();
            wordVectors = WordVectorSerializer.loadStaticModel(localFile);
            log.info("Finished loading word vectors - duration {} sec", (System.currentTimeMillis()-start)/1000);
            return wordVectors;
        } catch (Exception e){
            localFile.delete();
            throw e;
        }
    }

}

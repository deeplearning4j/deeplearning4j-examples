package org.deeplearning4j.examples;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.datavec.hadoop.conf.ConfigurationUtil;

/**
 * Base class for HDFS Iterator. This class holds your HDFS configs
 *
 * @author: Ousmane A. Dia
 */
public class BaseDataSetIterator {

    protected volatile RemoteIterator<LocatedFileStatus> hdfsIterator;

    private final Configuration configuration;

    protected volatile FileSystem fs;
    protected volatile String hdfsUrl;

    public BaseDataSetIterator(Configuration configuration, String hdfsUrl) {
        this.configuration = configuration;
        initIterator(hdfsUrl);
    }

    /**
     * This method creates an instance of {@code org.apache.hadoop.conf.Configuration} to pass to the constructor
     * {@link BaseDataSetIterator#BaseDataSetIterator(Configuration, String)}
     * 
     * @param baseConfPath Config path
     * @return an instance of {@code org.apache.hadoop.conf.Configuration}
     */
    public static final Configuration initialize(String baseConfPath) {

        Configuration configuration = ConfigurationUtil.generateConfig(baseConfPath);
        configuration.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
        configuration.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());
        return configuration;
    }

    protected String getRelativeFilename(String path) {
        String index = null;
        StringTokenizer pathTokens = new StringTokenizer(path, "/");
        while (pathTokens.hasMoreTokens()) {
            index = pathTokens.nextToken();
        }
        return index;
    }

    /**
     * Adding this method to help reset the iterator (see {@link MDSIterator#reset()}
     */
    protected void initIterator(String hdfsUrl) {
        try {
            this.hdfsUrl = hdfsUrl;
            fs = FileSystem.get(configuration);
            hdfsIterator = fs.listFiles(new Path(this.hdfsUrl), true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

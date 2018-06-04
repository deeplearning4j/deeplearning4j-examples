package org.deeplearning4j.utils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * @Description: The clusters've two nodes, one is master node that its domain name is "cluster1" , the domain name of the slave node is "cluster2"
 * @author wangfeng
 */
public class CommonUtils {
    public static final String SERVER_PATH = "hdfs://cluster1:9003";//because the 9000 port have been used,so here's 9003

    public static final String TRAIN_HDFS_PATH = SERVER_PATH + "/user/hadoop/animals/train";
    public static final String VALIDATE_HDFS_PATH = SERVER_PATH + "/user/hadoop/animals/validate";

    public static FileSystem openHdfsConnect() {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", SERVER_PATH);
        FileSystem fs = null;
        try {
            fs = FileSystem.newInstance(new URI(SERVER_PATH),conf);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return fs;
    }
    public static void closeHdfsConnect(FileSystem fs) {
        try {
            if (fs != null) {
                fs.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static JavaSparkContext createConf() {

        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("animalClass");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        return sc;
    }
}

package org.deeplearning4j.utils;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;


/**
 * @Description: showing the training stats
 * @author wangfeng
 */
public class ShowData2UIServer {

    private static final boolean hdfsPath = false;
    public static void main(String[] args) throws Exception{

        //load the data saved, and then it'll be showed ,in the browser:http://localhost:9000

        File statsFile = null;
        if (hdfsPath) {
            statsFile = File.createTempFile("tmp", "dl4j");
            OutputStream os = new FileOutputStream(statsFile);
            FileSystem fs = CommonUtils.openHdfsConnect();
            InputStream in = fs.open(new Path("/user/hadoop/trainlog/AnimalModelByHdfsTrainingStatsSpark2.dl4j"));
            IOUtils.copyBytes(in, os, 4096, false); //复制到标准输出流
            IOUtils.closeStream(in);
            CommonUtils.closeHdfsConnect(fs);
            os.close();
        } else {
            statsFile = new File("/home/AnimalModelByHdfsTrainingStats1.dl4j");
        }
        StatsStorage statsStorage = new FileStatsStorage(statsFile);
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
    }
}

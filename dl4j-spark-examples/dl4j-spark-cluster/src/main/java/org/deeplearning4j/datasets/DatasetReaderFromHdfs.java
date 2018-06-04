package org.deeplearning4j.datasets;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.*;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.utils.CommonUtils;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


/**
 * @Description: The clusters've two nodes, one is master node that its domain name is "cluster1" , the domain name of the slave node is "cluster2"
 * @author wangfeng
 */
public class DatasetReaderFromHdfs extends NativeImageLoader implements Serializable {

    private static final Logger log = LoggerFactory.getLogger(DatasetReaderFromHdfs.class);



    private static volatile RemoteIterator<LocatedFileStatus> hdfsIter;
    private static int height = 100;
    private static int width = 100;
    private static int channels = 3;
    private Iterator<String> fileIterator;
    private List<String> fileNames;
    private int numExample = 0;
    private List<String> labels;
    private boolean train;


    public DatasetReaderFromHdfs() {
        this( true );
    }

    public DatasetReaderFromHdfs(boolean train) {
        this( height, width, channels, train, System.currentTimeMillis(), true );
    }
    public DatasetReaderFromHdfs(int height, int width, int channels,  boolean train, long seed, boolean shuffle) {
        super(height, width, channels, (ImageTransform)null);
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.train = train;
        this.labels = new ArrayList<String>();
        this.fileNames = new ArrayList<>();
        doInitialize();
    }

    protected void doInitialize() {
        FileSystem fs = CommonUtils.openHdfsConnect();
        try {
            if (train) {
                hdfsIter= fs.listFiles(new Path(CommonUtils.TRAIN_HDFS_PATH), true);
            } else {
                hdfsIter= fs.listFiles(new Path(CommonUtils.VALIDATE_HDFS_PATH), true);
            }
            while (hdfsIter.hasNext()) {
                LocatedFileStatus next = hdfsIter.next();
                Path path = next.getPath();
                String currentPath = path.toUri().getPath();
                fileNames.add(path.toString());
                String name = FilenameUtils.getBaseName((new File(currentPath)).getParent());
                if (!labels.contains(name)) {
                    labels.add(name);
                }

            }
            Collections.shuffle(fileNames);
            fileIterator = fileNames.iterator();
            numExample = fileNames.size();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            CommonUtils.closeHdfsConnect(fs);
        }
    }
    public DataSet convertDataSet(int num)  {
        int batchNumCount = 0;
        List<DataSet> dataSets = new ArrayList();
        FileSystem fs = CommonUtils.openHdfsConnect();
        try {
            while (batchNumCount != num && fileIterator.hasNext()) {
                ++ batchNumCount;
                String fullPath = fileIterator.next();

                Writable labelText = new Text(FilenameUtils.getBaseName((new File(fullPath)).getParent()));
                INDArray features = null;
                INDArray label =  Nd4j.zeros(1, labels.size()).putScalar(new int[]{0, labels.indexOf(labelText)}, 1);

                InputStream imageios = fs.open(new Path(fullPath));
                features = asMatrix(imageios);
                imageios.close();
                Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST);
                dataSets.add(new DataSet(features, label));
            }
        } catch (Exception e) {
            throw  new RuntimeException(e.getCause());
        } finally {
            CommonUtils.closeHdfsConnect(fs);
        }
        if (dataSets.size() == 0) {
            return new DataSet();
        } else {
            DataSet result = DataSet.merge( dataSets );
            return result;
        }
    }

    public DataSet next(int batchSize) {
        DataSet result = convertDataSet( batchSize );
        return result;
    }

    public void reset() {
        numExample = 0;
       // doInitialize();
        fileIterator = fileNames.iterator();
        numExample = fileNames.size();
    }
    public int totalExamples() {
        return numExample;
    }
    public List<String> getLabels() {
        return this.labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }


    public static void main(String[] args) throws IOException{
        DatasetReaderFromHdfs ds = new DatasetReaderFromHdfs();
        int j = 0;
        while (hdfsIter.hasNext()) {
            LocatedFileStatus next = hdfsIter.next();
            Path path = next.getPath();
            String currentPath = path.toUri().getPath();
            //String index = getRelativeFilename(currentPath);
            System.out.println("file name : i = " + j ++ + " path=" + currentPath);
        }
    }


}

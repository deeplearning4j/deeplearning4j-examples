package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**the data is from the image, not ubyte format file
 * @author wangfeng
 */
public class MnistLoader extends NativeImageLoader implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(MnistLoader.class);

    private static int height = 28;
    private static int width = 28;
    private static int channels = 1;
    private File fullDir;
    private Iterator<File> fileIterator;
    private boolean train;
    private long seed;
    private boolean shuffle;
    private int fileNum = 0;
    private int numExample = 0;


    public MnistLoader() {
        this( true );
    }

    public MnistLoader(boolean train) {
        this( train, (File) null );
    }

    public MnistLoader(boolean train, File fullPath) {
        this( height, width, channels, train, fullPath, System.currentTimeMillis(), true );
    }

    public MnistLoader(int height, int width, int channels,  boolean train, File fullDir, long seed, boolean shuffle) {
        super(height, width, channels, null);
        this.shuffle = true;
        this.fileNum = 0;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.train = train;
        this.seed = seed;
        this.shuffle = shuffle;
        this.fullDir = fullDir != null && fullDir.exists()? fullDir : new File( "/home/datasets/MNIST/");
        load();
    }

    protected void load() {
        try {
            if (train) {
                File trainFile = new File(fullDir, "train");
                Collection<File> trainFiles = FileUtils.listFiles( trainFile, new String[]{"jpg"}, true );
                fileIterator = trainFiles.iterator();
                numExample = trainFiles.size();
            } else {
                File testFile = new File(fullDir, "test");
                Collection<File> testFiles = FileUtils.listFiles(testFile, new String[]{"jpg"}, true );
                fileIterator = testFiles.iterator();
                numExample = testFiles.size();
            }
        } catch (Exception var4) {
            throw new RuntimeException( var4 );
        }
    }


    public DataSet convertDataSet(int num)  {
        int batchNumCount = 0;
        List<DataSet> dataSets = new ArrayList();
        while (batchNumCount != num && fileIterator.hasNext()) {
            ++ batchNumCount;
            File image = fileIterator.next();
            INDArray features = null;
            try {
                features = asMatrix(image);
            } catch (Exception e) {
                log.error("loading the file showing exception ", e);
                throw new RuntimeException(e.getCause());
            }
            features =  features.muli(1.0/255.0);
            features = features.ravel();
            Nd4j.getAffinityManager().ensureLocation(features, AffinityManager.Location.DEVICE);
            dataSets.add(new DataSet(features, features));
            fileNum ++;
        }
        log.info("deal the data {}", fileNum);
        if (dataSets.size() == 0) {
            return new DataSet();
        } else {
            DataSet result = DataSet.merge( dataSets );
            if (shuffle && num > 1) {
                result.shuffle( seed );
            }
            return result;
        }
    }


    public DataSet next(int batchSize) {
        DataSet result = convertDataSet( batchSize );
        return result;
    }

    public void reset() {
        fileNum = 0;
        load();
    }
    public int totalExamples() {
        return numExample;
    }

}

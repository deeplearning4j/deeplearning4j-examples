package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.apache.commons.io.FileUtils;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**the data is from the image, not ubyte format file
 * @author wangfeng
 */
public class MnistLoader implements Serializable {
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
    public MnistLoader(int height, int width, int channels, ImageTransform imgTransform, boolean train, boolean shuffle) {
        this( height, width, channels, train,  (File) null, System.currentTimeMillis(), shuffle );
    }
    public MnistLoader(int height, int width, int channels,  boolean train, File fullDir, long seed, boolean shuffle) {
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

    public INDArray asRowVector(File f) throws IOException {
        throw new UnsupportedOperationException();
    }

    public INDArray asRowVector(InputStream inputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    public INDArray asMatrix(File f)  {
        byte [] img = image2Bytes(f);


        float[] featureVec = new float[img.length];

        for (int j = 0; j < img.length; j++) {
            float v = ((int) img[j]) & 0xFF; //byte is loaded as signed -> convert to unsigned
            featureVec[j] = v / 255.0f;
        }
        INDArray features = Nd4j.create(featureVec);
        return features;
    }

    public  byte[] image2Bytes(File f) {
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(f);
        } catch(Exception e) {
            log.error("when loading the file it's exception",e);
        }
        if(bi == null) {
            return new byte[0];
        }
        BufferedImage grayImage = new BufferedImage(width, height,BufferedImage.TYPE_BYTE_GRAY);
        new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(bi, grayImage);
        // getting the pixes bytes from the images
        return (byte[]) grayImage.getData().getDataElements(0, 0, width, height, null);
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
            INDArray row = asMatrix(image);
            Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE);
            dataSets.add(new DataSet(row, row));
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

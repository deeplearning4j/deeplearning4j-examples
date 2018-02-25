package org.deeplearning4j.examples.convolution.captcharecognition;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;


/**
 * @author WangFeng
 */
public class MulRecordDataLoader extends NativeImageLoader implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(MulRecordDataLoader.class);

    private static int height = 60;
    private static int width = 160;
    private static int channels = 1;
    private File fullDir = null;
    private Iterator<File> fileIterator;
    private boolean train;
    private long seed;
    private boolean shuffle;
    private int fileNum = 0;
    private int numExample = 0;


    public MulRecordDataLoader(String dataSetType) {
        this( height, width, channels, null, dataSetType, System.currentTimeMillis(), true );
    }
    public MulRecordDataLoader(ImageTransform imageTransform, String dataSetType)  {
        this( height, width, channels, imageTransform, dataSetType, System.currentTimeMillis(), true );
    }
    public MulRecordDataLoader(int height, int width, int channels, ImageTransform imageTransform, String dataSetType, long seed, boolean shuffle) {
        super(height, width, channels, imageTransform);
        this.shuffle = true;
        this.fileNum = 0;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.train = train;
        this.seed = seed;
        this.shuffle = shuffle;
        try {
            this.fullDir = fullDir != null && fullDir.exists() ? fullDir : new ClassPathResource("/captchaImage").getFile();
        } catch (Exception e) {
            log.error("the datasets directory failed, plz checking", e);
        }
        this.fullDir = new File(fullDir, dataSetType);
        load();
    }

    protected void load() {
        try {
            List<File> dataFiles = (List<File>) FileUtils.listFiles(fullDir, new String[]{"jpeg"}, true );
            Collections.shuffle(dataFiles);
            fileIterator = dataFiles.iterator();
            numExample = dataFiles.size();
        } catch (Exception var4) {
            throw new RuntimeException( var4 );
        }
    }

    public MultiDataSet convertDataSet(int num) throws Exception {
        int batchNumCount = 0;

        INDArray[] featuresMask = null;
        INDArray[] labelMask = null;

        List<MultiDataSet> multiDataSets = new ArrayList<>();

        while (batchNumCount != num && fileIterator.hasNext()) {
            File image = fileIterator.next();
            String imageName = image.getName().substring(0,image.getName().lastIndexOf('.'));
            String[] imageNames = imageName.split("");
            INDArray feature = asMatrix(image);
            INDArray[] features = new INDArray[]{feature};
            INDArray[] labels = new INDArray[6];

            Nd4j.getAffinityManager().ensureLocation(feature, AffinityManager.Location.DEVICE);
            if (imageName.length() < 6) {
                imageName = imageName + "0";
                imageNames = imageName.split("");
            }
            for (int i = 0; i < imageNames.length; i ++) {
                int digit = Integer.parseInt(imageNames[i]);
                labels[i] = Nd4j.zeros(1, 10).putScalar(new int[]{0, digit}, 1);
            }
            feature =  feature.muli(1.0/255.0);

            multiDataSets.add(new MultiDataSet(features, labels, featuresMask, labelMask));

            fileNum ++;
            batchNumCount ++;
        }
        MultiDataSet result = MultiDataSet.merge(multiDataSets);
        return result;
    }

    public MultiDataSet next(int batchSize) {
        try {
            MultiDataSet result = convertDataSet( batchSize );
            return result;
        } catch (Exception e) {
            log.error("the next function shows error", e);
        }
        return null;
    }

    public void reset() {
        fileNum = 0;
        load();
    }
    public int totalExamples() {
        return numExample;
    }
}

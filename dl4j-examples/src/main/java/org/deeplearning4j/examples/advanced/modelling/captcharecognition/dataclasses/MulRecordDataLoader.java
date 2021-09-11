/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.captcharecognition.dataclasses;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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
    private int numExample = 0;


    public MulRecordDataLoader(String dataSetType) throws Exception {
        this( height, width, channels, null, dataSetType);
    }
    public MulRecordDataLoader(ImageTransform imageTransform, String dataSetType) throws Exception {
        this( height, width, channels, imageTransform, dataSetType );
    }
    public MulRecordDataLoader(int height, int width, int channels, ImageTransform imageTransform, String dataSetType) throws Exception {
        super(height, width, channels, imageTransform);
        this.height = height;
        this.width = width;
        this.channels = channels;
        String dataLocalPath = DownloaderUtility.CAPTCHAIMAGE.Download();
        try {
            this.fullDir = fullDir != null && fullDir.exists() ? fullDir : new File(dataLocalPath);
        } catch (Exception e) {
           // log.error("the datasets directory failed, plz checking", e);
            throw new RuntimeException( e );
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
        load();
    }
    public int totalExamples() {
        return numExample;
    }
}

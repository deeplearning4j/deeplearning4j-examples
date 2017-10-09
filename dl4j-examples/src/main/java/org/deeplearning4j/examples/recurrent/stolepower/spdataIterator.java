package org.deeplearning4j.examples.recurrent.stolepower;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by wangfeng on 2017/9/21.
 */
public class spdataIterator implements DataSetIterator {
    private static final Logger log = LoggerFactory.getLogger(spdataIterator.class);
    private final int vectorSize = 12;
    private final int batchSize;
    private final int teamLength;
    private int maxLength;
    private final String dataDirectory;
    private final List<Pair<String, List<String>>> thiefData = new ArrayList<>();
    private int cursor = 0;
    private int totalDataSize = 0;
    private int newsPosition = 0;
    private final List<String> labels;
    private int currCategory = 0;

    public spdataIterator(String dataDirectory, int batchSize, boolean train) {
        this.dataDirectory = dataDirectory;
        this.batchSize = batchSize;
        this.teamLength = 365;
        this.loadData(train);
        this.labels = new ArrayList<>();
        for (int i = 0; i < this.thiefData.size(); i++) {
            this.labels.add(this.thiefData.get(i).getKey());
        }
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        cursor = 0;
        newsPosition = 0;
        currCategory = 0;
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    @Override
    public boolean hasNext() {
        return this.cursor < this.totalDataSize;
    }

    @Override
    public DataSet next() {

        if (cursor >= this.totalDataSize || totalDataSize  <= 0 ) {
            throw new NoSuchElementException();
        }
        return next(batchSize);
    }
    public DataSet next(int num) {

        //int actualBatchSize = Math.min(num, dataRecord.size());
        //int actualLength = Math.min(exampleLength,dataList.size()-dataRecord.get(0)-1);
       // INDArray features = Nd4j.create(new int[]{actualBatchSize,VECTOR_SIZE,actualLength}, 'f');
       // INDArray labels = Nd4j.create(new int[]{actualBatchSize,1,actualLength}, 'f');

        List<String> yearDataList = new ArrayList<>(num);
        int[] category = new int[num];

        for (int i = 0; i < num && cursor < totalExamples(); i++) {
            if (currCategory < thiefData.size()) {
                yearDataList.add(this.thiefData.get(currCategory).getValue().get(newsPosition));
                category[i] = Integer.parseInt(this.thiefData.get(currCategory).getKey());
                currCategory++;
                cursor++;
            } else {
                currCategory = 0;
                newsPosition++;
                i--;
            }
        }



        //Create data for training
        //Here: we have news.size() examples of varying lengths
        int maxLength = 365;
        INDArray features = Nd4j.create(yearDataList.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(yearDataList.size(), this.thiefData.size(), maxLength);    //Three labels: Crime, Politics, Bollywood

        int[] temp = new int[2];
        for (int i = 0; i < yearDataList.size(); i++) {

        }

        DataSet ds = new DataSet(features, labels);
        return ds;

    }
    private void loadData(boolean train) {
        String curFileName = train == true ?
            this.dataDirectory + File.separator + "train" + File.separator + "STEAL_RESULT.csv" :
            this.dataDirectory + File.separator + "test" + File.separator + "STEAL_RESULT.csv";
        File dataFile = new File(curFileName);
        try {
            FileInputStream fis = new FileInputStream(dataFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis,"UTF-8"));
            System.out.println("读取数据..");
            String tempLine = "";
            List<String> stealFeatureList = new ArrayList<>();
            List<String> unStealFeatureList = new ArrayList<>();

            Pair<String, List<String>> stealPair = Pair.of("1", stealFeatureList);
            Pair<String, List<String>> unStealPair = Pair.of("0", unStealFeatureList);

            while ((tempLine = br.readLine()) != null) {
                String[] features = tempLine.substring(0,tempLine.lastIndexOf( ',' )).split(",");
                String label = tempLine.substring(tempLine.lastIndexOf( ',' ));
                for (String d: features) {
                    if ("1".equalsIgnoreCase( label )) {
                        stealFeatureList.add(d);
                    } else {
                        unStealFeatureList.add(d);
                    }
                    this.totalDataSize++;
                }
            }
            this.thiefData.add(stealPair);
            this.thiefData.add(unStealPair);

            br.close();
            fis.close();
        } catch(Exception e) {
            log.error("Exception in reading file :" + e.getMessage());
        }


    }
}

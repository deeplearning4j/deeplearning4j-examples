package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * @Description lottery value -> lottery value
 * @Author WangFeng
 */

public class LotteryCombinationDataSetReader implements BaseDataSetReader  {
    private static final Logger log = LoggerFactory.getLogger(LotteryCombinationDataSetReader.class);

    private Iterator<String> iter;
    private Path filePath;
    private int lotteryLength = 5;

    private int totalExamples;
    private int currentCursor;

    public LotteryCombinationDataSetReader() {}
    public LotteryCombinationDataSetReader(File file) {
        filePath = file.toPath();
        doInitialize();
    }

    public void doInitialize(){
        List<String> dataLines;
        try {
            dataLines = Files.readAllLines(filePath, Charset.forName("UTF-8"));
        } catch (Exception e) {
            throw new RuntimeException("loading data failed");
        }
        iter = dataLines.iterator();
        totalExamples = dataLines.size();
        currentCursor = 0;
    }

  public DataSet next(int num) {
      String currentValStr = "";
      INDArray features = Nd4j.create(new int[]{num-1, lotteryLength, 10});
      INDArray labels = Nd4j.create(new int[]{num-1, lotteryLength, 10});
      for (int i =0; i < num - 1  && iter.hasNext(); i ++) {
          String featureStr = "";
          if (null == currentValStr || "".equals(currentValStr)) {
              currentValStr = iter.next();
              currentCursor ++;
          }
          featureStr = currentValStr;

          featureStr = featureStr.split(",")[2];
          String[] featureAry = featureStr.split("");
          for (int j = 0; j < lotteryLength; j ++) {
              int l = Integer.parseInt(featureAry[j]);
              features.putScalar(new int[]{i, j, l}, 1);
          }

          String labelStr = iter.next();
          currentCursor ++;
          currentValStr = labelStr;
          labelStr = labelStr.split(",")[2];
          String[] labelAry = labelStr.split("");
          for (int j = 0; j < lotteryLength; j ++) {
              int l = Integer.parseInt(labelAry[j]);
              labels.putScalar(new int[]{i, j, l}, 1);
          }

      }
      DataSet result = new DataSet(features, labels);
      return result;
  }

    public boolean hasNext() {
        if (iter != null && iter.hasNext() ) {
            return true;
        }
        return false;
    }

    public List<String> getLabels() {
        return null;
    }

    public void reset() {
        doInitialize();
    }
    public int totalExamples() {
        return totalExamples;
    }
    public int cursor() {
        return currentCursor;
    }


    //based on the lottery rule,here will the openning lottery date and term switch to the long integer
    private String decorateRecordData(String line) {
        if (line == null && line.isEmpty()) {
            return null;
        }
        String[] strArr = line.split(",");
        if (strArr.length >= 2) {
            //translation all time,
            String openDateStr = strArr[0].substring(0, strArr[0].length() - 3);
            openDateStr = openDateStr.substring(0, 4) + "-" + openDateStr.substring(4, 6) + "-" + openDateStr.substring(6, 8);
            String issueNumStr = strArr[0].substring(strArr[0].length() - 3);
            int issueNum = Integer.parseInt(issueNumStr);
            int minutes = 0;
            int hours = 0;
            if (issueNum >= 24 && issueNum < 96) {
                int temp = (issueNum - 24) * 10;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 10;
            } else if (issueNum >= 96 && issueNum <= 120) {
                int temp = (issueNum - 96) * 5;
                minutes = temp % 60;
                hours = temp / 60;
                hours += 22;
            } else {
                int temp = issueNum * 5;
                minutes = temp % 60;
                hours = temp / 60;
            }

            openDateStr = openDateStr + " " + hours + ":" + (minutes == 0 ? "00" : minutes) + ":00";
            long openDateStrNum = 0;

            try {
                SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//yyyy-MM-dd HH:mm:ss
                Date midDate = formatter.parse(openDateStr);
                openDateStrNum = midDate.getTime();

            } catch (Exception e) {
                throw  new RuntimeException("the decorateRecordData function shows exception!", e.getCause());
            }
            String lotteryValue = strArr[1];
            lotteryValue = lotteryValue.replace("", ",");
            line = openDateStrNum + lotteryValue;
        }
        return line;
    }



}

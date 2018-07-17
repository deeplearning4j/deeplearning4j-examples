package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

/**
 * @author WangFeng
 */
public abstract class BaseDataSetReader implements Serializable {

    protected Iterator<String> iter;
    protected Path filePath;
    protected int totalExamples;
    protected int currentCursor;

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

    public DataSet next(int num){
        return null;
    }

    public boolean hasNext() {
        return iter != null && iter.hasNext();
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

}

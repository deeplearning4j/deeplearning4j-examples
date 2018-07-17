package org.deeplearning4j.examples.recurrent.processlottery;

import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.List;


public interface BaseDataSetReader extends Serializable {

    public void doInitialize();

    public DataSet next(int num);

    public boolean hasNext();

    public List<String> getLabels();

    public void reset();
    public int totalExamples();
    public int cursor();
}

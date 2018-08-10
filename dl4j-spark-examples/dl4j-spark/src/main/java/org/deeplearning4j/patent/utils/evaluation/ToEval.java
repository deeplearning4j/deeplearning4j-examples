package org.deeplearning4j.patent.utils.evaluation;

import java.io.File;

public class ToEval {
    private final File f;
    private final int count;
    private final long durationSoFar;

    public ToEval(File f, int count, long  durationSoFar){
        this.f = f;
        this.count = count;
        this.durationSoFar = durationSoFar;
    }

    public File getFile(){
        return f;
    }

    public int getCount(){
        return count;
    }

    public long getDurationSoFar(){
        return durationSoFar;
    }

}

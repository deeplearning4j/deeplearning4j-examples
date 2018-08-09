package org.deeplearning4j.patent.preprocessing;

/**
 * A simple class used for holding patent text (and classification - i.e., class label) for the patent example
 */
public class Patent {

    protected String id;
    protected String classificationUS;
    protected String allText;

    public void setClassificationUS(String s){
        this.classificationUS = s;
    }

    public void setAllText(String text){
        this.allText = text;
    }

    public String getAllText(){
        return allText;
    }

    public String getClassificationUS(){
        return classificationUS;
    }
}

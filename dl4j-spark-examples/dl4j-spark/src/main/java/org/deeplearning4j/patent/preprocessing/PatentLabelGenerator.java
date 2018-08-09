package org.deeplearning4j.patent.preprocessing;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * This class handles inferring labels for the patent example.
 * Specifically, it infers the USPTO labels from the label text.
 * Note that USPTO classifications are heirarchical; "tier 1" is the most general class (plant, design, etc)
 * Whereas tier 2 is for specific categories of inventions
 *
 * @author Alex Black
 */
public class PatentLabelGenerator {
    /**
     * Tier 1:
     * - Design: D prefix
     * - Plant: PLT prefix
     * - Utility: Numerical, 1 to 999
     * - G: ???
     *
     * @param in "main-classification" (USPO US classification) text from patent raw data
     * @return
     */
    public static String tier1Label(String in){
        if(in.matches("\\s*D.*")){
            return "D";
        }
        if(in.matches("\\s*PLT.*")){
            return "PLT";
        }
        if(in.matches("\\s*\\d.*")){
            return "U";
        }
        if(in.matches("\\s*G.*")){
            return "G";
        }
        throw new RuntimeException(in);
    }

    /**
     * Main classification can have multiple formats, but are represeneted heirarchically...
     * We know: leading " " means 1 or 2x leading 0s in number...
     *
     * @param in "main-classification" (USPO US classification) text from patent raw data
     */
    public static String tier2Label(String in){
        if (in.matches("\\s*D.*")) {
            if(in.startsWith("D ")){
                in = in.replace("D ", "D0");
            }
            if(in.startsWith(" D")){
                in = in.replace(" D", "D0");
            }
            String tier2 = in.substring(0, 3);
            return tier2;
        } else if(in.startsWith("PLT")){
            return "PLT";
        } else if(in.matches("\\s*\\d.*")){
            //Utility

            //First: strip any non-digits (these are tier 3: "62DIG" -> "62"
            for( int i=0; i<in.length(); i++ ){
                if(!Character.isDigit(in.charAt(i)) && in.charAt(i) != ' '){
                    in = in.substring(0, i);
                    break;
                }
            }

            String ret;
            if(in.startsWith(" ")){ //Leading double space - "  " never occurs...
                String sub = in.substring(1);
                if(sub.contains(" ")){
                    //" 72 3106"        ->  072 /   031.06
                    //" 56 50"          ->  056 /   050
                    String[] split = sub.split(" ");
                    ret = split[0];
                } else {
                    if(sub.length() >= 3) {
                        //" 70224"          ->  070 /   224
                        ret = sub.substring(0, 2);
                    } else {
                        ret = sub;
                    }
                }
            } else if(in.length() >= 3) {
                //Must be leading 3 digits present
                //"100117"              ->  100 /   117
                ret = in.substring(0, 3);
            } else {
                ret = in;
            }

            String out;
            if(ret.length() == 1){
                out = "00" + ret;
            } else if(ret.length() == 2){
                out = "0" + ret;
            } else {
                out = ret;
            }

            if(classLabelNames().containsKey(out)){
                return out;
            }
            return null;    //Old/obsolete label
        }
        return null;
    }

    private static Map<String,String> classLabelNames;

    public static synchronized Map<String,String> classLabelNames(){
        if(classLabelNames != null){
            return classLabelNames;
        }

        String s;
        try {
            s = FileUtils.readFileToString(new ClassPathResource("PatentClassLabels.txt").getFile(), Charset.forName("UTF-8"));
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        Map<String,String> m = new LinkedHashMap<>();
        String[] lines = s.split("\n");
        for(String line : lines){
            String key = line.substring(0,3);
            String name = line.substring(4);
            m.put(key, name);
        }
        classLabelNames = m;
        return classLabelNames;
    }


    private static Map<String,Integer> classLabelsFilteredCounts;

    public static synchronized Map<String,Integer> classLabelFilteredCounts(){
        if(classLabelsFilteredCounts != null){
            return classLabelsFilteredCounts;
        }

        String s;
        try {
            s = FileUtils.readFileToString(new ClassPathResource("FilteredPatentClassCounts.txt").getFile(), Charset.forName("UTF-8"));
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        Map<String,Integer> m = new LinkedHashMap<>();
        String[] lines = s.split("\n");
        for(String line : lines){
            String[] split = line.split(",");
            m.put(split[0], Integer.parseInt(split[1]));
        }
        classLabelsFilteredCounts = m;
        return classLabelsFilteredCounts;
    }

    private static Map<String,Integer> classLabelToIndex;
    public static synchronized Map<String,Integer> classLabelToIndex(){
        if(classLabelToIndex != null){
            return classLabelToIndex;
        }
        Map<String,Integer> m = new LinkedHashMap<>();

        String s;
        try {
            s = FileUtils.readFileToString(new ClassPathResource("FilteredPatentClassCounts.txt").getFile(), Charset.forName("UTF-8"));
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        int i=0;
        for(String line : s.split("\n")){
            String[] split = line.split(",");
            m.put(split[0].trim(), i++);
        }
        classLabelToIndex = m;
        return classLabelToIndex;
    }

}

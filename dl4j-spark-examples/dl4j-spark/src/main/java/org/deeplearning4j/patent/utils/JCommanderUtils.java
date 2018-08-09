package org.deeplearning4j.patent.utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;

public class JCommanderUtils {

    private JCommanderUtils(){ }

    public static void parseArgs(Object obj, String[] args){
        JCommander jcmdr = new JCommander(obj);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();  //User provides invalid input -> print the usage info
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }
    }
}

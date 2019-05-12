package com.deeplearning4java.commons;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

public class PrefManager {

    private static final String MODEL_PATH = "dl4j_model_path";

    public static void setModelPath(Context context, String value) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        prefs.edit().putString(MODEL_PATH, value).apply();
    }

    public static String getModelPath(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        return prefs.getString(MODEL_PATH, null);
    }
}

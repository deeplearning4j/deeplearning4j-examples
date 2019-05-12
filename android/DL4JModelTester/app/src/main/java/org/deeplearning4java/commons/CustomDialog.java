package com.deeplearning4java.commons;

import android.content.Context;
import android.content.DialogInterface;

import androidx.appcompat.app.AlertDialog;

public class CustomDialog {

    public CustomDialog(Context context, String msg, DialogInterface.OnClickListener positive) {
        this(context, msg, positive, null);
    }

    public CustomDialog(Context context, String msg, DialogInterface.OnClickListener positive, DialogInterface.OnClickListener negative) {
        AlertDialog.Builder builder1 = new AlertDialog.Builder(context);
        builder1.setMessage(msg);
        builder1.setCancelable(false);

        builder1.setPositiveButton("Yes", positive);

        if(negative != null) {
            builder1.setNegativeButton("No", negative);
        }

        AlertDialog alert = builder1.create();
        alert.show();
    }
}

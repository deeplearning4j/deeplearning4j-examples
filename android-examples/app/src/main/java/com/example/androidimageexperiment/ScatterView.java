package com.example.androidimageexperiment;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.AsyncTask;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.Nullable;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class ScatterView extends View {

    Paint redPaint;
    Paint greenPaint;
    Paint bluePaint;
    float[][] data;
    float minx, maxx;

    public ScatterView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        data = null;
        redPaint = new Paint();
        redPaint.setColor(Color.RED);
        greenPaint = new Paint();
        greenPaint.setColor(Color.GREEN);
        bluePaint= new Paint();
        bluePaint.setColor(Color.BLUE);

        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    ReadCSV();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    @Override
    public void onDraw(Canvas canvas){
        int h = this.getHeight();
        int w = this.getWidth();

        if (null == data) {
            canvas.drawColor(Color.rgb(32, 32, 32));
            canvas.drawCircle(800, 500, 200, redPaint);
            canvas.drawCircle(325, 900, 300, greenPaint);
            canvas.drawCircle(900, 1600, 400, bluePaint);
        } else {
            int numPoints = data.length;
            for(int i=0; i<numPoints; i++){
                int x = (int) (data[i][1] * w);
                int y = (int) (data[i][2] * h) ;
                Paint p = (data[i][0] == 0.0f) ? redPaint : greenPaint;
                canvas.drawCircle(x, y, 10, p);

            }
        }
    }

    /**
     * this is not the regular way to read a csv file into a dataset with DL4j.
     * In this example we have put the data in th eassets forlder so that the demo works offline.
     * @throws IOException
     */
    private void ReadCSV() throws IOException {
        InputStreamReader is = new InputStreamReader(MainActivity.getInstance().getApplicationContext().getAssets()
                .open("linear_data_train.csv"));

        BufferedReader reader = new BufferedReader(is);
        ArrayList<String> rawSVC = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            rawSVC.add(line);
        }

        float[][] tmpdata = new float[rawSVC.size()][3];
        minx = 1.0f;
        maxx = -1.0f;

        int index = 0;
        for(String l : rawSVC){
            String[] values = l.split(",");
            for(int col = 0; col<3l; col++){
                tmpdata[index][col] = Float.parseFloat(values[col]);
            }

            index++;
        }

        normalizeColumn(1, tmpdata);
        normalizeColumn(2, tmpdata);

        this.data = tmpdata;
        this.invalidate();
        Log.i("INFO", "Read file");
    }

    /**
     *  Normalize the data in a given column. Normally one would use datavec.
     * @param c column to normakuse.
     * @param tmpdata java loat array.
     */
    private void normalizeColumn(int c, float[][] tmpdata){
        int numPoints = tmpdata.length;
        float min= tmpdata[0][c];
        float max= tmpdata[0][c];
        for (int i=0; i<numPoints; i++){
            float x = tmpdata[i][c];
            if (x < min){min = x; }
            if (x > max){max = x; }
        }

        for (int i=0; i<numPoints; i++){
            float x = tmpdata[i][c];
            tmpdata[i][c] = (x - min)  / (max - min);
        }
    }
}

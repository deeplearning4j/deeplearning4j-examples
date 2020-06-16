package com.example.androidDl4jClassifier;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ScatterView extends View implements OnTrainingUpdateEventListener{

    private final Paint redPaint;
    private final Paint greenPaint;
    private final Paint lightGreenPaint;
    private final Paint lightRedPaint;


    private INDArray modelOut = null; // nn output for the grid.

    private final TrainingTask task;

    public ScatterView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        redPaint = new Paint();
        redPaint.setColor(Color.RED);
        greenPaint = new Paint();
        greenPaint.setColor(Color.GREEN);

        lightGreenPaint = new Paint();
        lightGreenPaint.setColor(Color.rgb(225, 255, 225));
        lightRedPaint = new Paint();
        lightRedPaint.setColor(Color.rgb(255, 153, 152));

        task = new TrainingTask();
        task.setListener(this);
        showDataset("linear_data_train.csv");
    }

    public void showDataset(String filename)  {
        task.executeTask(filename);
    }

    @Override
    public void onDraw(Canvas canvas) {
        int h = this.getHeight();
        int w = this.getWidth();

        //draw the nn predictions:
        if (modelOut != null) {
            int halfRectHeight = h / task.getnPointsPerAxis();
            int halfRectWidth = w / task.getnPointsPerAxis();
            INDArray  xyGrid = task.getXyGrid();
            int nRows = xyGrid.rows();

            for (int i = 0; i< nRows; i++){
                int  x =  (int)(xyGrid.getFloat(i, 0) * w);
                int y = (int) (xyGrid.getFloat(i, 1)  * h);
                float z = modelOut.getFloat(i, 0);
                Paint p = (z >= 0.5f) ? lightGreenPaint : lightRedPaint;
                canvas.drawRect(x-halfRectWidth, y-halfRectHeight, x+halfRectWidth, y+halfRectHeight, p);
                //  }
            }
        }

        //draw the data set if we have one.
        if (null != task.getData()) {

            for (float[] datum : task.getData()) {
                int x = (int) (datum[1] * w);
                int y = (int) (datum[2] * h);
                Paint p = (datum[0] == 0.0f) ? redPaint : greenPaint;
                canvas.drawCircle(x, y, 10, p);
            }
        }
    }

    @Override
    public void OnTrainingUpdate(INDArray modelOut) {
        this.post(() -> {
            this.modelOut =  modelOut; // update from within the UI thread.
            this.invalidate(); // have the UI thread redraw at its own convenience.
        });
    }
}

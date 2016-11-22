package org.deeplearning4j.examples.feedforward.classification;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.AxisLocation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.GrayPaintScale;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.PaintScaleLegend;
import org.jfree.data.xy.*;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;

/**Simple plotting methods for the MLPClassifier examples
 * @author Alex Black
 */
public class PlotUtil {

    /**Plot the training data. Assume 2d input, classification output
     * @param features Training data features
     * @param labels Training data labels (one-hot representation)
     * @param backgroundIn sets of x,y points in input space, plotted in the background
     * @param backgroundOut results of network evaluation at points in x,y points in space
     * @param nDivisions Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    public static void plotTrainingData(INDArray features, INDArray labels, INDArray backgroundIn, INDArray backgroundOut, int nDivisions){
        double[] mins = backgroundIn.min(0).data().asDouble();
        double[] maxs = backgroundIn.max(0).data().asDouble();

        XYZDataset backgroundData = createBackgroundData(backgroundIn, backgroundOut);
        JPanel panel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTrain(features, labels)));

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);
    }

    /**Plot the training data. Assume 2d input, classification output
     * @param features Training data features
     * @param labels Training data labels (one-hot representation)
     * @param predicted Network predictions, for the test points
     * @param backgroundIn sets of x,y points in input space, plotted in the background
     * @param backgroundOut results of network evaluation at points in x,y points in space
     * @param nDivisions Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    public static void plotTestData(INDArray features, INDArray labels, INDArray predicted, INDArray backgroundIn, INDArray backgroundOut, int nDivisions){

        double[] mins = backgroundIn.min(0).data().asDouble();
        double[] maxs = backgroundIn.max(0).data().asDouble();

        XYZDataset backgroundData = createBackgroundData(backgroundIn, backgroundOut);
        JPanel panel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTest(features, labels, predicted)));

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Test Data");

        f.setVisible(true);

    }


    /**Create data for the background data set
     */
    private static XYZDataset createBackgroundData(INDArray backgroundIn, INDArray backgroundOut) {
        int nRows = backgroundIn.rows();
        double[] xValues = new double[nRows];
        double[] yValues = new double[nRows];
        double[] zValues = new double[nRows];
        for( int i=0; i<nRows; i++ ){
            xValues[i] = backgroundIn.getDouble(i,0);
            yValues[i] = backgroundIn.getDouble(i,1);
            zValues[i] = backgroundOut.getDouble(i);

        }

        DefaultXYZDataset dataset = new DefaultXYZDataset();
        dataset.addSeries("Series 1",
                new double[][]{xValues, yValues, zValues});
        return dataset;
    }

    //Training data
    private static XYDataset createDataSetTrain(INDArray features, INDArray labels ){
        int nRows = features.rows();

        int nClasses = labels.columns();

        XYSeries[] series = new XYSeries[nClasses];
        for( int i=0; i<series.length; i++) series[i] = new XYSeries("Class " + String.valueOf(i));
        INDArray argMax = Nd4j.getExecutioner().exec(new IMax(labels), 1);
        for( int i=0; i<nRows; i++ ){
            int classIdx = (int)argMax.getDouble(i);
            series[classIdx].add(features.getDouble(i, 0), features.getDouble(i, 1));
        }

        XYSeriesCollection c = new XYSeriesCollection();
        for( XYSeries s : series) c.addSeries(s);
        return c;
    }

    //Test data
    private static XYDataset createDataSetTest(INDArray features, INDArray labels, INDArray predicted ){
        int nRows = features.rows();

        int nClasses = labels.columns();

        XYSeries[] series = new XYSeries[nClasses*nClasses];    //new XYSeries("Data");
        for( int i=0; i<nClasses*nClasses; i++){
            int trueClass = i/nClasses;
            int predClass = i%nClasses;
            String label = "actual=" + trueClass + ", pred=" + predClass;
            series[i] = new XYSeries(label);
        }
        INDArray actualIdx = Nd4j.getExecutioner().exec(new IMax(labels), 1);
        INDArray predictedIdx = Nd4j.getExecutioner().exec(new IMax(predicted), 1);
        for( int i=0; i<nRows; i++ ){
            int classIdx = (int)actualIdx.getDouble(i);
            int predIdx = (int)predictedIdx.getDouble(i);
            int idx = classIdx * nClasses + predIdx;
            series[idx].add(features.getDouble(i, 0), features.getDouble(i, 1));
        }

        XYSeriesCollection c = new XYSeriesCollection();
        for( XYSeries s : series) c.addSeries(s);
        return c;
    }

    private static JFreeChart createChart(XYZDataset dataset, double[] mins, double[] maxs, int nPoints, XYDataset xyData) {
        NumberAxis xAxis = new NumberAxis("X");
        xAxis.setRange(mins[0],maxs[0]);


        NumberAxis yAxis = new NumberAxis("Y");
        yAxis.setRange(mins[1], maxs[1]);

        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth((maxs[0]-mins[0])/(nPoints-1));
        renderer.setBlockHeight((maxs[1] - mins[1]) / (nPoints - 1));
        PaintScale scale = new GrayPaintScale(0, 1.0);
        renderer.setPaintScale(scale);
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));
        JFreeChart chart = new JFreeChart("", plot);
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false);


        NumberAxis scaleAxis = new NumberAxis("Probability (class 0)");
        scaleAxis.setAxisLinePaint(Color.white);
        scaleAxis.setTickMarkPaint(Color.white);
        scaleAxis.setTickLabelFont(new Font("Dialog", Font.PLAIN, 7));
        PaintScaleLegend legend = new PaintScaleLegend(new GrayPaintScale(),
                scaleAxis);
        legend.setStripOutlineVisible(false);
        legend.setSubdivisionCount(20);
        legend.setAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
        legend.setAxisOffset(5.0);
        legend.setMargin(new RectangleInsets(5, 5, 5, 5));
        legend.setFrame(new BlockBorder(Color.red));
        legend.setPadding(new RectangleInsets(10, 10, 10, 10));
        legend.setStripWidth(10);
        legend.setPosition(RectangleEdge.LEFT);
        chart.addSubtitle(legend);

        ChartUtilities.applyCurrentTheme(chart);

        plot.setDataset(1, xyData);
        XYLineAndShapeRenderer renderer2 = new XYLineAndShapeRenderer();
        renderer2.setBaseLinesVisible(false);
        plot.setRenderer(1, renderer2);

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);

        return chart;
    }

}

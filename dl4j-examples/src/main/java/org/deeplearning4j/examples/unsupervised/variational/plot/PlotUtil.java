package org.deeplearning4j.examples.unsupervised.variational.plot;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Plotting methods for the VariationalAutoEncoder example
 * @author Alex Black
 */
public class PlotUtil {

    public static void scatterPlot(List<List<double[]>> data, double axisMin, double axisMax, String title ){

        int nClasses = data.size();

        XYSeries[] series = new XYSeries[nClasses];
        for( int i=0; i<nClasses; i++){
            series[i] = new XYSeries(String.valueOf(i));
        }
        for( int i=0; i<nClasses; i++ ){
            for(double[] d : data.get(i)){
                series[i].add(d[0], d[1]);
            }
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        for( XYSeries s : series) dataset.addSeries(s);

        JFreeChart chart = ChartFactory.createScatterPlot(title,
            "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.getRenderer().setBaseOutlineStroke(new BasicStroke(0));
        plot.setNoDataMessage("NO DATA");

        plot.setDomainPannable(false);
        plot.setRangePannable(false);
        plot.setDomainZeroBaselineVisible(true);
        plot.setRangeZeroBaselineVisible(true);

        plot.setDomainGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainGridlinePaint(Color.blue);
        plot.setRangeGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeGridlinePaint(Color.blue);

        plot.setDomainMinorGridlinesVisible(true);
        plot.setRangeMinorGridlinesVisible(true);

        XYLineAndShapeRenderer renderer
            = (XYLineAndShapeRenderer) plot.getRenderer();
        renderer.setSeriesOutlinePaint(0, Color.black);
        renderer.setUseOutlinePaint(true);
        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setAutoRangeIncludesZero(false);
        domainAxis.setRange(axisMin, axisMax);

        domainAxis.setTickMarkInsideLength(2.0f);
        domainAxis.setTickMarkOutsideLength(2.0f);

        domainAxis.setMinorTickCount(2);
        domainAxis.setMinorTickMarksVisible(true);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setTickMarkInsideLength(2.0f);
        rangeAxis.setTickMarkOutsideLength(2.0f);
        rangeAxis.setMinorTickCount(2);
        rangeAxis.setMinorTickMarksVisible(true);
        rangeAxis.setRange(axisMin, axisMax);


        JPanel panel = new ChartPanel(chart);
        final JFrame f = new JFrame();
        f.add(panel);
        f.pack();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
    }

    public static void plotData(List<INDArray> xyVsIter, INDArray labels, double axisMin, double axisMax, int plotFrequency){

        JPanel panel = new ChartPanel(createChart(xyVsIter.get(0), labels, axisMin, axisMax));
        JSlider slider = new JSlider(0,xyVsIter.size()-1,0);
        slider.setSnapToTicks(true);

        final JFrame f = new JFrame();
        slider.addChangeListener(new ChangeListener() {

            private JPanel lastPanel = panel;
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider slider = (JSlider)e.getSource();
                int  value = slider.getValue();
                JPanel panel = new ChartPanel(createChart(xyVsIter.get(value), labels, axisMin, axisMax));
                if(lastPanel != null){
                    f.remove(lastPanel);
                }
                lastPanel = panel;
                f.add(panel, BorderLayout.CENTER);
                f.setTitle(getTitle(value, plotFrequency));
                f.revalidate();
            }
        });

        f.setLayout(new BorderLayout());
        f.add(slider, BorderLayout.NORTH);
        f.add(panel, BorderLayout.CENTER);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle(getTitle(0, plotFrequency));

        f.setVisible(true);
    }

    private static String getTitle(int recordNumber, int plotFrequency){
        return "MNIST Test Set - Latent Space Encoding at Training Iteration " + recordNumber * plotFrequency;
    }

    //Test data
    private static XYDataset createDataSet(INDArray features, INDArray labelsOneHot){
        int nRows = features.rows();

        int nClasses = labelsOneHot.columns();

        XYSeries[] series = new XYSeries[nClasses];
        for( int i=0; i<nClasses; i++){
            series[i] = new XYSeries(String.valueOf(i));
        }
        INDArray classIdx = Nd4j.argMax(labelsOneHot, 1);
        for( int i=0; i<nRows; i++ ){
            int idx = classIdx.getInt(i);
            series[idx].add(features.getDouble(i, 0), features.getDouble(i, 1));
        }

        XYSeriesCollection c = new XYSeriesCollection();
        for( XYSeries s : series) c.addSeries(s);
        return c;
    }

    private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax) {

        XYDataset dataset = createDataSet(features, labels);

        JFreeChart chart = ChartFactory.createScatterPlot("Variational Autoencoder Latent Space - MNIST Test Set",
            "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.getRenderer().setBaseOutlineStroke(new BasicStroke(0));
        plot.setNoDataMessage("NO DATA");

        plot.setDomainPannable(false);
        plot.setRangePannable(false);
        plot.setDomainZeroBaselineVisible(true);
        plot.setRangeZeroBaselineVisible(true);

        plot.setDomainGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainGridlinePaint(Color.blue);
        plot.setRangeGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeGridlinePaint(Color.blue);

        plot.setDomainMinorGridlinesVisible(true);
        plot.setRangeMinorGridlinesVisible(true);

        XYLineAndShapeRenderer renderer
            = (XYLineAndShapeRenderer) plot.getRenderer();
        renderer.setSeriesOutlinePaint(0, Color.black);
        renderer.setUseOutlinePaint(true);
        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setAutoRangeIncludesZero(false);
        domainAxis.setRange(axisMin, axisMax);

        domainAxis.setTickMarkInsideLength(2.0f);
        domainAxis.setTickMarkOutsideLength(2.0f);

        domainAxis.setMinorTickCount(2);
        domainAxis.setMinorTickMarksVisible(true);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setTickMarkInsideLength(2.0f);
        rangeAxis.setTickMarkOutsideLength(2.0f);
        rangeAxis.setMinorTickCount(2);
        rangeAxis.setMinorTickMarksVisible(true);
        rangeAxis.setRange(axisMin, axisMax);
        return chart;
    }


    public static class MNISTLatentSpaceVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private int plotFrequency;
        private int gridWidth;

        public MNISTLatentSpaceVisualizer(double imageScale, List<INDArray> digits, int plotFrequency) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.plotFrequency = plotFrequency;
            this.gridWidth = (int)Math.sqrt(digits.get(0).size(0)); //Assume square, nxn rows
        }

        private String getTitle(int recordNumber){
            return "Reconstructions Over Latent Space at Training Iteration " + recordNumber * plotFrequency;
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(getTitle(0));
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            JPanel panel = new JPanel();
            panel.setLayout(new GridLayout(0,gridWidth));

            JSlider slider = new JSlider(0,digits.size()-1, 0);
            slider.addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent e) {
                    JSlider slider = (JSlider)e.getSource();
                    int  value = slider.getValue();
                    panel.removeAll();
                    List<JLabel> list = getComponents(value);
                    for(JLabel image : list){
                        panel.add(image);
                    }
                    frame.setTitle(getTitle(value));
                    frame.revalidate();
                }
            });
            frame.add(slider, BorderLayout.NORTH);


            List<JLabel> list = getComponents(0);
            for(JLabel image : list){
                panel.add(image);
            }

            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
            frame.pack();
        }

        private List<JLabel> getComponents(int idx){
            List<JLabel> images = new ArrayList<>();
            List<INDArray> temp =  new ArrayList<>();
            for( int i=0; i<digits.get(idx).size(0); i++ ){
                temp.add(digits.get(idx).getRow(i));
            }
            for( INDArray arr : temp ){
                BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<784; i++ ){
                    bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*arr.getDouble(i)));
                }
                ImageIcon orig = new ImageIcon(bi);
                Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale*28),(int)(imageScale*28),Image.SCALE_REPLICATE);
                ImageIcon scaled = new ImageIcon(imageScaled);
                images.add(new JLabel(scaled));
            }
            return images;
        }
    }
}

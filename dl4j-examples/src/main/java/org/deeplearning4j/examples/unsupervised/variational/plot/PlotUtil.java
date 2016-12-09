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

/**Simple plotting methods for the MLPClassifier examples
 * @author Alex Black
 */
public class PlotUtil {



    /**Plot the training data. Assume 2d input, classification output
//     * @param features Training data features
     * @param labels Training data labels (one-hot representation)
     */
    public static void plotData(List<INDArray> xyVsEpoch, INDArray labels){

        JPanel panel = new ChartPanel(createChart(xyVsEpoch.get(0), labels));
        JPanel p2 = new JPanel();


        JSlider slider = new JSlider(0,xyVsEpoch.size()-1,0);
        slider.setMajorTickSpacing(1);
        slider.setPaintLabels(true);
        slider.setSnapToTicks(true);
        p2.add(slider);

        final JFrame f = new JFrame();
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider slider = (JSlider)e.getSource();

                int  value = slider.getValue();

                JPanel panel = new ChartPanel(createChart(xyVsEpoch.get(value), labels));
                f.add(panel, BorderLayout.CENTER);
                f.revalidate();
            }
        });



        f.setLayout(new BorderLayout());
        f.add(p2, BorderLayout.NORTH);
        f.add(panel, BorderLayout.CENTER);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Latent Space");

        f.setVisible(true);

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

    private static JFreeChart createChart(INDArray features, INDArray labels) {

        XYDataset dataset = createDataSet(features, labels);

        JFreeChart chart = ChartFactory.createScatterPlot("Scatter Plot Demo 1",
            "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
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
        domainAxis.setRange(-1, 1);

        domainAxis.setTickMarkInsideLength(2.0f);
        domainAxis.setTickMarkOutsideLength(2.0f);

        domainAxis.setMinorTickCount(2);
        domainAxis.setMinorTickMarksVisible(true);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setTickMarkInsideLength(2.0f);
        rangeAxis.setTickMarkOutsideLength(2.0f);
        rangeAxis.setMinorTickCount(2);
        rangeAxis.setMinorTickMarksVisible(true);
        rangeAxis.setRange(-1,1);
        return chart;
    }


    public static class MNISTLatentSpaceVisualizer {
        private double imageScale;
        private List<INDArray> digits;  //Digits (as row vectors), one per INDArray
        private String title;
        private int gridWidth;

        public MNISTLatentSpaceVisualizer(double imageScale, List<INDArray> digits, String title) {
            this.imageScale = imageScale;
            this.digits = digits;
            this.title = title;
            this.gridWidth = (int)Math.sqrt(digits.get(0).size(0)); //Assume square, nxn rows
        }

        public void visualize(){
            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel panel = new JPanel();
            panel.setLayout(new GridLayout(0,gridWidth));

            List<JLabel> list = getComponents();
            for(JLabel image : list){
                panel.add(image);
            }

            frame.add(panel);
            frame.setVisible(true);
            frame.pack();
        }

        private List<JLabel> getComponents(){
            List<JLabel> images = new ArrayList<>();
            for( INDArray arr : digits ){
                BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
                for( int i=0; i<768; i++ ){
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

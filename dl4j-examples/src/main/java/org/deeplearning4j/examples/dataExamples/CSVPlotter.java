package org.deeplearning4j.examples.dataExamples;

import java.io.File;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Read a csv file and plot the data using Deeplearning4J. 
 * 
 * @author Robert Altena  
 */
public class CSVPlotter {
	public static void main( String[] args ) throws IOException, InterruptedException
    {
    	String filename = "src/main/resources/DataExamples/CSVplotData.csv";
    	DataSet ds =  ReadCSVDataset(filename);
    	PlotDataset(ds);
    }
    
    /**
     * Read a CSV file into a dataset.
     */
	public static DataSet ReadCSVDataset(String filename) throws IOException, InterruptedException{
		int batchSize = 1000;
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(filename)));
    	DataSetIterator Iter  = new RecordReaderDataSetIterator(rr,batchSize);
		
    	return Iter.next(); 
	}
	
	/**
	 * Generate an xy plot of the dataset.
	 */
	static void PlotDataset(DataSet ds ){
		
		INDArray features = ds.getFeatures();
    	int nRows = features.rows(); 
    	XYSeries series = new XYSeries("S1");
    	for( int i=0; i<nRows; i++ ){
    	     series.add(features.getDouble(i, 0), features.getDouble(i, 1));
    	 }
    	
    	XYSeriesCollection c = new XYSeriesCollection();
   	    c.addSeries(series);
		XYDataset dataset = c;
		
		String title = "title";
		String xAxisLabel = "xAxisLabel";
		String yAxisLabel = "yAxisLabel";	
		PlotOrientation orientation = PlotOrientation.VERTICAL;
		boolean legend = false;
		boolean tooltips = false;
		boolean urls = false;
		JFreeChart chart =ChartFactory.createScatterPlot(title , xAxisLabel, yAxisLabel, dataset , orientation , legend , tooltips , urls);
    	JPanel panel = new ChartPanel(chart);
    	
    	 JFrame f = new JFrame();
    	 f.add(panel);
    	 f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
         f.pack();
         f.setTitle("Training Data");

         f.setVisible(true);
	}
}

package org.datavec.transform.timeseries.spark.convert;

import java.util.List;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class CSVSingleLineToSequenceDataSet implements Function<String, DataSet> {

    private RecordReader recordReader;
    private int labelIdx;
    private int numOutComes;

    public CSVSingleLineToSequenceDataSet(RecordReader recordReader, int labelIdx, int numOutComes){
        this.labelIdx = labelIdx;
        this.numOutComes = numOutComes;
        this.recordReader = recordReader;
    }
    
    /**
     * This data set function is meant to be called on a single line of an input file where each line is a separate timeseries
     * 
     * 
     * 
     * 
     */
    @Override
    public DataSet call(String s) throws Exception {


        recordReader.initialize(new StringSplit(s));
        List<Writable> lw = recordReader.next();
        
        
        int inputColumnCount = 1;
        int outputColumnCount = this.numOutComes;
        int maxTimestepLength = lw.size() - 1;
        
        //int labelClassIndex = 
        
		INDArray input  = Nd4j.zeros(new int[]{ 1, inputColumnCount, maxTimestepLength });
		INDArray labels = Nd4j.zeros(new int[]{ 1, outputColumnCount, maxTimestepLength });
		INDArray mask   = Nd4j.zeros(new int[]{ 1, maxTimestepLength });
		INDArray mask2 = Nd4j.zeros(new int[]{ 1, maxTimestepLength });

		int csvIndex = 0;
		
		// TODO: probably should account for a label index other that index 0...
		for ( int timestepIndex = 0; timestepIndex < maxTimestepLength; timestepIndex++ ) {
			
			// set the features
			input.putScalar( new int[]{ 0, 0, timestepIndex }, lw.get( csvIndex + 1 ).toDouble() );
			
			// set the mask
			mask.putScalar(new int[]{ 0, timestepIndex }, 1.0);

			// set the label for every timestep at the class column
			labels.putScalar(new int[]{ 0, lw.get( 0 ).toInt() - 1, timestepIndex }, 1.0);
			
			csvIndex++;
				
			
		}
		
		
		Nd4j.copy(mask, mask2);
		
		return new DataSet(input,labels, mask, mask2);
    }
}

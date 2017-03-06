package com.codor.alchemy.forecast;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Stack;
import java.util.StringTokenizer;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AdvisorDataSetIterator extends BaseDataSetIterator implements DataSetIterator {

	private static final long serialVersionUID = -8938692790057042187L;

	private int cursor = 0;
	private int vectorSize = 0;
	private int labelSize = 0;
	private int numRows;
	private int position = 0;
	private final int batchSize;

	private volatile String hdfsUrl;
	private List<String> advisors = new ArrayList<String>();
	private List<String> exclude = new ArrayList<String>();
	

	private List<Integer> featureInd = new ArrayList<Integer>();

	private static final Logger LOG = LoggerFactory.getLogger(AdvisorDataSetIterator.class);

	public AdvisorDataSetIterator(int batchSize, boolean train) {
		this(DATA_DIR, batchSize, train);
	}

	public AdvisorDataSetIterator(String dataDirectory, int batchSize, boolean train) {
		this(dataDirectory, batchSize, train, 71, 1);
	}

  	public AdvisorDataSetIterator(String dataDirectory, int batchSize, boolean train, int vectorSize, int labelSize) {
		super(HDFS_URL + dataDirectory + (train ? "/train" : "/test"));
		this.batchSize = batchSize;
		int pos = dataDirectory.lastIndexOf("/");
		dataDirectory = (pos > -1 ? dataDirectory.substring(0, pos) : dataDirectory);
		numRows = train ? 16 : 1;
		this.vectorSize = vectorSize;
		this.labelSize = labelSize;
	}

	public void setExclude(List<String> exclude) {
		this.exclude = exclude == null || exclude.isEmpty() ? new ArrayList<String>() : exclude;
	}

	@Override
	public boolean hasNext() {
		try {
			return hdfsIterator != null && hdfsIterator.hasNext();
		} catch (IOException e) {
			return false;
		}
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public List<String> getLabels() { // TODO
		return null;
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return null;
	}

	@Override
	public int inputColumns() { // TODO
		return 0;
	}

	@Override
	public DataSet next(int num) {
		try {
			if (!hdfsIterator.hasNext())
				throw new NoSuchElementException();
			return nextDataSet(num);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private boolean toExclude(Path path) {
		if (exclude == null || exclude.isEmpty()) return false;
		if (path == null) return true;
		String p = path.toUri().getPath();
		StringTokenizer pathTokens = new StringTokenizer(p, "/");
		String index = null;
		while (pathTokens.hasMoreTokens()) {
			index = pathTokens.nextToken();
		}
		return exclude.contains(index);
	} 

	private void populate(INDArray features, INDArray labels, Path path) throws IOException {
		Reader reader = new InputStreamReader(fs.open(path));
		LineIterator iter = IOUtils.lineIterator(reader);
		try {
			while (iter.hasNext()) {
				String content = iter.next();
				String[] tokens = iter.nextLine().split(",");
				int j = 0;
				int col = 0;
				for (; j < Math.min(vectorSize, tokens.length - 1); j++) {
					features.put(position, j, Double.valueOf(tokens[j].replace("\"", "").trim()).doubleValue());
				}
				for (; j < tokens.length && col < labelSize; j++) {
					labels.putScalar(position, col++, Double.valueOf(tokens[j].replace("\"", "").trim()).doubleValue());
				}
				position--;
			}
			cursor++;
		} catch (Exception e) {

		}
		iter.close();
	}

	private DataSet nextDataSet(int num) throws IOException {
		List<List<Double>> instances = new ArrayList<List<Double>>(num);
		List<Double> targets = new ArrayList<Double>();
		
		INDArray labels = Nd4j.create(numRows, labelSize); // one class
		INDArray features = Nd4j.create(numRows, vectorSize);
		position = numRows - 1;
		
		Stack<Path> stack = new Stack<Path>();
		String previousPath = "";

		for (int i = 0; i < num && hdfsIterator.hasNext(); i++) {
			
			LocatedFileStatus next = hdfsIterator.next();
			Path path = next.getPath();
			
			String currentPath = path.toUri().getPath();
			String index = getRelativeFilename(currentPath);

			if (previousPath.contains(index.split("_")[0])) {
				if (i >= num || !hdfsIterator.hasNext()) {
					String p = stack.peek() == null ? "" : stack.peek().toUri().toString();
					if (p.contains(index.split("_")[0])) {
						stack.push(path);
						while (!stack.isEmpty()) {
							populate(features, labels, stack.pop());
						}
					} else {
						labels = Nd4j.create(labelSize);//(numRows, labelSize);
						features = Nd4j.create(vectorSize);//(numRows, vectorSize);
						populate(features, labels, path);
					}
				} else {
					stack.push(path);
				}			
				previousPath = currentPath;
			} else {
				labels = Nd4j.create(labelSize);//(numRows, labelSize);
				features = Nd4j.create(vectorSize);//(numRows, vectorSize);
				while (!stack.isEmpty()) {
					populate(features, labels, stack.pop());
				}
				previousPath = currentPath;
				stack.push(path);
			}
		}
		return new DataSet(features, labels);//, null, labelsMask);
	}

	@Override
	public int numExamples() { // TODO
		return 0;
	}

	@Override
	public void reset() {
		try {
			hdfsIterator = fs.listFiles(new Path(hdfsUrl), true);
		} catch (IllegalArgumentException | IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public boolean resetSupported() { // TODO
		return false;
	}
	
	public List<String> getAdvisors() {
		return advisors;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor processor) {
	}

	@Override
	public int totalExamples() { // TODO
		return 0;
	}

	@Override
	public int totalOutcomes() { // TODO
		return labelSize;
	}
	
	public void remove() {
		
	}

}

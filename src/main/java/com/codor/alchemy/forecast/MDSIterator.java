package com.codor.alchemy.forecast;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Stack;

import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import com.codor.alchemy.conf.Constants;

public class MDSIterator extends BaseDataSetIterator implements MultiDataSetIterator {

	private int vectorSize = 0;
	private int labelSize = 0;
	private final int batchSize;
	private int numSteps = 6;
	private Stack<Path> stack = new Stack<Path>();

	private StackSequenceRecordReader ssRecordReader;

	private static final long serialVersionUID = -2132071188514707198L;

	public MDSIterator(int batchSize, double holdout) {
		this(DATA_DIR, batchSize, holdout);
	}

	public MDSIterator(String dataDirectory, int batchSize, double holdout) {
		this(dataDirectory, batchSize, 86, 1);
	}

	public MDSIterator(String dataDirectory, int batchSize, int vectorSize, int labelSize) {
		this(dataDirectory, batchSize, vectorSize, labelSize, 
			Constants.END_SEQ() - Constants.START_SEQ() + 1, 0);
	}

	public MDSIterator(String dataDirectory, int batchSize, int vectorSize, int labelSize,
			int numSteps, int flag) {
		super(HDFS_URL + dataDirectory + (flag == 0 ? "/train" : flag == 1 ? "/test" : "/predict"));
		this.batchSize = batchSize;
		int pos = dataDirectory.lastIndexOf("/");
		dataDirectory = (pos > -1 ? dataDirectory.substring(0, pos) : dataDirectory);
		String folder = flag == 0 ? "/train" : flag == 1 ? "/test" : "/predict"; 
		this.hdfsUrl = HDFS_URL + dataDirectory + folder;
		this.vectorSize = vectorSize;
		this.labelSize = labelSize;
		int start = Constants.START_SEQ();
		int end = Constants.END_SEQ();
		start = flag == 2 ? start + 1 : start; // TODO
		end = flag == 2 ? end : end - 1; // TODO
		ssRecordReader = new StackSequenceRecordReader(fs, start, end);
		this.numSteps = numSteps;
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
	public MultiDataSet next() {
		return next(batchSize);
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public MultiDataSet next(int num) {
		try {
			if (!hdfsIterator.hasNext())
				throw new NoSuchElementException();
			MultiDataSet mds = nextMultiDataSet(num);
			//double random = Math.random();
			//if (random > holdout && hdfsIterator.hasNext()) {
			//	mds = next(num);
			//}
			while(mds == null && hdfsIterator.hasNext()) {
				mds = nextMultiDataSet(num);
			}
			return mds;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private void pushAndClear(Path path, String index) {
		String p = stack.isEmpty() ? "" : stack.peek().toUri().toString();
		if (p.contains(index.split("_")[0])) {
                	stack.push(path);
                } else {
                	ssRecordReader.newRecord(stack);
                	stack.push(path);
                }
                ssRecordReader.newRecord(stack);
	}

	private MultiDataSet nextMultiDataSet(int num) throws IOException {
		String previousPath = stack.isEmpty() ? "" : stack.peek().toUri().getPath();

		for (int i = 0; i < num && hdfsIterator.hasNext(); i++) {
			for (int j = 0; j < numSteps; j++) {
				if (!hdfsIterator.hasNext()) break;
				LocatedFileStatus next = hdfsIterator.next();
				Path path = next.getPath();

				String currentPath = path.toUri().getPath();
				String index = getRelativeFilename(currentPath);

				if (previousPath.contains(index.split("_")[0])) {
					if (j >= numSteps - 1 || !hdfsIterator.hasNext()) {
						pushAndClear(path, index);
					} else {
						stack.push(path);
					}
					previousPath = currentPath;
				} else {
					if (j >= numSteps - 1 || !hdfsIterator.hasNext()) {
						pushAndClear(path, index);
					}
					ssRecordReader.newRecord(stack);
					stack.push(path);
					if (!previousPath.isEmpty()) {
						break;
					}
					previousPath = currentPath;
				}
			}
		}
		return ssRecordReader.toMultiDataSet(vectorSize, labelSize);
	}

	@Override
	public void reset() {
		initialize();
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preprocessor) {

	}
	
	@Override
    	public void remove() {
        	throw new UnsupportedOperationException("Remove not yet supported");
    	}
}


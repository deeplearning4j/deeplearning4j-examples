package com.codor.alchemy.forecast;

import java.io.IOException;
import java.io.Serializable;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerFlatMap;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerMultiDataSetFlatMap;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerPathFlatMap;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerPathMDSFlatMap;
import org.deeplearning4j.spark.data.BatchAndExportDataSetsFunction;
import org.deeplearning4j.spark.data.BatchAndExportMultiDataSetsFunction;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingResult;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingAggregationTuple;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementAddFunction;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementCombineFunction;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author: Ousmane A. Dia
 */
public class BaseParameterAveragingTrainingMaster extends ParameterAveragingTrainingMaster implements Serializable {

   	private static final long serialVersionUID = 8577088657995704083L;
	private static final Logger log = LoggerFactory.getLogger(ParameterAveragingTrainingMaster.class);

	private int batchSizePerWorker;
	protected int lastExportedRDDId = Integer.MIN_VALUE;
	protected String lastRDDExportPath;
	private Integer numWorkers;
	private RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
	private StorageLevel storageLevel;
	private int rddDataSetNumExamples;
	
	private RepartitionStrategy repartitionStrategy;
	private Repartition repartition;
	private StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
	private int averagingFrequency;
	private Random rng;

   	public BaseParameterAveragingTrainingMaster() {
		this(true, 3, 1, 10, 5, 0);
	}
	
	public BaseParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples,
			int batchSizePerWorker, int averagingFrequency, int prefetchNumBatches) {
		super(saveUpdater, 3, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency,
				prefetchNumBatches, Repartition.Always, RepartitionStrategy.Balanced, 
				StorageLevel.MEMORY_ONLY_SER(), false);
		this.batchSizePerWorker = batchSizePerWorker;
		this.numWorkers = numWorkers;
		this.storageLevel = StorageLevel.MEMORY_ONLY_SER();
		this.rddDataSetNumExamples = rddDataSetNumExamples;
		this.repartitionStrategy = RepartitionStrategy.Balanced;
		this.repartition = Repartition.Always;
		this.rng = new Random();
		this.averagingFrequency = averagingFrequency;
	}

	public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        	if (super.getNumWorkers() == null) super.setNumWorkers(network.getSparkContext().defaultParallelism());

        	if (super.getRddTrainingApproach() == RDDTrainingApproach.Direct) {
            		executeTrainingDirect(network, trainingData);
       		} else {
            	//Export data if required (or, use cached export)
            	JavaRDD<String> paths = exportIfRequired(network.getSparkContext(), trainingData);
            	executeTrainingPathsHelper(network, paths, super.getBatchSizePerWorker());     
	    	//Originally (pre-export): had rddDataSetNumExamples per DataSet. Now we have batchSizePerWorker per exported DataSet
        	}
    	}
	

	public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
		if (numWorkers == null)
			numWorkers = graph.getSparkContext().defaultParallelism();
		if (rddTrainingApproach == RDDTrainingApproach.Direct) {

			if (storageLevel != null)
				trainingData.persist(storageLevel);
			long totalDataSetObjectCount = trainingData.count();
			int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

			JavaRDD<MultiDataSet>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount,
					dataSetObjectsPerSplit, trainingData, rng.nextLong());
			int splitNum = 1;
			for (JavaRDD<MultiDataSet> split : splits) {

				JavaRDD<MultiDataSet> splitData = split;

				splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
						numObjectsEachWorker(rddDataSetNumExamples), numWorkers);

				FlatMapFunction<Iterator<MultiDataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerMultiDataSetFlatMap<>(
						getWorkerInstance(graph));
				JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
				processResults(null, graph, result, splitNum++, splits.length);
			}
		} else {
			JavaRDD<String> paths = exportIfRequiredMDS(graph.getSparkContext(), trainingData);
			executeTrainingPathsMDSHelper(graph, paths, batchSizePerWorker);
		}		
	}	


	private void executeTrainingPathsMDSHelper(SparkComputationGraph network, JavaRDD<String> trainingMultiDataPaths,
			int dataSetObjectsNumExamples) {
		if (numWorkers == null)
			numWorkers = network.getSparkContext().defaultParallelism();

		if (storageLevelStreams != null)
			trainingMultiDataPaths.persist(storageLevelStreams);

		long totalDataSetObjectCount = trainingMultiDataPaths.count();

		int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(dataSetObjectsNumExamples);
		JavaRDD<String>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit,
				trainingMultiDataPaths, rng.nextLong());

		int splitNum = 1;
		for (JavaRDD<String> split : splits) {
			doIterationPathsMDS(network, split, splitNum++, splits.length, dataSetObjectsNumExamples);
		}
	}


	private void doIterationPathsMDS(SparkComputationGraph graph, JavaRDD<String> split, int splitNum, int numSplits,
			int dataSetObjectNumExamples) {
		log.info(
				"Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
				splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);

		JavaRDD<String> splitData = split;
		splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
				numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);

		FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function = new ExecuteWorkerPathMDSFlatMap<>(
				getWorkerInstance(graph));

		JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
		processResults(null, graph, result, splitNum, numSplits);
	}

	private JavaRDD<String> exportIfRequiredMDS(JavaSparkContext sc, JavaRDD<MultiDataSet> trainingData) {;

        int currentRDDUid = trainingData.id();   

        String baseDir;
        if (lastExportedRDDId == Integer.MIN_VALUE) {
            baseDir = exportMDS(trainingData);
        } else {
            if (lastExportedRDDId == currentRDDUid) {
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                deleteTempDir(sc, lastRDDExportPath);
                baseDir = exportMDS(trainingData);
            }
        }

        return sc.textFile(baseDir + "paths/");
    	}

	
	private String exportMDS(JavaRDD<MultiDataSet> trainingData) {
        String baseDir = getBaseDirForRDD(trainingData);
        String dataDir = baseDir + "data/";
        String pathsDir = baseDir + "paths/";

        log.info("Initiating RDD<MultiDataSet> export at {}", baseDir);
        JavaRDD<String> paths = trainingData.mapPartitionsWithIndex(new BatchAndExportMultiDataSetsFunction(batchSizePerWorker, dataDir), true);
        paths.saveAsTextFile(pathsDir);
        log.info("RDD<MultiDataSet> export complete at {}", baseDir);

        lastExportedRDDId = trainingData.id();
        lastRDDExportPath = baseDir;
        return baseDir;
    	}

	

	private void executeTrainingPathsHelper(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths, int dataSetObjectsNumExamples){
        if (getNumWorkers() == null) setNumWorkers(network.getSparkContext().defaultParallelism());
        boolean collectTrainingStats = getIsCollectTrainingStats();
        
        if (collectTrainingStats) getStats().logFitStart();
        if (getStorageLevelStreams() != null) trainingDataPaths.persist(getStorageLevelStreams());

        if (collectTrainingStats) getStats().logCountStart();
        long totalDataSetObjectCount = trainingDataPaths.count();
        if (collectTrainingStats) getStats().logCountEnd();

        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(dataSetObjectsNumExamples);
        if (collectTrainingStats) getStats().logSplitStart();
        JavaRDD<String>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingDataPaths, getRng().nextLong());
        if (collectTrainingStats) getStats().logSplitEnd();


        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPaths(network, null, split, splitNum++, splits.length, dataSetObjectsNumExamples);
        }

        if (collectTrainingStats) getStats().logFitEnd((int) totalDataSetObjectCount);
    }
	
	private void doIterationPaths(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<String> split,
			int splitNum, int numSplits, int dataSetObjectNumExamples) {
		log.info(
				"Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
				splitNum, numSplits, getBatchSizePerWorker(), getAveragingFrequency(), getNumWorkers());
		boolean collectTrainingStats = getIsCollectTrainingStats();
		
		if (collectTrainingStats)
			getStats().logMapPartitionsStart();

		JavaRDD<String> splitData = split;
		if (collectTrainingStats)
			getStats().logRepartitionStart();
		splitData = SparkUtils.repartition(splitData, getRepartition(), getRepartitionStrategy(),
				numObjectsEachWorker(dataSetObjectNumExamples), getNumWorkers());
		int nPartitions = splitData.partitions().size();
		if (collectTrainingStats && getRepartition() != Repartition.Never)
			getStats().logRepartitionEnd();

		FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function;
		if (network != null)
			function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(network));
		else
			function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(graph));

		JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
		processResults(network, graph, result, splitNum, numSplits);

		if (collectTrainingStats)
			getStats().logMapPartitionsEnd(nPartitions);
	}
	
	
	private void executeTrainingDirect(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if (super.getIsCollectTrainingStats()) super.getStats().logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        boolean collectTrainingStats = getIsCollectTrainingStats();
        
        if (getStorageLevel() != null) trainingData.persist(getStorageLevel());

        if (collectTrainingStats) super.getStats().logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) getStats().logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(getRddDataSetNumExamples());

        if (collectTrainingStats) super.getStats().logSplitStart();
        JavaRDD<DataSet>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, getRng().nextLong());
        if (collectTrainingStats) super.getStats().logSplitEnd();

        int splitNum = 1;
        for (JavaRDD<DataSet> split : splits) {
            doIteration(network, split, splitNum++, splits.length);
        }

        if (collectTrainingStats) super.getStats().logFitEnd((int) totalDataSetObjectCount);
    }
	
	private int getNumDataSetObjectsPerSplit(int numExamplesEachRddObject) {
        int dataSetObjectsPerSplit;
        if (numExamplesEachRddObject == 1) {
            dataSetObjectsPerSplit = getNumWorkers() * getBatchSizePerWorker() * getAveragingFrequency();
        } else {
            int numDataSetObjsReqEachWorker = numObjectsEachWorker(numExamplesEachRddObject);
            if (numDataSetObjsReqEachWorker < 1) {
                //In this case: more examples in a DataSet object than we actually require
                //For example, 100 examples in DataSet, with batchSizePerWorker=50 and averagingFrequency=1
                numDataSetObjsReqEachWorker = 1;
            }

            dataSetObjectsPerSplit = numDataSetObjsReqEachWorker * getNumWorkers();
        }
        return dataSetObjectsPerSplit;
    }
	
	private int numObjectsEachWorker(int numExamplesEachRddObject) {
        return getBatchSizePerWorker() * getAveragingFrequency() / numExamplesEachRddObject;
    }
	
	private void doIteration(SparkDl4jMultiLayer network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
		boolean collectTrainingStats = getIsCollectTrainingStats();
		
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, getBatchSizePerWorker(), getAveragingFrequency(), getNumWorkers());
        if (collectTrainingStats) getStats().logMapPartitionsStart();

        JavaRDD<DataSet> splitData = split;
        if (collectTrainingStats) getStats().logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, getRepartition(), getRepartitionStrategy(), 
        		numObjectsEachWorker(getRddDataSetNumExamples()), getNumWorkers());
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && getRepartition() != Repartition.Never) getStats().logRepartitionEnd();


        FlatMapFunction<Iterator<DataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerFlatMap<>(getWorkerInstance(network));
        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, null, result, splitNum, numSplits);

        if (collectTrainingStats) getStats().logMapPartitionsEnd(nPartitions);
    }
	
	private void processResults(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<ParameterAveragingTrainingResult> results, int splitNum, int totalSplits) {
        //Need to do parameter averaging, and where necessary also do averaging of the updaters
        //Let's do all of this in ONE step, such that we don't have extra synchronization costs
		boolean collectTrainingStats = getIsCollectTrainingStats();
        if (collectTrainingStats) getStats().logAggregateStartTime();
        ParameterAveragingAggregationTuple tuple = results.aggregate(null,
                new ParameterAveragingElementAddFunction(),
                new ParameterAveragingElementCombineFunction());
        INDArray params = tuple.getParametersSum();
        int aggCount = tuple.getAggregationsCount();
        SparkTrainingStats aggregatedStats = tuple.getSparkTrainingStats();
        if (collectTrainingStats) getStats().logAggregationEndTime();


        if (collectTrainingStats) getStats().logProcessParamsUpdaterStart();
        params.divi(aggCount);
        INDArray updaterState = tuple.getUpdaterStateSum();
        if (updaterState != null) updaterState.divi(aggCount);   //May be null if all SGD updaters, for example

        if (network != null) {
            MultiLayerNetwork net = network.getNetwork();
            net.setParameters(params);
            if (updaterState != null) net.getUpdater().setStateViewArray(null, updaterState, false);

            network.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
        } else {
            ComputationGraph g = graph.getNetwork();
            g.setParams(params);
            if (updaterState != null) g.getUpdater().setStateViewArray(updaterState);

            graph.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
        }

        if (collectTrainingStats) {
        	getStats().logProcessParamsUpdaterEnd();
        	getStats().addWorkerStats(aggregatedStats);
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        log.info("Completed training of split {} of {}", splitNum, totalSplits);

        if (getListeners() != null) {
            if (network != null) {
                MultiLayerNetwork net = network.getNetwork();
                net.setScore(network.getScore());
                for (IterationListener il : getListeners()) {
                    il.iterationDone(net, getIterationCount());
                }
            } else {
                ComputationGraph g = graph.getNetwork();
                g.setScore(graph.getScore());
                for (IterationListener il : getListeners()) {
                    il.iterationDone(g, getIterationCount());
                }
            }
        }

        setIterationCount(getIterationCount() + 1);
    }
	
	private JavaRDD<String> exportIfRequired(JavaSparkContext sc, JavaRDD<DataSet> trainingData) {
		
		boolean collectTrainingStats = getIsCollectTrainingStats();
		
        if (collectTrainingStats) getStats().logExportStart();

        //Two possibilities here:
        // 1. We've seen this RDD before (i.e., multiple epochs training case)
        // 2. We have not seen this RDD before
        //    (a) And we havent got any stored data -> simply export
        //    (b) And we previously exported some data from a different RDD -> delete the last data
        int currentRDDUid = trainingData.id();       //Id is a "A unique ID for this RDD (within its SparkContext)."

        String baseDir;
        if (getLastExportedRDDId() == Integer.MIN_VALUE) {
            //Haven't seen a RDD<DataSet> yet in this training master -> export data
            baseDir = export(trainingData);
        } else {
            if (getLastExportedRDDId() == currentRDDUid) {
                //Use the already-exported data again for another epoch
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                //The new RDD is different to the last one
                // Clean up the data for the last one, and export
                deleteTempDir(sc, getLastRDDExportPath());
                baseDir = export(trainingData);
            }
        }

        if (collectTrainingStats) getStats().logExportEnd();

        return sc.textFile(baseDir + "paths/");
    }
	
	private String export(JavaRDD<DataSet> trainingData) {
		String baseDir = getBaseDirForRDD(trainingData);
		String dataDir = baseDir + "data/";
		String pathsDir = baseDir + "paths/";

		log.info("Initiating RDD<DataSet> export at {}", baseDir);
		JavaRDD<String> paths = trainingData
				.mapPartitionsWithIndex(new BatchAndExportDataSetsFunction(getBatchSizePerWorker(), dataDir), true);
		paths.saveAsTextFile(pathsDir);
		log.info("RDD<DataSet> export complete at {}", baseDir);

		setLastExportedRDDId(trainingData.id());
		setLastRDDExportPath(baseDir);
		return baseDir;
	}

	private String getBaseDirForRDD(JavaRDD<?> rdd) {
		if (getExportDirectory() == null) {
			setExportDirectory(getDefaultExportDirectory(rdd.context()));
		}
	
		String exportDirectory = "/user/odia/mackenzie/dl4j/";
		return exportDirectory + (exportDirectory.endsWith("/") ? "" : "/") + getTrainingMasterUID() + "/"
				+ rdd.id() + "/";
		//return getExportDirectory() + (getExportDirectory().endsWith("/") ? "" : "/") + getTrainingMasterUID() + "/"
		//		+ rdd.id() + "/";
	}
	
	private String getDefaultExportDirectory(SparkContext sc) {
        //String hadoopTmpDir = sc.hadoopConfiguration().get("hadoop.tmp.dir");
        //if (!hadoopTmpDir.endsWith("/") && !hadoopTmpDir.endsWith("\\")) hadoopTmpDir = hadoopTmpDir + "/";
        return "/user/odia/mackenzie/dl4j"; //"/user/hduser/dl4j/";
    }
	
	private boolean deleteTempDir(JavaSparkContext sc, String tempDirPath) {
		log.info("Attempting to delete temporary directory: {}", tempDirPath);

		Configuration hadoopConfiguration = sc.hadoopConfiguration();
		FileSystem fileSystem;
		try {
			fileSystem = FileSystem.get(new URI(tempDirPath), hadoopConfiguration);
		} catch (URISyntaxException | IOException e) {
			throw new RuntimeException(e);
		}

		try {
			fileSystem.delete(new Path(tempDirPath), true);
			log.info("Deleted temporary directory: {}", tempDirPath);
			return true;
		} catch (IOException e) {
			log.warn("Could not delete temporary directory: {}", tempDirPath, e);
			return false;
		}
	}
	
	public static class Builder extends ParameterAveragingTrainingMaster.Builder {
		
        public BaseParameterAveragingTrainingMaster build() {
            return new BaseParameterAveragingTrainingMaster(this);
        }
        
        private boolean saveUpdater;
        private Integer numWorkers;
        private int rddDataSetNumExamples;
        private int batchSizePerWorker = 16;
        private int averagingFrequency = 5;
        private int prefetchNumBatches = 0;
        
        private Repartition repartition = Repartition.Always;
        private RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        private StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        private StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        private RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
        private String exportDirectory = "/user/odia/mackenzie/dl4j/"; //null;

        public Builder(int rddDataSetNumExamples) {
            this(null, rddDataSetNumExamples);
	    
            super.storageLevel(storageLevel);
            super.repartionData(repartition);
            super.storageLevelStreams(storageLevelStreams);
            super.rddTrainingApproach(rddTrainingApproach);
            super.exportDirectory(exportDirectory);
        }
        
        public Builder(Integer numWorkers, int rddDataSetNumExamples) {
            super(numWorkers, rddDataSetNumExamples);
            this.numWorkers = numWorkers;
            this.rddDataSetNumExamples = rddDataSetNumExamples;
        }

        public Builder batchSizePerWorker(int batchSizePerWorker) {
            this.batchSizePerWorker = batchSizePerWorker;
	    super.batchSizePerWorker(batchSizePerWorker);
            return this;
        }

        public Builder averagingFrequency(int averagingFrequency) {
            if (averagingFrequency <= 0)
                throw new IllegalArgumentException("Ivalid input: averaging frequency must be >= 1");
            this.averagingFrequency = averagingFrequency;
            super.averagingFrequency(averagingFrequency);
            return this;
        }

        public Builder workerPrefetchNumBatches(int prefetchNumBatches) {
            this.prefetchNumBatches = prefetchNumBatches;
	    super.workerPrefetchNumBatches(prefetchNumBatches);
            return this;
        }

        public Builder saveUpdater(boolean saveUpdater) {
            this.saveUpdater = saveUpdater;
            super.saveUpdater(saveUpdater);
	    return this;
        }

	}
	
	private BaseParameterAveragingTrainingMaster(Builder builder) {
		this(builder.saveUpdater, builder.numWorkers, builder.rddDataSetNumExamples, builder.batchSizePerWorker,
				builder.averagingFrequency, builder.prefetchNumBatches);
	}

}


package org.deeplearning4j.examples.convolution;

import org.apache.commons.lang3.RandomStringUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Collections;

public class MainSimpleExampleExtern {

    public static void main(String[] args) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

        Nd4j.create(1);

        CudaEnvironment.getInstance().getConfiguration()
            .setMaximumDeviceCache((5L * 1024 * 1024 * 1024L));


        Nd4j.getMemoryManager().togglePeriodicGc(false);


        for(int i=0; i<1000; i++){
            System.out.println("=========================");
            System.out.println("============ Training Loop " + i + " =============");
            System.out.println("=========================");
            training();
        }
    }

    private static void training(){
        //=========================================
        //============== Settings ==============
        //=========================================
        boolean outputAndScoreBeforeTraining = false;

        // ========================================

        //Not important Settings...
        //DataSources
        boolean createDataGarbage = true;

        //Iterator-Style
        boolean useCustomAsync = false;
        boolean useExternalIterator = false;
        boolean usePredefinedDataSets = true;

        boolean useBatchNorm = true;

        int hiddenLayerXCount = 0;


        //=========================================
        //=========================================
        //=========================================




        int epochs = 500;

        int featureCount = 150;
        int classCount = 5;

        double randomFactorDataCount = 0;
        double randomFactorBatchSize = 0.0;

        int trainDataCount = 70000;
        int valDataCount = 1500;
        int testDataCount = 1500;

//        int batchSize = 2048;
        int batchSize = 4096;



        ArrayList<GarbageObject> dummyList = new ArrayList<>();
        if(createDataGarbage){
            System.out.println("======= Create GargabeData ========");
            dummyList = getGarbageObjectList(1000, 1000);         //1000*1000 = 1Mio Objects      => GPU: Ultra Slow below 20 samples/sec
        }

        System.out.println("======= Create Data ========");

        ArrayList<DataSet> predefinedDataSets = new ArrayList<>();
        DoublesDataSetIterator trainDataIterator = null;
        DataSet valDataSet;
        DataSet testDataSet;

        System.out.println("======= Create RANDOM Data ========");
        System.out.println("------- Train Data --------");
        ArrayList<Pair<double[], double[]>> trainDataList = getDataList(trainDataCount + (int)(randomFactorDataCount*Math.random()*(trainDataCount/2)), featureCount, classCount);
        trainDataIterator = new DoublesDataSetIterator(trainDataList, batchSize + (int)(randomFactorBatchSize*(Math.random()*(batchSize))) );

        System.out.println("------- Val Data --------");
        valDataSet = getDataSet(valDataCount + (int)(randomFactorDataCount*Math.random()*(valDataCount/2)), featureCount, classCount);

        System.out.println("------- Test Data --------");
        testDataSet = getDataSet(testDataCount + (int)(randomFactorDataCount*Math.random()*(testDataCount/2)), featureCount, classCount);


//        dummyList.clear();
//        dummyList = null;
//        System.gc();


        System.out.println("------- Normalize Data --------");
        long normStart = System.currentTimeMillis();

        NormalizerMinMaxScaler normalizerMinMaxScaler = new NormalizerMinMaxScaler(-1.0, 1.0);
        normalizerMinMaxScaler.fit(trainDataIterator);
        normalizerMinMaxScaler.transform(valDataSet);
        normalizerMinMaxScaler.transform(testDataSet);
        trainDataIterator.setPreProcessor(normalizerMinMaxScaler);

        if(usePredefinedDataSets && predefinedDataSets.isEmpty()){
            trainDataIterator.reset();
            while(trainDataIterator.hasNext()){
                DataSet predefinedDS = trainDataIterator.next();
                predefinedDS.shuffle();
                predefinedDataSets.add(predefinedDS);
            }
        }

        System.out.println("Normalize Time: " + (System.currentTimeMillis() - normStart)/1000 + "s");

        System.out.println("======= Create Network ========");
        int seed = 432012;
        int iterations = 1;

        NeuralNetConfiguration.Builder preBuilder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2);

        preBuilder.updater(Updater.ADADELTA).rho(0.95).epsilon(1e-5);



        ComputationGraphConfiguration.GraphBuilder graphBuilder = preBuilder
            .trainingWorkspaceMode(WorkspaceMode.NONE)
            .inferenceWorkspaceMode(WorkspaceMode.NONE)
            .miniBatch(true)
            .regularization(true)
            .l1(0.00001)
            .l2(0.0001)
            .useDropConnect(false)
            .momentum(0.8)
            .graphBuilder();



        int lastLayer = -1;
        int currentLayer = 0;

        String lastLayerName = null;

        graphBuilder.addInputs("inputs");

        if(useBatchNorm){
            //Input Layer
            DenseLayer inputLayer = new DenseLayer.Builder().nIn(featureCount).nOut((int)(featureCount*0.9))
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .dropOut(0.9)
                .build();

            BatchNormalization bnInputLayer = new BatchNormalization.Builder()
                .nOut((int) (featureCount * 0.9))
                .build();

            ActivationLayer activationInputLayer = new ActivationLayer.Builder()
                .activation(Activation.LEAKYRELU)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, inputLayer, "inputs");
            graphBuilder.addLayer("bn" + currentLayer, bnInputLayer, "dl" + currentLayer);
            graphBuilder.addLayer("ac" + currentLayer, activationInputLayer, "bn" + currentLayer);

            lastLayer++;
            currentLayer++;


            //BN: 1. HiddenLayer
            DenseLayer hiddenLayer1 = new DenseLayer.Builder().nIn((int)(featureCount*0.9)).nOut((int)(featureCount*0.8))
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .dropOut(0.8)
                .build();

            BatchNormalization bnLayer1 = new BatchNormalization.Builder()
                .nOut((int)(featureCount*0.8))
                .build();

            ActivationLayer activationLayer1 = new ActivationLayer.Builder()
                .activation(Activation.LEAKYRELU)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, hiddenLayer1, "ac" + lastLayer);
            graphBuilder.addLayer("bn" + currentLayer, bnLayer1, "dl" + currentLayer);
            graphBuilder.addLayer("ac" + currentLayer, activationLayer1, "bn" + currentLayer);

            lastLayer++;
            currentLayer++;

            // ======
            //x. HiddenLayer
            for(int i=0; i<hiddenLayerXCount; i++) {
                DenseLayer hiddenLayerX = new DenseLayer.Builder().nIn((int)(featureCount*0.8)).nOut((int)(featureCount*0.8))
                    .weightInit(WeightInit.RELU)
                    .activation(Activation.IDENTITY)
                    .dropOut(0.8)
                    .build();

                BatchNormalization bnLayerX = new BatchNormalization.Builder()
                    .nOut((int)(featureCount*0.8))
                    .build();

                ActivationLayer activationLayerX = new ActivationLayer.Builder()
                    .activation(Activation.LEAKYRELU)
                    .build();

                graphBuilder.addLayer("dl" + currentLayer, hiddenLayerX, "ac" + lastLayer);
                graphBuilder.addLayer("bn" + currentLayer, bnLayerX, "dl" + currentLayer);
                graphBuilder.addLayer("ac" + currentLayer, activationLayerX, "bn" + currentLayer);

                lastLayer++;
                currentLayer++;
            }
            // ======

            //BN: 2. HiddenLayer
            DenseLayer hiddenLayer2 = new DenseLayer.Builder().nIn((int)(featureCount*0.8)).nOut((int)(featureCount*0.7))
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .dropOut(0.8)
                .build();

            BatchNormalization bnLayer2 = new BatchNormalization.Builder()
                .nOut((int)(featureCount*0.7))
                .build();

            ActivationLayer activationLayer2 = new ActivationLayer.Builder()
                .activation(Activation.LEAKYRELU)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, hiddenLayer2, "ac" + lastLayer);
            graphBuilder.addLayer("bn" + currentLayer, bnLayer2, "dl" + currentLayer);
            graphBuilder.addLayer("ac" + currentLayer, activationLayer2, "bn" + currentLayer);

            lastLayerName = "ac" + currentLayer;
        }else{
            //Input Layer
            DenseLayer inputLayer = new DenseLayer.Builder().nIn(featureCount).nOut((int)(featureCount*0.9))
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .dropOut(0.95)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, inputLayer, "inputs");

            lastLayer++;
            currentLayer++;

            //1. HiddenLayer
            DenseLayer hiddenLayer1 = new DenseLayer.Builder().nIn((int)(featureCount*0.9)).nOut((int)(featureCount*0.8))
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .dropOut(0.8)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, hiddenLayer1, "dl" + lastLayer);

            lastLayer++;
            currentLayer++;

            // ======
            //x. HiddenLayer
            for(int i=0; i<hiddenLayerXCount; i++){
                DenseLayer hiddenLayerX = new DenseLayer.Builder().nIn((int)(featureCount*0.8)).nOut((int)(featureCount*0.8))
                    .weightInit(WeightInit.RELU)
                    .activation(Activation.LEAKYRELU)
                    .dropOut(0.8)
                    .build();

                graphBuilder.addLayer("dl" + currentLayer, hiddenLayerX, "dl" + lastLayer);

                lastLayer++;
                currentLayer++;
            }
            // ======


            //2. HiddenLayer
            DenseLayer hiddenLayer2 = new DenseLayer.Builder().nIn((int)(featureCount*0.8)).nOut((int)(featureCount*0.7))
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .dropOut(0.8)
                .build();

            graphBuilder.addLayer("dl" + currentLayer, hiddenLayer2, "dl" + lastLayer);

            lastLayerName = "dl" + currentLayer;
        }

        //ORDINAL - OutputLayer
        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
            .nIn((int)(featureCount*0.7))
            .nOut(classCount)
            .activation(Activation.SIGMOID)
            .build();

        graphBuilder.addLayer("output", outputLayer, lastLayerName);
        graphBuilder.setOutputs("output");

        graphBuilder.pretrain(false).backprop(true);

        ComputationGraphConfiguration conf = graphBuilder.build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        PerformanceListener performanceListener = new PerformanceListener(100);
        model.setListeners(performanceListener);



        System.out.println("======= Train Network ========");
        model.fit(valDataSet);

        if(outputAndScoreBeforeTraining){
            model.output(valDataSet.getFeatures());
            double valScore = model.score();

            model.output(testDataSet.getFeatures());
            double testScore = model.score();

            System.out.println("Val Score: " + valScore + " - Test Score: " + testScore + " - Scores equal: " + (valScore == testScore));
        }


        System.out.println("Model Params: " + model.numParams());


        for(int e=0; e<epochs; e++){
            //System.out.println("Train epoch: " + e);

            if(usePredefinedDataSets && predefinedDataSets != null && !predefinedDataSets.isEmpty()){
                //System.out.println("Train with predefined DataSets");

                Collections.shuffle(predefinedDataSets);

                //Preloaded DataSets
                for(DataSet ds:predefinedDataSets){
                    ds.shuffle();
                    model.fit(ds);
                }
            }else{
                //Iterator
                if(useExternalIterator){
                    trainDataIterator.reset();
                    while(trainDataIterator.hasNext()){
                        model.fit(trainDataIterator.next());
                    }
                }else if(useCustomAsync){
                    AsyncDataSetIterator asyncDataSetIterator = new AsyncDataSetIterator(trainDataIterator, 16);
                    model.fit(asyncDataSetIterator);
                }else{
                    model.fit(trainDataIterator);
                }
            }

            //Simulate testing
            //model.output(valDataSet.getFeatures());
            //double valScore = model.score();
            double valScore = model.score(valDataSet);

            //model.output(testDataSet.getFeatures());
            //double testScore = model.score();
            double testScore = model.score(testDataSet);

            System.out.println("Val Score: " + valScore + " - Test Score: " + testScore + " - Scores equal: " + (valScore == testScore));

            Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();

        }

        System.gc();


        //Ensure list is not garbadge collected
        if(dummyList != null){
            System.out.println("END - Length of GarbadgeArrayList: " + dummyList.size());
        }
    }


    // ==== Helper Functions ====


    private static ArrayList<Pair<double[], double[]>> getDataList(int dataCount, int featureCount, int labelCount){
        ArrayList<Pair<double[], double[]>> dataList = new ArrayList<>();
        for(int i=0; i<dataCount; i++){
            dataList.add(new Pair<>(getFeatureData(featureCount), getLabelData(labelCount)));
        }
        return dataList;
    }

    private static DataSet getDataSet(int dataCount, int featureCount, int labelCount){
        double[][] featureVectorArray = new double[dataCount][featureCount];
        double[][] labelVectorArray = new double[dataCount][labelCount];

        for(int i=0; i<dataCount; i++){
            featureVectorArray[i] = getFeatureData(featureCount);
            labelVectorArray[i] = getLabelData(labelCount);
        }

        return new DataSet(Nd4j.create(featureVectorArray), Nd4j.create(labelVectorArray));
    }

    private static double[] getFeatureData(int featureCount){
        double[] result = new double[featureCount];

        for(int i=0; i<result.length; i++){
            result[i] = (Math.random()*30)-10;
        }

        return result;
    }

    private static double[] getLabelData(int labelCount){
        double[] result = new double[labelCount];

        int classNumber = (int)(Math.random()*labelCount);

        for(int i=0; i<result.length; i++){
            if(i <= classNumber){
                result[i] = 1;
            }else{
                result[i] = 0;
            }
        }

        return result;
    }


    private static ArrayList<GarbageObject> getGarbageObjectList(int garbageObjectCount, int garbageSubObjectCount){
        ArrayList<GarbageObject> result = new ArrayList<>();

        for(int i=0; i<garbageObjectCount; i++){
            result.add(new GarbageObject(garbageSubObjectCount));
        }

        return result;
    }


    public static class GarbageObject{
        public int val1;
        public double val2;
        public String val3;
        public String val4;

        public ArrayList<GarbageSubObject> subObjects = new ArrayList<>();

        public GarbageObject(int subObjectCount){
            this.val1 = (int)Math.random()*100;
            this.val2 = Math.random();
            this.val3 = RandomStringUtils.random(10);
            this.val4 = RandomStringUtils.random(8);

            for(int i=0; i<subObjectCount; i++){
                this.subObjects.add(new GarbageSubObject());
            }
        }

    }

    public static class GarbageSubObject{
        public int val1;
        public double val2;
        public String val3;
        public String val4;

        public GarbageSubObject(){
            this.val1 = (int)Math.random()*100;
            this.val2 = Math.random();
            this.val3 = RandomStringUtils.random(10);
            this.val4 = RandomStringUtils.random(8);
        }
    }

}

package org.deeplearning4j.examples.deepwalk;


import net.lingala.zip4j.core.ZipFile;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.data.EdgeLineProcessor;
import org.deeplearning4j.graph.data.GraphLoader;
import org.deeplearning4j.graph.graph.Graph;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.deeplearning4j.graph.vertexfactory.StringVertexFactory;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


/**DeepWalk graph vectorization and classification example.
 * DeepWalk is a vector model similar to Word2Vec, except that it creates vectors for each vertex in a graph.
 * The graph could represent many things; in this particular example, the graph is a subset of a social network for
 * bloggers.
 * Specifically, each vertex in the graph represents a user, and the edges represent the friendship network.
 *
 * The goal of this example is as follows:
 * 1. Train a DeepWalk model on the graph. This gives a vector representation for each vertex.
 * 2. Use the DeepWalk vector representations as input to a multi-label classifier. In this case, we are attempting to
 *    predict whether a user is a member of each group; note that users may be members of multiple groups simultaneously.
 *
 * See:
 * DeepWalk: Online Learning of Social Representations
 * Perozzi, Al-Rfou, Skiena (2014)
 * http://arxiv.org/abs/1403.6652
 *
 * For the data set being used in this example, see:
 * http://socialcomputing.asu.edu/datasets/BlogCatalog3
 *
 * @author Alex Black
 */
public class DeepWalkGraphExample {

    private static String DATA_URL = "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip";
    private static String DATA_PATH = System.getProperty("java.io.tmpdir") + "/DL4JDeepWalkExample/";
    private static String EDGE_LIST_PATH = DATA_PATH + "BlogCatalog-dataset/data/edges.csv";
    private static String GROUP_DATA_PATH = DATA_PATH + "BlogCatalog-dataset/data/group-edges.csv";

    private static final int NUM_VERTICES = 10312;

    public static void main(String[] args) throws Exception {

        //First: download the data if required
        downloadData();

        //Load the graph:
        Graph<String,String> graph = loadGraph();
//        System.out.println(graph);

        //Configure the DeepWalk model:
        int vectorSize = 100;   //Each vertex in graph: represented by vector of length 100
        int windowSize = 10;
        int walkLength = 40;
        int numEpochsDeepWalk = 100; //Number of random walks started at each vertex in the graph used for training

        DeepWalk<String,String> deepWalk = new DeepWalk.Builder<String,String>()
                .vectorSize(vectorSize)
                .windowSize(windowSize)
                .learningRate(3e-3)
                .build();

        deepWalk.initialize(graph);

        for( int i=0; i<numEpochsDeepWalk; i++ ) {
            deepWalk.fit(graph, walkLength);
            System.out.println("----- Epoch " + i + " complete -----");
        }


        //Save the DeepWalk network
        //TODO

        //Get data for training:
        int minibatchSize = 50;
        int nEpochs = 30;
        DataSetIterator[] data = getTrainTestData(minibatchSize,0.8,deepWalk);
        DataSetIterator train = data[0];
        DataSetIterator test = data[1];

        //Create the network for classification:
        int nOut = 39;  //39 categories

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.01)
                .updater(Updater.ADAGRAD)
                .regularization(true).l2(0.0001)
                .list(4)
                .layer(0, new DenseLayer.Builder().nIn(vectorSize)
                        .nOut(200).activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(200)
                        .nOut(300).activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(300)
                        .nOut(150).activation("relu")
                        .weightInit(WeightInit.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder().nIn(150)
                        .nOut(nOut).activation("sigmoid")
                        .weightInit(WeightInit.RELU)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(10),new HistogramIterationListener(10));

        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            System.out.println("----- Network training epoch " + i + " complete -----");

            //Evalute at current epoch:
            Evaluation evaluation = evaluateNetwork(net,test);
            System.out.println(evaluation.stats());
        }


    }


    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String zipPath = DATA_PATH + "BlogCatalog-datazet.zip";
        File zipFile = new File(zipPath);

        if( !zipFile.exists() ){
            FileUtils.copyURLToFile(new URL(DATA_URL), zipFile);
            System.out.println("Data (zip file) downloaded to " + zipFile.getAbsolutePath());
        } else {
            System.out.println("Data (zip file) already exists at " + zipFile.getAbsolutePath());
        }

        //Extract zip file to output directory
        ZipFile zf = new ZipFile(zipPath);
        zf.extractAll(DATA_PATH);
    }

    private static Graph<String,String> loadGraph() throws IOException{

        //This particular file format: vertices are indexed 1..numVertices
        //We  need 0..numVertices-1 for DL4J graph representation
        VertexFactory<String> vertexFactory = new StringVertexFactory();
        EdgeLineProcessor<String> edgeLineProcessor = new EdgeLineProcessor<String>() {
            @Override
            public Edge<String> processLine(String line) {
                String[] split = line.split(",");
                int from = Integer.parseInt(split[0])-1;
                int to = Integer.parseInt(split[1])-1;
                return new Edge<>(from,to,from+"--"+to,false);  //create an undirected edge
            }
        };

        return GraphLoader.loadGraph(EDGE_LIST_PATH,    //location of data
                edgeLineProcessor,                      //EdgeLineProcessor: each line in file -> Edge
                vertexFactory,                          //VertexFactory creates vertex objects for each vertex index
                NUM_VERTICES,                           //Number of vertices in graph
                false);                                 //Whether multiple edges between two vertices should be allowed
    }

    private static INDArray loadGroupData() throws IOException {

        INDArray groupData = Nd4j.create(NUM_VERTICES, 39);  //39 possible groups

        try(BufferedReader br = new BufferedReader(new FileReader(new File(GROUP_DATA_PATH)))){
            String line;
            while( (line = br.readLine()) != null ) {
                String[] split = line.split(",");
                int userVertexIdx = Integer.parseInt(split[0])-1;
                int groupIdx = Integer.parseInt(split[1])-1;
                groupData.put(userVertexIdx,groupIdx,1.0);
            }
        }

        return groupData;
    }

    /** Returns two data set iterators.
     * @param fractionTrain Fraction of labelled data t use for training
     * @return array with 2 DataSetIterators; first for training, second for testing
     */
    private static DataSetIterator[] getTrainTestData(int minibatchSize, double fractionTrain,
                                                      DeepWalk<String,String> deepWalk ) throws IOException{
        INDArray groupData = loadGroupData();   //targets

        int nVertices = deepWalk.numVertices();

        List<DataSet> allData = new ArrayList<>(nVertices);
        for( int i=0; i<nVertices; i++ ){
            INDArray vertexVector = deepWalk.getVertexVector(i);
            INDArray targetGroups = groupData.getRow(i);
            DataSet ds = new DataSet(vertexVector,targetGroups);
            allData.add(ds);
        }

        Collections.shuffle(allData, new Random(12345));

        int nTrain = (int)(nVertices * fractionTrain);
        int nTest = nVertices - nTrain;

        List<DataSet> listTrain = new ArrayList<>(nTrain);
        List<DataSet> listTest = new ArrayList<>(nTest);

        for( int i=0; i<nTrain; i++ ) listTrain.add(allData.get(i));
        for( int i=nTrain; i<nVertices; i++ ) listTest.add(allData.get(i));

        DataSetIterator[] iterators = new DataSetIterator[2];
        iterators[0] = new ListDataSetIterator(listTrain,minibatchSize);
        iterators[1] = new ListDataSetIterator(listTest,minibatchSize);
        return iterators;
    }

    private static Evaluation evaluateNetwork(MultiLayerNetwork network, DataSetIterator data ){
        //Manually do evaluation here. We are doing multi-label classification -> can't simply use Evaluation.eval()
        Evaluation evaluation = new Evaluation();
        while(data.hasNext()){
            DataSet ds = data.next();
            INDArray features = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();
            INDArray out = network.output(features);

            int nRows = labels.rows();
            int nCols = labels.columns();

            for( int i=0; i<nRows; i++ ){
                for( int j=0; j<nCols; j++ ){
                    double predicted = out.getDouble(i,j);
                    double actual = labels.getDouble(i,j);

                    if(predicted > 0.5 ){
                        if(actual == 0.0){
                            evaluation.incrementFalsePositives(j);  //Predicted true, is false
                        } else {
                            evaluation.incrementTruePositives(j);   //Predicted true, is true
                        }
                    } else {
                        if(actual == 0.0){
                            evaluation.incrementTrueNegatives(j);  //Predicted false, is false
                        } else {
                            evaluation.incrementFalseNegatives(j);  //Predicted false, is true
                        }
                    }
                }
            }
        }
        return evaluation;
    }
}

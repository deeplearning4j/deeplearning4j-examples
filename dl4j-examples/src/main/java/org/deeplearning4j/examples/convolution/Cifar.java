package org.deeplearning4j.examples.convolution;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.StringUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * train model by cifar
 * identification unkonw file
 *
 * @author wangfeng
 * @since June 7,2017
 */

//@Slf4j
public class Cifar {
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir"), "cifar-model/");
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);

    private static int height = 8;
    private static int width = 8;
    private static String labelStr = "[]";
    public static void main(String[] args) throws IOException {

        Cifar cf = new Cifar();
        //train model and eval model
        MultiLayerNetwork model = null;
        String modelType = "alexNet";//bp,cnn,alexNet,costom
        switch (modelType) {
            case "bp":
                model = cf.trainModelByCifarDataWithDenseBP();
                break;
            case "cnn":
                model = cf.trainModelByCifarData();
                break;
            case "alexNet":
                model = cf.trainModelByCifarWithAlexNet();//ignore
                break;
            case "costom":
                model = cf.trainModelByCifarOtherNet();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        //provied service by makeing use of the model
        cf.testModelByUnkownImage(model);
    }
    //one DenseLayer,one OutLayer
    public MultiLayerNetwork trainModelByCifarDataWithDenseBP() {
        height = 100;
        width = 100;
        int channels = 3;
        int rngseed = 123;
        int numSamples = 60000;
        int batchSize = 128;
        int iterations = 3;
        Random rng = new Random(rngseed);
        int outputNum = CifarLoader.NUM_LABELS;
        int numEpochs = 1;
        boolean preProceeCifar = true;
        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples,
            new int[] {height, width, channels}, false, true);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(cifar);
        cifar.setPreProcessor(scaler);
        labelStr = String.join(",", cifar.getLabels().toArray(new String[cifar.getLabels().size()]));

        // Build Our Neural Network
        log.info("**** Build Model ****");
        //no add weight F1 score =0.3552,
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).momentum(0.9)// improve train time加快训练速度
            .iterations(iterations)//
            .learningRate(0.0002)//the point of the learningRate is weight - deviation*learningRate(学习率是在反向传播中改变误差的一种方式）
            .regularization(true).l2(1e-4)//To prevent over fitting and combining loss function（防止过拟合,配合损失函数）
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width * channels)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height,width,channels))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        log.info("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(cifar);
        }
        //Data Setup -> transformation,tranform and pollute image that tranform the picture over and deal with the pollution,
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        // Train with transformations
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            scaler.fit(cifar);
            cifar.setPreProcessor(scaler);
            model.fit(cifar);
        }
        log.info("******EVALUATE MODEL******");

        Evaluation eval = model.evaluate(cifar);
        log.info(eval.stats(true));
        cifar.reset();
        DataSet testDataSet = cifar.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = model.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print( "\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");

        return saveModel(model, "trainModelByCifarWithSimpleData_model.json");
    }
    //one ConvolutionLayer,one SubsamplingLayer,one OutLayer
    public MultiLayerNetwork trainModelByCifarData() {

        height = 32;
        width = 32;
        int channels = 3;
        int outputNum = CifarLoader.NUM_LABELS;
        int numSamples = 60000;
        int batchSize = 10;
        int iterations = 2;
        int seed = 123;
        int listenerFreq = iterations;
        boolean preProcessCifar = true;//use Zagoruyko's preprocess for Cifar
        Random rng = new Random(seed);

        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples,
            new int[] {height, width, channels}, preProcessCifar, true);
        labelStr = String.join(",", cifar.getLabels().toArray(new String[cifar.getLabels().size()]));

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(cifar);
        cifar.setPreProcessor(scaler);

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS).momentum(0.9)// improve train time加快训练速度
            .learningRate(0.003)//the point of the learningRate is weight - deviation*learningRate(学习率是在反向传播中改变误差的一种方式）
            .regularization(true).l2(1e-4)//To prevent over fitting and combining loss function（防止过拟合,配合损失函数）
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(channels)
                .nOut(6)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutionalFlat(height, width, channels));

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        model.fit(cifar);

        ImageTransform flipTransform1 = new FlipImageTransform();
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            scaler.fit(cifar);
            cifar.setPreProcessor(scaler);
            model.fit(cifar);
        }

        Evaluation eval = model.evaluate(cifar);
        log.info(eval.stats(true));
        cifar.reset();
        DataSet testDataSet = cifar.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = model.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print( "\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");

        //save model
        return saveModel(model, "trainModelByCifarData_model.json");
    }
    //five ConvolutionLayer,three SubsamplingLayer,two DenseLayer,one OutLayer
    public MultiLayerNetwork trainModelByCifarWithAlexNet() throws IOException {
        log.info("this is AlexNet");
        height = 32;
        width = 32;
        int channels = 3;
        int numLabels = CifarLoader.NUM_LABELS;
        int numSamples = 60000;
        int batchSize = 20;
        int iterations = 1;
        int seed = 123;
        Random rng = new Random(seed);
        int listenerFreq = 1;
        double nonZeroBias = 1;
        double dropOut = 0.5;
        boolean preProcessCifar = true;//use Zagoruyko's preprocess for Cifar
        int epochs = 50;
        int nCores = 2;
        /**
         * CifarDataSetIterator=load binary file and Sequence combine file,instantiate data labels（下载或者加载数据，并且把数据按照训练和测试分类，如果存在多个文件就进行流合并）
         */
        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples,
            new int[] {height, width, channels}, preProcessCifar, true);

        labelStr = String.join(",", cifar.getLabels().toArray(new String[cifar.getLabels().size()]));
        //Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(cifar);
        cifar.setPreProcessor(scaler);




        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(1e-2*2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false)
            .list()
            .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{1, 1}).name("cnn1").nIn(3).nOut(96).biasInit(0).build())
            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{1,1}).name("maxpool1").build())
            .layer(3, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name("cnn2").nOut(256).biasInit(nonZeroBias).build())
            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}).name("maxpool2").build())
            .layer(6, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {1,1}).name("cnn3").nOut(384).biasInit(0).build())
            .layer(7, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {1,1}).name("cnn4").nOut(384).biasInit(nonZeroBias).build())
            .layer(8, new ConvolutionLayer.Builder(new int[]{2,2}, new int[] {1,1}, new int[] {1,1}).name("cnn5").nOut(256).biasInit(nonZeroBias).build())
            .layer(9, new SubsamplingLayer.Builder(new int[]{1,1}, new int[]{1,1}).name("maxpool3").build())
            .layer(10, new DenseLayer.Builder().name("ffn1").nOut(100).biasInit(nonZeroBias).dropOut(dropOut).dist(new GaussianDistribution(0, 0.005)).build())
            .layer(11, new DenseLayer.Builder().name("ffn2").nOut(100).biasInit(nonZeroBias).dropOut(dropOut).dist(new GaussianDistribution(0, 0.005)).build())
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //the visual ui
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        // model.setListeners(new ScoreIterationListener(listenerFreq));
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, cifar, nCores);

        model.fit(trainIter);


        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        // Train with transformations
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            scaler.fit(cifar);
            cifar.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, cifar, nCores);
            model.fit(trainIter);
        }

        cifar.test(10);
        Evaluation eval = new Evaluation(cifar.getLabels());
        while (cifar.hasNext()) {
            DataSet testDS = cifar.next(batchSize);
            INDArray output = model.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats(true));
        //save model
        return saveModel(model, "trainModelByCifarWithAlexNet_model.json");
    }

    public MultiLayerNetwork trainModelByCifarOtherNet() {

        return null;
    }

    public MultiLayerNetwork saveModel(MultiLayerNetwork model, String fileName) {
        File locationModelFile = new File(DATA_PATH + fileName);
        boolean saveUpdater = false;
        try {
            ModelSerializer.writeModel(model,locationModelFile,saveUpdater);
        } catch (Exception e) {
            log.error("Saving model is not success !",e);
        }
        return model;
    }
    public void testModelByUnkownImage(MultiLayerNetwork model) {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        String filename = "";
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            filename = file.getAbsolutePath();
        }
        AnalysisUnkownImage ui = new AnalysisUnkownImage(filename,model);
        ui.showGUI();
    }



    private class AnalysisUnkownImage extends JFrame{
        public static final int CLASS_LOADER_CHAR = 0X12FFAADE ;

        private JTextArea jta;
        MultiLayerNetwork model;
        public AnalysisUnkownImage(String filename, MultiLayerNetwork model){
            this.model = model;
            JLabel label1=new JLabel("Directory/File(Plz fill in path");
            JLabel label3=new JLabel("  skymind.ai");
            JLabel label4=new JLabel("welcome to here  ");
            label3.setBackground(new Color(87,105,227));
            label4.setBackground(new Color(87,105,227));
            label3.setForeground(new Color(28,11,185));
            label4.setForeground(new Color(239,16,228));
            JButton jbt = new JButton("The identification of unknown and show(Click Here)");
            JTextField jtf = new JTextField(380);
            jtf.setText(filename);
            jta = new JTextArea(15,80);
            jta.setLineWrap(true);
            jta.setWrapStyleWord(true);
            JScrollPane jsp = new JScrollPane(jta);
            LineBorder lb = new LineBorder(Color.red,1);
            TitledBorder tb = new TitledBorder(new TitledBorder(""), " Identification result",0, 5, new Font("Serif",Font.BOLD,12), Color.BLUE);

            JPanel panel = new JPanel();
            JPanel panel1 = new JPanel();
            JPanel panel2 = new JPanel();
            JPanel panel3 =new JPanel();
            JPanel panel4 = new JPanel();
            JPanel panel5 = new JPanel();
            JPanel panel7 = new JPanel();

            panel5.setLayout(new BorderLayout(5,5));
            panel5.add(label1,BorderLayout.WEST);
            panel5.add(jtf,BorderLayout.CENTER);
            panel7.setLayout(new GridLayout(1,2,5,5));
            panel7.add(panel5);
            panel1.setLayout(new BorderLayout(5,5));
            panel1.setBorder(lb);
            panel1.add(panel7,BorderLayout.CENTER);
            jbt.setForeground(Color.BLUE);
            jbt.setBackground(Color.GREEN);
            panel2.setLayout(new BorderLayout(5,5));
            panel2.add(label3,BorderLayout.WEST);
            panel2.add(jbt,BorderLayout.CENTER);
            panel2.add(label4, BorderLayout.EAST);
            panel3.setLayout(new BorderLayout(1,1));
            panel3.add(jsp,BorderLayout.CENTER);
            panel3.setBorder(tb);
            panel4.setLayout(new BorderLayout(5,5));
            panel4.add(panel2,BorderLayout.NORTH);
            panel4.add(panel3,BorderLayout.CENTER);

            panel.setLayout(new BorderLayout(5,5));
            panel.add(panel1,BorderLayout.NORTH);
            panel.add(panel4,BorderLayout.CENTER);
            //register listener
            jbt.addActionListener( new ActionListener() {
                public void actionPerformed(ActionEvent ae) {
                    String fileAbsolutePath = jtf.getText();
                    jta.setText("labels->->->" + labelStr);
                    jta.append("\n");
                    double numz = 0.0;
                    boolean judge = true;
                    File file = new File(fileAbsolutePath);
                    File [] files = new File[1];
                    if (file.exists()) {
                        if (file.isDirectory()) {
                            files = file.listFiles();
                        }else {
                            files[0] = file;
                        }
                        analysisFileName(files);

                    }
                }
            });
            add(panel);
        }
        public void analysisFileName(File[] files) {
            for (int i = 0; i < files.length; i ++) {
                if (files[i].isDirectory()) {
                    analysisFileName(files[i].listFiles());
                } else {
                    //the suffix of the file
                    String suffix = files[i].getName();
                    suffix = suffix.substring(suffix.lastIndexOf(".")+1);
                    String  formatAllows = StringUtils.arrayToString(NativeImageLoader.ALLOWED_FORMATS);
                    if(formatAllows.contains(suffix)){
                        File file = files[i];
                        // Use NativeImageLoader to convert to numerical matrix
                        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
                        // Get the image into an INDarray
                        INDArray image = null;
                        try {
                            image = loader.asMatrix(file);
                        } catch (Exception e) {
                            log.error("the loader.asMatrix have any abnormal",e);
                        }
                        if (image == null) {
                            return;
                        }
                        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
                        scaler.transform(image);
                        INDArray output = model.output(image);
                        String modelResult = output.toString();
                        int [] predict = model.predict(image);
                        modelResult += "===" + Arrays.toString(predict);
                        jta.append("the file chosen:");
                        jta.append("\n");
                        jta.append( files[i].getAbsolutePath());
                        jta.append("\n");
                        jta.append("the  identification result :" + modelResult);
                        jta.append("\n");

                    }


                }
            }
        }
        public void showGUI() {
            setSize(560,500);
            setLocationRelativeTo(null);
            setBackground(Color.green);
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            setVisible(true);
        }
    }
}

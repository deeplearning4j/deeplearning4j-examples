package org.deeplearning4j.examples.convolution;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.util.StringUtils;
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


/**
 * train model by cifar
 * identification unkonw file
 *
 * @author wangfeng
 * @since June 7,2017
 */

//@Slf4j
public class Cifar {
    private static final File DATA_PATH = new File(System.getProperty("user.dir"));
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);


    private static String labelStr = "[]";
    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int numLabels = CifarLoader.NUM_LABELS;
    private static int batchSize = 96;
    private static long seed = 123L;
    private static int epochs = 50;

    public static void main(String[] args) throws Exception {
//        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        Cifar cf = new Cifar();

        Cifar10DataSetIterator cifar = new Cifar10DataSetIterator(batchSize, new int[]{height, width, channels}, DataSetType.TRAIN, null, seed);
        Cifar10DataSetIterator cifarEval = new Cifar10DataSetIterator(batchSize, new int[]{height, width, channels}, DataSetType.TEST, null, seed);

        //train model and eval model
        MultiLayerNetwork model = cf.trainModelByCifarWithNet();//ignore
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File("java.io.tmpdir"));
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(50), new EvaluativeListener(cifarEval, 1, InvocationType.EPOCH_END));

        labelStr = String.join(",", cifar.getLabels().toArray(new String[cifar.getLabels().size()]));
        model.fit(cifar, epochs);

        cf.testModelByUnkownImage(model);
        cf.saveModel(model, "trainModelByCifarWithAlexNet_model.json");
    }


    public MultiLayerNetwork trainModelByCifarWithNet() throws IOException {
        log.info("this is Net for the cifar");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new AdaDelta())
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .l1(1e-4)
            .list()
            .layer(new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).padding(2,2).activation(Activation.RELU)
                .nIn(channels).nOut(16).build())
            .layer(new BatchNormalization())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())

            .layer(new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).padding(2,2).activation(Activation.RELU)
                .nOut(20).build())
            .layer(new BatchNormalization())
            .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())

            .layer(new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).padding(2,2).activation(Activation.RELU)
                .nOut(20).build())
            .layer(new BatchNormalization())
            .layer(new SubsamplingLayer.Builder().kernelSize(5,5).stride(2,2).build())

            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .dropOut(0.8)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }


    public MultiLayerNetwork saveModel(MultiLayerNetwork model, String fileName) {
        File locationModelFile = new File(DATA_PATH, fileName);
        boolean saveUpdater = false;
        try {
            ModelSerializer.writeModel(model, locationModelFile, saveUpdater);
            log.info("trained model was saved to {}", locationModelFile);
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
                       /* DataNormalization scaler = new ImagePreProcessingScaler(0,1);
                        scaler.transform(image);*/
                        INDArray output = model.output(image);

                        log.info("## The Neural Nets Pediction ##");
                        log.info("## list of probabilities per label ##");
                        //log.info("## List of Labels in Order## ");
                        // In new versions labels are always in order
                        log.info(output.toString());

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


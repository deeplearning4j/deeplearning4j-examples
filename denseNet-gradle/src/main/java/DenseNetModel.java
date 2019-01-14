import org.deeplearning4j.nn.graph.ComputationGraph;

public class DenseNetModel {

    private static DenseNetModel instance;

    public static DenseNetModel getInstance() {
        if (instance == null) {
            instance = new DenseNetModel();
        }
        return instance;
    }

    public ComputationGraph buildNetwork(long seed, int channels, int numLabels, int width, int height) {

        DenseNetBuilder denseNetModel = new DenseNetBuilder(height, width, channels, seed, 12); //227x227x3

        String init = denseNetModel.initLayer(7, 2, 1, channels); //56x56x24
        String[] block1 = denseNetModel.addDenseBlock(6, true, "db1", new String[]{init});
        String trans1 = denseNetModel.addTransitionLayer("tr1", 96, block1); //28x28x48
        String[] block2 = denseNetModel.addDenseBlock(12, true, "db2", new String[]{trans1});
        String trans2 = denseNetModel.addTransitionLayer("tr2", 192, block2); //14x14x96
        String[] block3 = denseNetModel.addDenseBlock(24, true, "db3", new String[]{trans2});
        String trans3 = denseNetModel.addTransitionLayer("tr3", 384, block3); //7x7x192
        String[] block4 = denseNetModel.addDenseBlock(16, true, "db4", new String[]{trans3});
        denseNetModel.addOutputLayer(7, 7, 384, numLabels, block4);

        return denseNetModel.getModel();
    }
}

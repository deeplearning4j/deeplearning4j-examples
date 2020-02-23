import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

public class AleTest {

    @Test
    public void TrainModelDataShape(){

        // Set up the training as in the sample.
        HistoryProcessor.Configuration ALE_HP = new HistoryProcessor.Configuration(
                4,       //History length
                84,      //resize width
                110,     //resize height
                84,      //crop width
                84,      //crop height
                0,       //cropping x offset
                0,       //cropping y offset
                4        //skip mod (one frame is picked every x
        );

        A3CDiscrete.A3CConfiguration ALE_A3C = new A3CDiscrete.A3CConfiguration(
                123,            //Random seed
                10000,          //Max step By epoch
                8000000,        //Max step
                8,              //Number of threads
                32,             //t_max
                500,            //num step noop warmup
                0.1,            //reward scaling
                0.99,           //gamma
                10.0            //td-error clipping
        );

        final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
                new ActorCriticFactoryCompGraphStdConv.Configuration(
                        0.000,   //l2 regularization
                        new Adam(0.00025), //learning rate
                        null, false
                );
        ALEMDP mdp = new ALEMDP("pong.bin");
        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv<ALEMDP.GameScreen>(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C);

        NeuralNetwork [] nns  = a3c.getNeuralNet().getNeuralNetworks();
        ComputationGraph g = (ComputationGraph ) nns[0];

        // Now pass in some dummy data in the expected shape.
        INDArray dummy = Nd4j.rand( 1,4, 84, 84);
        g.output(new INDArray[] {dummy}); //If we get the shape wrong we crash here.
    }

    @Test
    void LoadModel() throws IOException {
        //load the previous agent
        ACPolicy<ALEMDP.GameScreen> pol = ACPolicy.load("ale-a3c.model");
        NeuralNetwork [] nns  = pol.getNeuralNet().getNeuralNetworks();
        ComputationGraph g = (ComputationGraph ) nns[0];

        // Now pass in some dummy data in the expected shape.
        INDArray dummy = Nd4j.rand( 1,4, 84, 84);
        g.output(new INDArray[] {dummy}); //If we get the shape wrong we crash here.
    }
}

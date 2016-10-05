package org.deeplearning4j.examples.rl4j;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.vizdoom.DeadlyCorridor;
import org.deeplearning4j.rl4j.mdp.vizdoom.VizDoom;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * Main example for doom  DQN
 *
 * Note: those parameters do not converge to the gif
 * The gif was obtained with a higher learning rate (0.005)
 * no skip frame (skip frame = 1), on a deadly corridir configured
 * with less doomSkills (now 5, was default value before)
 * and a code that has been heavily modified.
 *
 * The example in the gif doesn't kill the ennemy, he just runs straight
 * which is "too easy"
 */
public class Doom {




    public static QLearning.QLConfiguration DOOM_QL =
            new QLearning.QLConfiguration(
                    123,      //Random seed
                    10000,    //Max step By epoch
                    8000000,  //Max step
                    1000000,  //Max size of experience replay
                    32,       //size of batches
                    10000,    //target update (hard)
                    50000,    //num step noop warmup
                    0.001,    //reward scaling
                    0.99,     //gamma
                    100.0,    //td-error clipping
                    0.1f,     //min epsilon
                    100000,   //num step for eps greedy anneal
                    true      //double-dqn
            );




    public static DQNFactoryStdConv.Configuration DOOM_NET =
            new DQNFactoryStdConv.Configuration(
                    0.00025, //learning rate
                    0.000    //l2 regularization
            );

    public static HistoryProcessor.Configuration DOOM_HP =
            new HistoryProcessor.Configuration(
                    4,       //History length
                    84,      //resize width
                    84,      //resize height
                    84,      //crop width
                    84,      //crop height
                    0,       //cropping x offset
                    0,       //cropping y offset
                    4        //skip mod (one frame is picked every x
            );

    public static void main(String[] args) {
        doomBasicQL();
    }

    public static void doomBasicQL() {

        //record the training data in rl4j-data in a new folder
        // install webapp-rl4j - access website on aws and play online - enable right port - slow on aws
        DataManager manager = new DataManager(true); // stores 1 episode (video & stats) in rl4j data

        //setup the Doom environment through VizDoom
        VizDoom mdp = new DeadlyCorridor(false); // render variable - show all episodes

        //setup the training
        QLearningDiscreteConv<VizDoom.GameScreen> dql = new QLearningDiscreteConv(mdp, DOOM_NET, DOOM_HP, DOOM_QL, manager);

        //start the training
        dql.train();

        //save the model at the end
        dql.getPolicy().save("doom-end.model");

        //close the doom env
        mdp.close();
    }
}

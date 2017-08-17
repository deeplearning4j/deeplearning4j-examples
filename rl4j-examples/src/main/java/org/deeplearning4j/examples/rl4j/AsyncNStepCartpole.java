package org.deeplearning4j.examples.rl4j;

import java.io.IOException;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * main example for Async NStep QLearning on cartpole
 */
public class AsyncNStepCartpole {


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration CARTPOLE_NSTEP =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,     //Random seed
                    200,     //Max step By epoch
                    300000,  //Max step
                    16,      //Number of threads
                    5,       //t_max
                    100,     //target update (hard)
                    10,      //num step noop warmup
                    0.01,    //reward scaling
                    0.99,    //gamma
                    100.0,   //td-error clipping
                    0.1f,    //min epsilon
                    9000     //num step for eps greedy anneal
            );


    public static DQNFactoryStdDense.Configuration CARTPOLE_NET_NSTEP =
        DQNFactoryStdDense.Configuration.builder()
        .l2(0.001).learningRate(0.0005).numHiddenNodes(16).numLayer(3).build();


    public static void main(String[] args) throws IOException {
        cartPole();
    }


    public static void cartPole() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);
        //define the mdp from gym (name, render)
        GymEnv mdp = null;
        try {
        mdp = new GymEnv("CartPole-v0", false, false);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }

        //define the training
        AsyncNStepQLearningDiscreteDense<Box> dql = new AsyncNStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET_NSTEP, CARTPOLE_NSTEP, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = (DQNPolicy<Box>)dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close connection)
        mdp.close();

        //reload the policy, will be equal to "pol"
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");
    }
}

package org.deeplearning4j.examples.rl4j;

import java.io.IOException;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import java.util.logging.Logger;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * Main example for Cartpole DQN
 *
 * **/
public class Cartpole
{

    public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,    //Random seed
                    200,    //Max step By epoch
                    150000, //Max step
                    150000, //Max size of experience replay
                    32,     //size of batches
                    500,    //target update (hard)
                    10,     //num step noop warmup
                    0.01,   //reward scaling
                    0.99,   //gamma
                    1.0,    //td-error clipping
                    0.1f,   //min epsilon
                    1000,   //num step for eps greedy anneal
                    true    //double DQN
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET =
        DQNFactoryStdDense.Configuration.builder()
            .l2(0.001).learningRate(0.0005).numHiddenNodes(16).numLayer(3).build();

    public static void main(String[] args) throws IOException {
        cartPole();
        loadCartpole();
    }

    public static void cartPole() throws IOException {

        //record the training data in rl4j-data in a new folder (save)
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp = null;
        try {
            mdp = new GymEnv("CartPole-v0", false, false);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }
        //define the training
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close http)
        mdp.close();


    }


    public static void loadCartpole() throws IOException {

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        GymEnv mdp2 = new GymEnv("CartPole-v0", true, false);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }
}

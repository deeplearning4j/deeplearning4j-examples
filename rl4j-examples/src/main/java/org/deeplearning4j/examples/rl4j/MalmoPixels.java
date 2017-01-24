package org.deeplearning4j.examples.rl4j;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

import com.microsoft.msr.malmo.MissionSpec;

import org.deeplearning4j.malmo.MalmoBox;
import org.deeplearning4j.malmo.MalmoActionSpaceDiscrete;
import org.deeplearning4j.malmo.MalmoConnectionError;
import org.deeplearning4j.malmo.MalmoDescretePositionPolicy;
import org.deeplearning4j.malmo.MalmoEnv;
import org.deeplearning4j.malmo.MalmoObservationSpace;
import org.deeplearning4j.malmo.MalmoObservationSpacePixels;
import org.deeplearning4j.malmo.MalmoResetHandler;

import java.io.IOException;
import java.util.Random;
import java.util.logging.Logger;

/**
 * More complex example for Malmo DQN w/ screen pixels as input. After the network learns how to find the reward
 * on a simple open plane, the mission is made more complex by putting lava in the way.
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoPixels {
    public static QLearning.QLConfiguration MALMO_QL = new QLearning.QLConfiguration(123, //Random seed
                    200, //Max step By epoch
                    100000, //Max step
                    50000, //Max size of experience replay
                    32, //size of batches
                    500, //target update (hard)
                    10, //num step noop warmup
                    0.01, //reward scaling
                    0.99, //gamma
                    1.0, //td-error clipping
                    0.1f, //min epsilon
                    10000, //num step for eps greedy anneal
                    true //double DQN
    );

    public static DQNFactoryStdConv.Configuration MALMO_NET = new DQNFactoryStdConv.Configuration(
    				0.01, //learning rate
                    0.00, //l2 regularization
                    null, // updater
                    null // Listeners
    );

    /*
     * The pixel input is 320x240, but using the history processor we scale that to 160x120
     * and then crop out a 160x80 segment to remove pixels that aren't needed 
     */
    public static HistoryProcessor.Configuration MALMO_HPROC = new HistoryProcessor.Configuration(1, // Number of frames
                    160, // Scaled width
                    120, // Scaled height
                    160, // Cropped width
                    80, // Cropped height
                    0, // X offset
                    30, // Y offset
                    1 // Number of frames to skip
    );

    public static void main(String[] args) throws IOException {
        try {
            malmoCliffWalk();
            loadMalmoCliffWalk();
        } catch (MalmoConnectionError e) {
            System.out.println(
                            "To run this example, download and start Project Malmo found at https://github.com/Microsoft/malmo.");
        }
    }

    private static MalmoEnv createMDP() {
        return createMDP(0);
    }

    private static MalmoEnv createMDP(final int initialCount) {
        MalmoActionSpaceDiscrete actionSpace =
                        new MalmoActionSpaceDiscrete("movenorth 1", "movesouth 1", "movewest 1", "moveeast 1");
        actionSpace.setRandomSeed(123);
        MalmoObservationSpace observationSpace = new MalmoObservationSpacePixels(320, 240);
        MalmoDescretePositionPolicy obsPolicy = new MalmoDescretePositionPolicy();

        MalmoEnv mdp = new MalmoEnv("cliff_walking_rl4j.xml", actionSpace, observationSpace, obsPolicy);

        final Random r = new Random(12345);

        mdp.setResetHandler(new MalmoResetHandler() {
            int count = initialCount;

            @Override
            public void onReset(MalmoEnv malmoEnv) {
                count++;

                if (count > 500) {
                    MissionSpec mission = MalmoEnv.loadMissionXML("cliff_walking_rl4j.xml");

                    for (int x = 1; x < 4; ++x)
                        for (int z = 1; z < 13; ++z)
                            if (r.nextFloat() < 0.1)
                                mission.drawBlock(x, 45, z, "lava");

                    malmoEnv.setMission(mission);
                }
            }
        });

        return mdp;
    }

    public static void malmoCliffWalk() throws MalmoConnectionError, IOException {
        //record the training data in rl4j-data in a new folder (save)
        DataManager manager = new DataManager(false);

        MalmoEnv mdp = createMDP();

        //define the training
        QLearningDiscreteConv<MalmoBox> dql =
                        new QLearningDiscreteConv<MalmoBox>(mdp, MALMO_NET, MALMO_HPROC, MALMO_QL, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<MalmoBox> pol = dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("cliffwalk_pixel.policy");

        //close the mdp
        mdp.close();
    }

    //showcase serialization by using the trained agent on a new similar mdp
    public static void loadMalmoCliffWalk() throws MalmoConnectionError, IOException {
        MalmoEnv mdp = createMDP(10000);

        //load the previous agent
        DQNPolicy<MalmoBox> pol = DQNPolicy.load("cliffwalk_pixel.policy");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 10; i++) {
            double reward = pol.play(mdp, new HistoryProcessor(MALMO_HPROC));
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        // Clean up
        mdp.close();

        Logger.getAnonymousLogger().info("average: " + rewards / 10);
    }
}

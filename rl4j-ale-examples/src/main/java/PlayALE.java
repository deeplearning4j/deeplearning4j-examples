import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.policy.ACPolicy;

import java.io.IOException;
import java.util.logging.Logger;

public class PlayALE {
    public static void main(String[] args) throws IOException {
        ALEMDP mdp = new ALEMDP("E:\\projects\\ArcadeLearningEnvironment\\pong.bin");

        //load the previous agent
        ACPolicy<ALEMDP.GameScreen> pol2 = ACPolicy.load("ale-a3c.model");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 10; i++) {
            mdp.reset();
            double reward = pol2.play(mdp);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }
}

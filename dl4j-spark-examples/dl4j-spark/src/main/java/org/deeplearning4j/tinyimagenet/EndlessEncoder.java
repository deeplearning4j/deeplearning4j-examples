package org.deeplearning4j.tinyimagenet;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class EndlessEncoder {

    private static float[] thresholds = new float[] {1e-3f, 1e-4f, 1e-5f, 1e-6f};

    private static WorkspaceConfiguration wsConf = WorkspaceConfiguration.builder()
                                                                                                                .initialSize(3L * 1024L * 1024L * 1024L)
                                                                                                                .policyReset(ResetPolicy.BLOCK_LEFT)
                                                                                                                .policyAllocation(AllocationPolicy.STRICT)
                                                                                                                .build();

    public static void main(String[] args) throws Exception {

        val threads = new ArrayList<Thread>();
        for (int e = 0; e < 2; e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    while (true) {
                        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConf, "id")) {
                            val paramsLength = RandomUtils.nextInt(100000, 10000000);
                            val params = Nd4j.create(paramsLength);


                            for (val tau: thresholds) {
                                Nd4j.getMemoryManager().memset(params);

                                // random limit by default
                                val limit = RandomUtils.nextInt(0, paramsLength / 20);

                                // uncomment this to test on low updates numbers
                                //val limit = RandomUtils.nextInt(0, 20);
                                for (int e = 0; e < limit; e++) {
                                    val position = RandomUtils.nextInt(0, paramsLength);
                                    params.putScalar(position, tau + Nd4j.EPS_THRESHOLD);
                                }

                                log.info("[Thread {}]: device: [{}]; paramsLength: [{}]; updates: [{}]; tau: [{}]", Thread.currentThread().getId(), Nd4j.getAffinityManager().getDeviceForCurrentThread(), paramsLength, limit, tau);

                                val encoded = Nd4j.getExecutioner().thresholdEncode(params, tau, null);

                                Nd4j.getExecutioner().commit();

                                if (encoded == null) {
                                    log.error("[Thread {}]: got null", Thread.currentThread().getId());
                                }
                            }
                        }
                    }
                }
            });

            t.start();
            threads.add(t);
        }

        for (val t:threads)
            t.join();
    }
}

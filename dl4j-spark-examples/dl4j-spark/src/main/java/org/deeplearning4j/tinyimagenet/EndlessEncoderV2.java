package org.deeplearning4j.tinyimagenet;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

@Slf4j
public class EndlessEncoderV2 {

    private static float[] thresholds = new float[]{1e-3f, 1e-4f, 1e-5f, 1e-6f};

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

                            //log normal
                            double mean = 1e-2;
                            double sigma = 3;
                            double mu = Math.log(mean) - sigma * sigma / 2;
                            INDArray out = Nd4j.getExecutioner().exec(new LogNormalDistribution(Nd4j.createUninitialized(1), mu, sigma));
                            double tau = out.getDouble(0);

                            Nd4j.getMemoryManager().memset(params);

                            // random limit by default
                            val limit = RandomUtils.nextInt(0, paramsLength / 20);

//                            // uncomment this to test on low updates numbers
//                            //val limit = RandomUtils.nextInt(0, 20);
//                            for (int e = 0; e < limit; e++) {
//                                val position = RandomUtils.nextInt(0, paramsLength);
//                                params.putScalar(position, tau + Nd4j.EPS_THRESHOLD);
//                            }

                            int perturb = RandomUtils.nextInt(0, 50);
                            int perturbCount = RandomUtils.nextInt(1, 10);
                            String perturbStr = "none";
                            switch (perturb){
                                case 0:
                                    double pow = RandomUtils.nextInt(1, 10);
                                    //A few large values
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), RandomUtils.nextDouble() * Math.pow(10, pow));
                                    }
                                    perturbStr = "largeValues[" + perturbCount + ", rand*10e" + pow + "]";
                                    break;
                                case 1:
                                    //A few tiny values
                                    double powSmall = -RandomUtils.nextInt(100, 0);
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), RandomUtils.nextDouble() * Math.pow(10, powSmall));
                                    }
                                    perturbStr = "smallValues[" + perturbCount + ", rand*10e" + powSmall + "]";
                                    break;
                                case 2:
                                    //A few zero values
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), 0.0);
                                    }
                                    perturbStr = "zeros[" + perturbCount + "]";
                                    break;
                                case 3:
                                    //A few +inf
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), Double.POSITIVE_INFINITY);
                                    }
                                    perturbStr = "positiveInf[" + perturbCount + "]";
                                    break;
                                case 4:
                                    //A few -inf
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), Double.NEGATIVE_INFINITY);
                                    }
                                    perturbStr = "negativeInf[" + perturbCount + "]";
                                    break;
                                case 5:
                                    //A few NaNs
                                    for( int i=0; i<perturbCount; i++ ){
                                        params.putScalar(RandomUtils.nextInt(0, paramsLength-1), Double.NaN);
                                    }
                                    perturbStr = "nans[" + perturbCount + "]";
                                    break;
                                case 6:
                                    //All zeros
                                    params.assign(0);
                                    perturbStr = "allZero";
                                    break;
                                case 7:
                                    //All inf
                                    params.assign(Double.POSITIVE_INFINITY);
                                    perturbStr = "allInf";
                                    break;
                                case 8:
                                    params.assign(Double.NEGATIVE_INFINITY);
                                    perturbStr = "allNegativeInf";
                                    break;
                                case 9:
                                    params.assign(Double.NaN);
                                    perturbStr = "allNaN";
                                    break;
                                default:
                                    perturbStr = "none";
                            }


                            log.info("[Thread {}]: device: [{}]; paramsLength: [{}]; updates: [{}]; tau: [{}]; perturbation [{}]", Thread.currentThread().getId(), Nd4j.getAffinityManager().getDeviceForCurrentThread(), paramsLength, limit, tau, perturbStr);

                            val encoded = Nd4j.getExecutioner().thresholdEncode(params, tau, null);

                            Nd4j.getExecutioner().commit();

                            if (encoded == null) {
                                log.error("[Thread {}]: got null", Thread.currentThread().getId());
                            }
                        }
                    }
                }
            });

            t.start();
            threads.add(t);
        }

        for (val t : threads)
            t.join();
    }
}

package org.nd4j.examples;

import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Random {

    public static void main(String[] args) throws Exception {
        int[] shapes = new int[] {16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

        WorkspaceConfiguration conf = WorkspaceConfiguration.builder()
            .initialSize(10L * 1024L * 1024L * 1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyReset(ResetPolicy.BLOCK_LEFT)

            .build();

        for (int shapeR: shapes) {
            for (int shapeC: shapes) {
                long[] shape = new long[]{shapeR, shapeC};
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                    System.out.println("Trying N: {" + shapeR + ", " + shapeC + "}");
                    INDArray arrayN = Nd4j.randn(shape);

                    Nd4j.getExecutioner().commit();
                }

                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                    System.out.println("Trying U: {" + shapeR + ", " + shapeC + "}");
                    INDArray arrayU = Nd4j.rand(shape);

                    Nd4j.getExecutioner().commit();
                }
            }
        }


        for (int e = 0; e < 100000; e++) {
            int shapeR = RandomUtils.nextInt(8, 32768);
            int shapeC = RandomUtils.nextInt(8, 32768);
            long[] shape = new long[]{shapeR, shapeC};

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                System.out.println("Trying N: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayN = Nd4j.randn(shape);

                Nd4j.getExecutioner().commit();
            }

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                System.out.println("Trying U: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayU = Nd4j.rand(shape);

                Nd4j.getExecutioner().commit();
            }
        }
    }
}

package org.nd4j.examples;

import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.LogNormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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
                    System.out.println("Mean: " + arrayN.meanNumber() + ", stdev: " + arrayN.stdNumber());


                    Nd4j.getExecutioner().commit();
                }

                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                    System.out.println("Trying U: {" + shapeR + ", " + shapeC + "}");
                    INDArray arrayU = Nd4j.rand(shape);
                    System.out.println("Mean: " + arrayU.meanNumber() + ", stdev: " + arrayU.stdNumber());

                    Nd4j.getExecutioner().commit();
                }

                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                    System.out.println("Trying LN: {" + shapeR + ", " + shapeC + "}");
                    INDArray arrayLN = Nd4j.getExecutioner().exec(new LogNormalDistribution(Nd4j.createUninitialized(shape)));
                    System.out.println("Mean: " + arrayLN.meanNumber() + ", stdev: " + arrayLN.stdNumber());
                    Transforms.log(arrayLN);
                    System.out.println("Mean (logx): " + arrayLN.meanNumber() + ", stdev (logx): " + arrayLN.stdNumber());

                    Nd4j.getExecutioner().commit();
                }
            }
        }


        for (int e = 0; e < 100000; e++) {
            int shapeR = RandomUtils.nextInt(8, 32768);
            int shapeC = RandomUtils.nextInt(8, 32768);
            long[] shape = new long[]{shapeR, shapeC};

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                System.out.println(e + " - Trying N: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayN = Nd4j.randn(shape);
                System.out.println("Mean: " + arrayN.meanNumber() + ", stdev: " + arrayN.stdNumber());

                Nd4j.getExecutioner().commit();
            }

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                System.out.println(e + " - Trying U: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayU = Nd4j.rand(shape);
                System.out.println("Mean: " + arrayU.meanNumber() + ", stdev: " + arrayU.stdNumber());


                Nd4j.getExecutioner().commit();
            }

            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "id")) {
                System.out.println(e + " - Trying LN: {" + shapeR + ", " + shapeC + "}");
                INDArray arrayLN = Nd4j.getExecutioner().exec(new LogNormalDistribution(Nd4j.createUninitialized(shape)));
                System.out.println("Mean: " + arrayLN.meanNumber() + ", stdev: " + arrayLN.stdNumber());
                Transforms.log(arrayLN);
                System.out.println("Mean (logx): " + arrayLN.meanNumber() + ", stdev (logx): " + arrayLN.stdNumber());

                Nd4j.getExecutioner().commit();
            }
        }
    }
}

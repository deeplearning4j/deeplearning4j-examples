package org.nd4j.examples;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;

/**
 * This example shows how to use memory Workspaces with ND4j for cyclic workloads.
 *
 * Background:
 *
 * ND4j Workspace is a memory chunk, allocated once, and reused over in over.
 * Basically it gives you a way to avoid garbage collection for off-heap memory if you work with cyclic workloads.
 *
 * PLEASE NOTE: Workspaces are OPTIONAL. If you prefer using original GC-based memory managemend - you can use it without any issues.
 * PLEASE NOTE: When working with workspaces, YOU are responsible for tracking scopes etc. You are NOT supposed to access any INDArray that's attached to some workspace, outside of it. Results will be unpredictable, up to JVM crashes.
 *
 * @author raver119@gmail.com
 */
public class Nd4jEx15_Workspaces {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Nd4jEx15_Workspaces.class);

    public static void main(String[] args) throws Exception {
        /**
         * Each workspace is tied to a JVM Thread via ID. So, same ID in different threads will point to different actual workspaces
         * Each workspace is created using some configuration, and different workspaces can either share the same config, or have their own
         */

        // we create config with 10MB memory space preallocated
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder()
            .initialSize(10 * 1024L * 1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.NONE)
            .build();


        INDArray result = null;

        // we use
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID")) {
            // now, every INDArray created within this try block will be allocated from this workspace pool
            INDArray array = Nd4j.rand(10, 10);

            // easiest way to see if this array is attached to some workspace. We expect TRUE printed out here.
            log.info("Array attached? {}", array.isAttached());

            // please note, new array mean will be also attached to this workspace
            INDArray mean = array.mean(1);

            /**
             * PLEASE NOTE: if after doing some operations on the workspace, you want to bring result away from it, you should either leverage it, or detach
             */

            result = mean.detach();
        }

        // Since we've detached array, we expect FALSE printed out here. So, result array is managed by GC now.
        log.info("Array attached? {}", result.isAttached());



        /**
         * Workspace can be initially preallocated as shown above, or can be learning their desired size over time, or after first loop
         */

        WorkspaceConfiguration learningConfig = WorkspaceConfiguration.builder()
            .policyAllocation(AllocationPolicy.STRICT) // <-- this option disables overallocation behavior
            .policyLearning(LearningPolicy.FIRST_LOOP) // <-- this option makes workspace learning after first loop
            .build();

        for (int x = 0; x < 10; x++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(learningConfig, "OTHER_ID")) {
                INDArray array = Nd4j.create(100);

                /**
                 * At first iteration, workspace will be spilling all allocations as separate memory chunks.
                 * But after first iteration is finished - workspace will be allocated, to match all required allocations in this loop
                 * So, further iterations will be reusing workspace memory over and over again
                 */
            }
        }


        /**
         * Workspaces can be nested. And INDArrays can migrate between them, if needed
         */

        try(MemoryWorkspace ws1 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID")) {
            INDArray array = Nd4j.create(10, 10).assign(1.0f);
            INDArray sumRes;

            try(MemoryWorkspace ws2 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "THIRD_ID")) {
                // PLEASE NOTE: we can access memory from parent workspace without any issues ONLY if it wasn't closed/reset yet.
                INDArray res = array.sum(1);

                // array is allocated at ws1, and res is allocated in ws2. But we can migrate them if/when needed.
                sumRes = res.leverageTo("SOME_ID");
            }

            // at this point sumRes contains valid data, allocated in current workspace. We expect 100 printed here.
            log.info("Sum: {}", sumRes.sumNumber().floatValue());
        }


        /**
         * You can break your workspace flow, if, for some reason you need part of calculations to be handled with GC
         */
        try(MemoryWorkspace ws1 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID")) {
            INDArray array1 = Nd4j.create(10, 10).assign(1.0f);
            INDArray array2;

            try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                // anything allocated within this try block will be managed by GC

                array2 = Nd4j.create(10, 10).assign(2.0f);
            }


            // at this point sumRes contains valid data, allocated in current workspace. We expect 300 printed here.
            log.info("Sum: {}", array1.addi(array2).sumNumber().floatValue());
        }


        /**
         * It's also possible to build workspace that'll be acting as circular buffer.
         */
        WorkspaceConfiguration circularConfig = WorkspaceConfiguration.builder()
            .initialSize(10 * 1024L * 1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.NONE)    // <-- this options disables workspace reallocation over time
            .policyReset(ResetPolicy.ENDOFBUFFER_REACHED) // <--- this option makes workspace act as circular buffer, beware.
            .build();

        for (int x = 0; x < 10; x++) {
            // since this workspace is ciruclar, we know that all pointers allocated before buffer ended - will be viable.
            try (MemoryWorkspace ws1 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(circularConfig, "CIRCULAR_ID")) {
                INDArray array = Nd4j.create(100);
                // so, you can use this array anywhere as long as YOU're sure buffer wasn't reset.
                // in other words: it's suitable for producer/consumer pattern use if you're in charge of flow
            }
        }
    }
}

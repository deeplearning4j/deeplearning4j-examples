package org.nd4j.examples;

import org.junit.jupiter.api.Test;
import org.nd4j.examples.advanced.lowlevelmodeling.MultiClassLogitExample;
import org.nd4j.examples.advanced.memoryoptimization.WorkspacesExample;
import org.nd4j.examples.advanced.operations.CustomOpsExamples;

public class QuickTest {


    @Test
    public void runTestExamples() throws Exception {
        System.out.println("Beginning MultiClassLogitExample");
        MultiClassLogitExample.main(new String[]{});
        System.out.println("Ending WorkspacesExample");
        WorkspacesExample.main(new String[]{});
        System.out.println("Beginning WorkspacesExample");
        System.out.println("Ending MultiClassLogitExample");
        CustomOpsExamples.main(new String[]{});


    }


}

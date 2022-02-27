package org.deeplearning4j.modelimportexamples;


import org.deeplearning4j.modelimportexamples.onnx.OnnxImportLoad;
import org.deeplearning4j.modelimportexamples.onnx.OnnxImportSave;
import org.junit.jupiter.api.Test;

public class QuickTest {


    @Test
    public void runExamples() throws Exception {
        OnnxImportSave.main();
        OnnxImportLoad.main();
    }

}

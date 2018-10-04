package org.deeplearning4j.patent;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.spark.time.TimeSourceProvider;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

public class ParamsCheckpointListener extends BaseTrainingListener implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(ParamsCheckpointListener.class);

    private final File localDir;
    private final long saveFreqMS;
    private transient long lastSaveTime;

    public ParamsCheckpointListener(File localDir, long saveFreqMS){
        this.localDir = localDir;
        this.saveFreqMS = saveFreqMS;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        long currTime = TimeSourceProvider.getInstance().currentTimeMillis();       //NTP time source by default, should be closely synced on each machine

        long lastSaveShouldBe = currTime - currTime % saveFreqMS;       //round down to nearest even multiple of saveFreqMS
        if(lastSaveTime <= lastSaveShouldBe){
            //Save parameters
            String uid = UIDProvider.getJVMUID().substring(0,6);
            String fileName = "params_jvm-" + uid + "_time-" + lastSaveShouldBe + "_actTime-" + currTime + "_epoch-" + epoch + "_iter-" + iteration + ".bin";
            if(!localDir.exists()){
                localDir.mkdirs();
            }
            File f = new File(localDir, fileName);
            INDArray params = model.params();
            try {
                Nd4j.saveBinary(params, f);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
            log.info("Saved parameters to file: {}", f.getAbsolutePath());
            lastSaveTime = currTime;
        }

    }

}

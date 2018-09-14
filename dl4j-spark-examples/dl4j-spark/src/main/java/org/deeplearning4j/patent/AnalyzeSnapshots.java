package org.deeplearning4j.patent;

import com.beust.jcommander.Parameter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.bytedeco.javacpp.annotation.Allocator;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class AnalyzeSnapshots {
    private static Logger log = LoggerFactory.getLogger(AnalyzeSnapshots.class);

    @Parameter(names = {"--snapshotsDir"}, description = "Local directory containing parameter snapshots", required = true)
    private String snapshotsDir = null;

    public static void main(String[] args) throws Exception {
        new AnalyzeSnapshots().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);


        //
        /*
        Expected folder format:
        snapshotsDir/10.0.2.5/params_jvm-<UID>_time-1536891502000_actTime-1536891502134_epoch-0_iter-313.bin
         */

        //Step 1: group snapshots based on time
        File rootDir = new File(snapshotsDir);
        List<Snapshot> allSnapshots = new ArrayList<>();
        Files.walk(rootDir.toPath())
            .filter(Files::isRegularFile)
            .forEach(p -> {
                File f = p.toFile();
                if(!f.isDirectory()){
                    allSnapshots.add(parseFileName(f));
                }
            });

        log.info("***** Found {} snapshot files *****", allSnapshots.size());

        Map<Long,Map<String,Snapshot>> groupedByTime = new HashMap<>();
        for(Snapshot s : allSnapshots){
            Map<String,Snapshot> m = groupedByTime.computeIfAbsent(s.saveTime, f -> new HashMap<>());
            if(m.containsKey(s.host)){
                throw new IllegalStateException("Duplicate host: " + s);
            }
            m.put(s.getHost(), s);
        }

        List<Long> allTimes = new ArrayList<>(groupedByTime.keySet());
        Collections.sort(allTimes);

        log.info("***** {} unique timestamps *****", allTimes.size());

        for(Long t : allTimes){
            Map<String,Snapshot> snapshotsForTime = groupedByTime.get(t);

            /*
            Need to compare snapshots. Approach used here: compare all n vs. n parameter arrays.
            (note that differences are symmetric)
            Calculate element-wise differences: max, mean?, 95 percentile, etc
            */

            double[][] diffMax = new double[8][8];
            double[][] diffMean = new double[8][8];
            double[][] diffP95 = new double[8][8];
            double[][] diffP99 = new double[8][8];
            double[][] diffP999 = new double[8][8];

            Map<String,INDArray> snapshotsLoaded = new HashMap<>();
            for(String s : snapshotsForTime.keySet()){
                snapshotsLoaded.put(s, snapshotsForTime.get(s).load());
            }

            for(int i=0; i<8; i++ ){
                for( int j=i; j<8; j++ ){
                    if(i == j){
                        diffMax[i][j] = 0;
                        diffMean[i][j] = 0;
                        diffP95[i][j] = 0;
                        diffP99[i][j] = 0;
                        diffP999[i][j] = 0;
                    } else {
                        String host1 = "10.0.2." + (5+i);
                        String host2 = "10.0.2." + (5+j);
                        if(!snapshotsLoaded.containsKey(host1) || !snapshotsLoaded.containsKey(host2)){
                            diffMax[i][j] = Double.NaN;
                            diffMean[i][j] = Double.NaN;
                            diffP95[i][j] = Double.NaN;
                            diffP99[i][j] = Double.NaN;
                            diffP999[i][j] = Double.NaN;
                        } else {
                            INDArray i1 = snapshotsLoaded.get(host1);
                            INDArray i2 = snapshotsLoaded.get(host2);

                            INDArray absDiff = Transforms.abs(i1.sub(i2), false);
                            diffMax[i][j] = absDiff.maxNumber().doubleValue();
                            diffMean[i][j] = absDiff.meanNumber().doubleValue();
//                            diffP95[i][j] = Double.NaN;
//                            diffP99[i][j] = Double.NaN;
//                            diffP999[i][j] = Double.NaN;
                        }
                    }
                }
            }
        }

    }

    private static Snapshot parseFileName(File file){
        //snapshotsDir/10.0.2.5/params_jvm-<UID>_time-1536891502000_actTime-1536891502134_epoch-0_iter-313.bin
        String path = file.getAbsolutePath().replaceAll("\\\\", "/");
        String[] split = path.split("/");
        String host = split[split.length-2];
        String filename = split[split.length-1];

        String[] nameSplit = filename.split("_");
        String jvmuid = nameSplit[1].substring(4);
        long saveTime = Long.parseLong(nameSplit[2].substring(5));
        long actTime = Long.parseLong(nameSplit[3].substring(5));

        return new Snapshot(file, host, jvmuid, saveTime, actTime);
    }

    @AllArgsConstructor
    @Data
    private static class Snapshot {
        private final File file;
        private final String host;
        private final String jvmuid;
        private final long saveTime;        //Optimal save time - exactly every 30 seconds, etc
        private final long actTime;         //Actual save time - when params were actually saved

        private INDArray load(){
            try {
                return Nd4j.readBinary(file);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }

}

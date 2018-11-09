package org.deeplearning4j.resnet50Benchmark;

import com.beust.jcommander.Parameter;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.util.SparkDataUtils;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * ----- Instructions -----
 * -- Step 1 --
 * Before running this preprocessing class, you will need to download one of the ImageNet ILSVRC2012 datasets MANUALLY from the following URL:
 * http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
 * Note that this script can be used with either the training OR validation sets (not the test set, as no labels are available).
 * Training set:    ILSVRC2012_img_train.tar
 * Validation set:  ILSVRC2012_img_val.tar
 *
 * For benchmarking purposes, (calculating images/batches per second for training), the validation set is recommended as
 * it is the smallest download that has labels.
 * (Clearly, the validation set should NOT be used when benchmarking convergence - i.e., accuracy vs. training time, but
 * this benchmark does not measure convergence speed, only examples/sec and batches/sec).
 *
 * -- Step 2 --
 * Once downloaded, run the class, with the following arguments:
 * (a) --sourceFile: the location (file on your local file system) to preprocess - either ILSVRC2012_img_val.tar or ILSVRC2012_img_train.tar
 * (b) --localSaveDir: where the processed data should be placed
 * (c) [OPTIONAL] --batchSize: if the batch size needs to be configured (default: 128)
 *
 *
 */
public class PreprocessLocal {
    public static final String TRAIN_MD5_TASK12 = "1d675b47d978889d74fa0da5fadfb00e";      //ILSVRC2012_img_train.tar
    public static final String VALIDATE_MD5 = "29b22e2961454d5413ddabcf34fc5622";          //ILSVRC2012_img_val.tar
    public static final String DEVKIT_FILE = "ILSVRC2012_devkit_t12.tar.gz";

    public static final String VALIDATION_LABEL_MAPPING_FILENAME = "imagenet_2012_validation_synset_labels.txt";
    public static final String VALIDATION_LABEL_MAPPING_FILE = "https://deeplearning4jblob.blob.core.windows.net/resources/imagenet/imagenet_2012_validation_synset_labels.txt";

    @Parameter(names = {"--sourceFile"}, description = "Path to dataset to process", required = true)
    private String sourceFile = null;

    @Parameter(names = {"--localSaveDir"}, description = "Directory to save the preprocessed data files on your local drive", required = true)
    private String localSaveDir = null;

    @Parameter(names = {"--batchSize"}, description = "Batch size for saving the data", required = false)
    private int batchSize = 128;

    public static void main(String[] args) throws Exception {
        new PreprocessLocal().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        //Validate file exists and has an expected MD5
        File f = new File(sourceFile);
        if(!f.exists() || !f.isFile()){
            throw new IllegalStateException("Could not find file at " + f.getAbsolutePath());
        }
        String md5 = md5(f);
        if(!TRAIN_MD5_TASK12.equals(md5) && !VALIDATE_MD5.equals(md5) ){
            throw new IllegalStateException("MD5 of file " + f.getAbsolutePath() + " does not match the MD5 of any expected" +
                " ImageNet ILSVRC2012 challenge files - got " + md5);
        }

        boolean validation = VALIDATE_MD5.equals(md5);

        //Extract files
        File rawImagesDir = new File(localSaveDir, "rawImages_" + (validation ? "validation" : "train"));
        if(rawImagesDir.exists())
            FileUtils.forceDelete(rawImagesDir);
        rawImagesDir.mkdirs();
        extractTarFile(f, rawImagesDir);

        if(validation){
            //Rearrange files to the same format as imagenet training set - i.e., parent directory is the class label
            rearrangeValidationSet(new File(localSaveDir));
            rawImagesDir = new File(localSaveDir, "validation" );
        }

        //Prepare files for Spark training
        File outDir = new File(localSaveDir, "preprocessedImages");
        outDir.mkdirs();
        SparkDataUtils.createFileBatchesLocal(rawImagesDir, NativeImageLoader.ALLOWED_FORMATS, true, outDir, batchSize);
        System.out.println("Output files written to: " + outDir.getAbsolutePath());

        System.out.println("----- Preprocessing Complete -----");
    }


    private static String md5(File file) throws IOException {
        try(InputStream in = FileUtils.openInputStream(file)) {
            return DigestUtils.md5Hex(in);
        }
    }

    private static void extractTarFile(File source, File destination) throws IOException{
        int fileCount = 0;
        int BUFFER = 32769;
        byte data[] = new byte[BUFFER];
        long start = System.currentTimeMillis();
        try(TarArchiveInputStream tarIn = new TarArchiveInputStream(new BufferedInputStream(new FileInputStream(source)))){

            TarArchiveEntry entry;
            while ((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null) {
                if (entry.isDirectory()) {
                    File f = new File(destination, entry.getName());
                    f.mkdirs();
                }

                // If the entry is a file, write the decompressed file to the disk and close destination stream.
                else {
                    int byteCount;
                    try (FileOutputStream fos = new FileOutputStream(new File(destination, entry.getName()));
                         BufferedOutputStream destStream = new BufferedOutputStream(fos, BUFFER);) {
                        while ((byteCount = tarIn.read(data, 0, BUFFER)) != -1) {
                            destStream.write(data, 0, byteCount);
                        }
                    }
                }
                fileCount++;
            }
        }
        long end = System.currentTimeMillis();

        System.out.println("Extracted " + fileCount + " files in " + (end-start)/1000 + " seconds to directory " + destination.getAbsolutePath());
    }

    private static void rearrangeValidationSet(File localSaveDir) throws Exception {
        File validationLabelsFile = new File(localSaveDir, VALIDATION_LABEL_MAPPING_FILENAME);
        if(!validationLabelsFile.exists()){
            FileUtils.copyURLToFile(new URL(VALIDATION_LABEL_MAPPING_FILE), validationLabelsFile);
        }

        File rawImagesRoot = new File(localSaveDir, "rawImages_validation");
        File[] list = rawImagesRoot.listFiles();
        if(list == null || list.length != 50000){
            throw new IllegalStateException("Expected exactly 50000 validation set images, got " + (list == null ? 0 : list.length));
        }

        File shuffledImagesRoot = new File(localSaveDir, "validation");
        List<String> mapping = FileUtils.readLines(validationLabelsFile, StandardCharsets.UTF_8);
        for(File f : list){
            String path = f.getAbsolutePath();
            String[] split = path.split("[_.]");
            int number = Integer.parseInt(split[split.length-2]);
            String label = mapping.get(number-1);
            File parentDir = new File(shuffledImagesRoot, label);
            if(!parentDir.exists()){
                parentDir.mkdirs();
            }
            File to = new File(parentDir, f.getName());
            FileUtils.moveFile(f, to);
        }
    }
}

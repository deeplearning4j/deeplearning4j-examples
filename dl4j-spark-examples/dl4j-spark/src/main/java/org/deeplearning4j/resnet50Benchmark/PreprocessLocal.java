package org.deeplearning4j.resnet50Benchmark;

import com.beust.jcommander.Parameter;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.spark.util.SparkDataUtils;
import org.nd4j.util.ArchiveUtils;

import java.io.*;
import java.net.URI;

/**
 * Instructions:
 * 1. Before running this preprocessing class, you will need to download one of the ImageNet ILSVRC2012 files MANUALLY from the following URL:
 * http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
 * Note that this script can be used with any of the sets (training, validation, test) - for benchmarking purposes when
 * looking at the throughput (images/batches per second), the validation set is recommended as it is the smallest download.
 * (Clearly, the validation set should NOT be used when benchmarking convergence - i.e., accuracy vs. training time).
 *
 * 2. Once downloaded, run the class, with the following arguments:
 * (a) --sourcePath: the location (on your local file system) of the downloaded file
 * (b) --localSaveDir: where the processed data should be placed
 * (c) [OPTIONAL] --batchSize: if the batch size needs to be configured (default: 128)
 *
 *
 */
public class PreprocessLocal {
    private static final String TRAIN_MD5_TASK12 = "1d675b47d978889d74fa0da5fadfb00e";      //ILSVRC2012_img_train.tar
    private static final String TRAIN_MD5_TASK3 = "ccaf1013018ac1037801578038d370da";       //ILSVRC2012_img_train_t3.tar
    private static final String VALIDATE_MD5 = "29b22e2961454d5413ddabcf34fc5622";          //ILSVRC2012_img_val.tar
    private static final String TEST_MD5 = "fe64ceb247e473635708aed23ab6d839";              //ILSVRC2012_img_test.tar

    @Parameter(names = {"--sourcePath"}, description = "Path to dataset to process", required = true)
    private String sourcePath = null;

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
        File f = new File(sourcePath);
        if(!f.exists() || !f.isFile()){
            throw new IllegalStateException("Could not find file at " + f.getAbsolutePath());
        }
        String md5 = md5(f);
        if(!TRAIN_MD5_TASK12.equals(md5) && !TRAIN_MD5_TASK3.equals(md5) && !VALIDATE_MD5.equals(md5) && !TEST_MD5.equals(md5)){
            throw new IllegalStateException("MD5 of file " + f.getAbsolutePath() + " does not match the MD5 of any expected ImageNet ILSVRC2012 challenge files - got " + md5);
        }


        //Extract files
        File rawImagesDir = new File(localSaveDir, "rawImages");
        rawImagesDir.mkdirs();
        if(rawImagesDir.exists())
            FileUtils.forceDelete(rawImagesDir);
        extractTarFile(f, rawImagesDir);

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
}

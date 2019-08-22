package org.deeplearning4j.examples.download;

import org.apache.commons.io.FilenameUtils;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.net.URL;

/**
 * @author susaneraly
 */
public enum DownloaderUtility {

    /*
        datavec-examples resources
     */
    BASICDATAVECEXAMPLE("datavec-examples", "BasicDataVecExample.zip", "92f87e0ceb81093ff8b49e2b4e0a5a02", "1KB"),
    INPUTSPLIT("datavec-examples", "inputsplit.zip", "f316b5274bab3b0f568eded9bee1c67f", "128KB"),
    IRISDATA("datavec-examples","IrisData.zip","bb49e38bb91089634d7ef37ad8e430b8","1KB"),
    JOINEXAMPLE("datavec-examples", "JoinExample.zip", "cbd6232cf1463d68ff24807d5dd8b530", "1KB"),
    /*
        dl4j-example resources
     */
    ANIMALS("dl4j-examples", "animals.zip", "1976a1f2b61191d2906e4f615246d63e", "820KB"),
    ANOMALYSEQUENCEDATA("dl4j-examples", "anomalysequencedata.zip", "51bb7c50e265edec3a241a2d7cce0e73", "3MB"),
    CAPTCHAIMAGE("dl4j-examples","captchaImage.zip","1d159c9587fdbb1cbfd66f0d62380e61","42MB"),
    CLASSIFICATIONDATA("dl4j-examples","classification.zip","dba31e5838fe15993579edbf1c60c355","77KB"),
    DATAEXAMPLES("dl4j-examples","DataExamples.zip","e4de9c6f19aaae21fed45bfe2a730cbb","2MB"),
    LOTTERYDATA("dl4j-examples","lottery.zip","1e54ac1210e39c948aa55417efee193a","2MB"),
    MODELIMPORT("dl4j-examples","modelimport.zip","411df05aace1c9ff587e430a662ce621","3MB"),
    NEWSDATA("dl4j-examples","NewsData.zip","0d08e902faabe6b8bfe5ecdd78af9f64","21MB"),
    NLPDATA("dl4j-examples","nlp.zip","1ac7cd7ca08f13402f0e3b83e20c0512","91MB"),
    PREDICTGENDERDATA("dl4j-examples","PredictGender.zip","42a3fec42afa798217e0b8687667257e","3MB"),
    STYLETRANSFER("dl4j-examples","styletransfer.zip","b2b90834d667679d7ee3dfb1f40abe94","3MB");

    private final String DATA_FOLDER;
    private final String ZIP_FILE;
    private final String MD5;
    private final String DATA_SIZE;

    DownloaderUtility(String dataFolder, String zipFile, String md5, String dataSize) {
        DATA_FOLDER = dataFolder;
        ZIP_FILE = zipFile;
        MD5 = md5;
        DATA_SIZE = dataSize;
    }

    public String Download() throws Exception {
        String dataURL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/" + DATA_FOLDER + "/" + ZIP_FILE;
        String resourceName = ZIP_FILE.substring(0, ZIP_FILE.lastIndexOf(".zip"));
        String downloadPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), ZIP_FILE);
        String extractDir = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/" + DATA_FOLDER);
        String dataPathLocal = FilenameUtils.concat(extractDir,resourceName);
        int downloadRetries = 10;
        if (!new File(dataPathLocal).exists()) {
                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data ("+DATA_SIZE+") and extracting to \n\t" + dataPathLocal);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(dataURL),
                    new File(downloadPath),
                    new File(extractDir),
                    MD5,
                    downloadRetries);
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + dataPathLocal);
            System.out.println("_______________________________________________________________________");
        }
        return dataPathLocal;
    }
}

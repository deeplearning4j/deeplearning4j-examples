/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.download;

import org.apache.commons.io.FilenameUtils;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.net.URL;

/**
 * Given a base url and a zipped file name downloads contents to a specified directory under ~/dl4j-examples-data
 * Will check md5 sum of downloaded file
 * <p>
 *
 * Sample Usage with an instantiation DATAEXAMPLE(baseurl,"DataExamples.zip","data-dir",md5,size):
 *
 * DATAEXAMPLE.Download() & DATAEXAMPLE.Download(true)
 * Will download DataExamples.zip from baseurl/DataExamples.zip to a temp directory,
 * Unzip it to ~/dl4j-example-data/data-dir
 * Return the string "~/dl4j-example-data/data-dir/DataExamples"
 *
 * DATAEXAMPLE.Download(false)
 * will perform the same download and unzip as above
 * But returns the string "~/dl4j-example-data/data-dir" instead
 *
 *
 * @author susaneraly
 */
public enum DownloaderUtility {

    /*
        Skymind datavec resources stored under AZURE_BLOB_URL/datavec-examples
     */
    BASICDATAVECEXAMPLE("BasicDataVecExample.zip", "datavec-examples", "92f87e0ceb81093ff8b49e2b4e0a5a02", "1KB"),
    INPUTSPLIT("inputsplit.zip", "datavec-examples", "f316b5274bab3b0f568eded9bee1c67f", "128KB"),
    IRISDATA("IrisData.zip", "datavec-examples", "bb49e38bb91089634d7ef37ad8e430b8", "1KB"),
    JOINEXAMPLE("JoinExample.zip", "datavec-examples", "cbd6232cf1463d68ff24807d5dd8b530", "1KB"),
    /*
        Skymind dl4j-examples resources stored under AZURE_BLOB_URL/dl4j-examples
     */
    ANIMALS("animals.zip", "dl4j-examples", "1976a1f2b61191d2906e4f615246d63e", "820KB"),
    ANOMALYSEQUENCEDATA("anomalysequencedata.zip", "dl4j-examples", "51bb7c50e265edec3a241a2d7cce0e73", "3MB"),
    CAPTCHAIMAGE("captchaImage.zip", "dl4j-examples", "1d159c9587fdbb1cbfd66f0d62380e61", "42MB"),
    CLASSIFICATIONDATA("classification.zip", "dl4j-examples", "dba31e5838fe15993579edbf1c60c355", "77KB"),
    DATAEXAMPLES("DataExamples.zip", "dl4j-examples", "e4de9c6f19aaae21fed45bfe2a730cbb", "2MB"),
    LOTTERYDATA("lottery.zip", "dl4j-examples", "1e54ac1210e39c948aa55417efee193a", "2MB"),
    MODELIMPORT("modelimport.zip", "dl4j-examples", "411df05aace1c9ff587e430a662ce621", "3MB"),
    NEWSDATA("NewsData.zip", "dl4j-examples", "0d08e902faabe6b8bfe5ecdd78af9f64", "21MB"),
    NLPDATA("nlp.zip", "dl4j-examples", "1ac7cd7ca08f13402f0e3b83e20c0512", "91MB"),
    PREDICTGENDERDATA("PredictGender.zip", "dl4j-examples", "42a3fec42afa798217e0b8687667257e", "3MB"),
    STYLETRANSFER("styletransfer.zip", "dl4j-examples", "b2b90834d667679d7ee3dfb1f40abe94", "3MB"),
    //This download is handled a little differently since the zip is not a single directory but a bunch of stuff at the top level
    BERTEXAMPLE("https://dl4jdata.blob.core.windows.net/testresources", "bert_mrpc_frozen_v1.zip", "bert-frozen-example", "7cef8bbe62e701212472f77a0361f443", "420MB"),

    /*
        Skymind tf-import-examples resources stored under AZURE_BLOB_URL/tf-import-examples
     */
    TFIMPORTEXAMPLES("resources.zip", "tf-import-examples", "4895e40e71b17799e4d6fb75d5a22491", "3MB"),

    /*
        Skymind dl4j-spark example resources stored under AZURE_BLOB_URL/dl4j-spark-examples
     */
    PATENTEXAMPLE("patentExample.zip", "dl4j-spark-examples", "435e2b814d866550678d2ac4d8cc5423", "10KB");

    private final String BASE_URL;
    private final String DATA_FOLDER;
    private final String ZIP_FILE;
    private final String MD5;
    private final String DATA_SIZE;
    private static final String AZURE_BLOB_URL = "https://dl4jdata.blob.core.windows.net/dl4j-examples";

    /**
     * For use with resources uploaded to Azure blob storage.
     *
     * @param zipFile    Name of zipfile. Should be a zip of a single directory with the same name
     * @param dataFolder The folder to extract to under ~/dl4j-examples-data
     * @param md5        of zipfile
     * @param dataSize   of zipfile
     */
    DownloaderUtility(String zipFile, String dataFolder, String md5, String dataSize) {
        this(AZURE_BLOB_URL + "/" + dataFolder, zipFile, dataFolder, md5, dataSize);
    }

    /**
     * Downloads a zip file from a base url to a specified directory under the user's home directory
     *
     * @param baseURL    URL of file
     * @param zipFile    Name of zipfile to download from baseURL i.e baseURL+"/"+zipFile gives full URL
     * @param dataFolder The folder to extract to under ~/dl4j-examples-data
     * @param md5        of zipfile
     * @param dataSize   of zipfile
     */
    DownloaderUtility(String baseURL, String zipFile, String dataFolder, String md5, String dataSize) {
        BASE_URL = baseURL;
        DATA_FOLDER = dataFolder;
        ZIP_FILE = zipFile;
        MD5 = md5;
        DATA_SIZE = dataSize;
    }

    public String Download() throws Exception {
        return Download(true);
    }

    public String Download(boolean returnSubFolder) throws Exception {
        String dataURL = BASE_URL + "/" + ZIP_FILE;
        String downloadPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), ZIP_FILE);
        String extractDir = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/" + DATA_FOLDER);
        if (!new File(extractDir).exists())
            new File(extractDir).mkdirs();
        String dataPathLocal = extractDir;
        if (returnSubFolder) {
            String resourceName = ZIP_FILE.substring(0, ZIP_FILE.lastIndexOf(".zip"));
            dataPathLocal = FilenameUtils.concat(extractDir, resourceName);
        }
        int downloadRetries = 10;
        if (!new File(dataPathLocal).exists() || new File(dataPathLocal).list().length == 0) {
            System.out.println("_______________________________________________________________________");
            System.out.println("Downloading data (" + DATA_SIZE + ") and extracting to \n\t" + dataPathLocal);
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

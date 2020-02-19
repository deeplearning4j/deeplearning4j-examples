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
 * Assumption is zipped file contains a single directory with the same name as the zip file
 *
 * eg. With an instantiation DATAEXAMPLE(baseurl,"DataExamples.zip","data-dir",md5,size)
 * DATAEXAMPLE.Download() will
 *      download DataExamples.zip from baseurl/DataExamples.zip to a temp directory,
 *      unzip it to ~/dl4j-example-data/data-dir
 *      and return the unzipped path ~/dl4j-example-data/data-dir/DataExamples
 *
 * @author susaneraly
 */
public enum DownloaderUtility {

    /*
        Skymind dl4j-examples resources stored under AZURE_BLOB_URL/dl4j-examples
     */
    DATAEXAMPLES("DataExamples.zip","dl4j-examples","e4de9c6f19aaae21fed45bfe2a730cbb","2MB");

    private final String BASE_URL;
    private final String DATA_FOLDER;
    private final String ZIP_FILE;
    private final String MD5;
    private final String DATA_SIZE;
    private static final String AZURE_BLOB_URL = "https://dl4jdata.blob.core.windows.net/dl4j-examples";

    /**
     * For use with resources uploaded to Azure blob storage.
     * @param zipFile Name of zipfile. Should be a zip of a single directory with the same name
     * @param dataFolder The folder to extract to under ~/dl4j-examples-data
     * @param md5 of zipfile
     * @param dataSize of zipfile
     */
    DownloaderUtility(String zipFile, String dataFolder, String md5, String dataSize) {
       this(AZURE_BLOB_URL + "/" + dataFolder , zipFile, dataFolder, md5, dataSize);
    }

    /**
     * Downloads a zip file from a base url to a specified directory under the user's home directory
     * @param baseURL URL of file
     * @param zipFile Name of zipfile to download from baseURL i.e baseURL+"/"+zipFile gives full URL
     * @param dataFolder The folder to extract to under ~/dl4j-examples-data
     * @param md5 of zipfile
     * @param dataSize of zipfile
     */
    DownloaderUtility(String baseURL, String zipFile, String dataFolder, String md5, String dataSize) {
        BASE_URL = baseURL;
        DATA_FOLDER = dataFolder;
        ZIP_FILE = zipFile;
        MD5 = md5;
        DATA_SIZE = dataSize;
    }

    public String Download() throws Exception {
        String dataURL = BASE_URL + "/" + ZIP_FILE;
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

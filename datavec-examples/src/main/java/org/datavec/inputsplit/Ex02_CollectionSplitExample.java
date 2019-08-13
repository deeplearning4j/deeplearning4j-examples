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

package org.datavec.inputsplit;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.Iterator;

/**
 * {@link org.datavec.api.split.CollectionInputSplit} is a basic implementation of
 * {@link org.datavec.api.split.InputSplit} that's useful when we want to create InputSplit from a list/collection of
 * URIs
 */
public class Ex02_CollectionSplitExample {

    public static final String DATA_LOCAL_PATH;

    static {
        final String DATA_URL = "https://deeplearning4jblob.blob.core.windows.net/dl4j-examples/datavec-examples/inputsplit.zip";
        final String MD5 = "f316b5274bab3b0f568eded9bee1c67f";
        final int DOWNLOAD_RETRIES = 10;
        final String DOWNLOAD_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "inputsplit.zip");
        final String EXTRACT_DIR = FilenameUtils.concat(System.getProperty("user.home"), "dl4j-examples-data/datavec-examples");
        DATA_LOCAL_PATH = FilenameUtils.concat(EXTRACT_DIR, "inputsplit");
        if (!new File(DATA_LOCAL_PATH).exists()) {
            try {

                System.out.println("_______________________________________________________________________");
                System.out.println("Downloading data (128KB) and extracting to \n\t" + DATA_LOCAL_PATH);
                System.out.println("_______________________________________________________________________");
                Downloader.downloadAndExtract("files",
                    new URL(DATA_URL),
                    new File(DOWNLOAD_PATH),
                    new File(EXTRACT_DIR),
                    MD5,
                    DOWNLOAD_RETRIES);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("_______________________________________________________________________");
            System.out.println("Example data present in \n\t" + DATA_LOCAL_PATH);
            System.out.println("_______________________________________________________________________");
        }

    }

    public static void main(String[] args) throws Exception {
        File directoryToLook = new File(DATA_LOCAL_PATH, "files/cats");

        /*
          Creating a FileSplit this just to receive a list of URIs. From those URIs we'll create the CollectionInputSplit.
          Specify the allowed extensions using an array of String where each entry denotes an extension to be included.
          Added extensions with an intuitive approach. eg: ".jpg", ".png" etc.
         */
        FileSplit fileSplit = new FileSplit(directoryToLook, new String[]{".jpg"}, false);

        /*
          Now you can create the CollectionInputSplit and print it as follows.
         */
        CollectionInputSplit collectionInputSplit = new CollectionInputSplit(fileSplit.locations());
        System.out.println("--------------- Printing the input splits from CollectionInputSplit ---------------");
        Iterator<URI> collectionInputSplitIterator = collectionInputSplit.locationsIterator();
        while (collectionInputSplitIterator.hasNext()) {
            System.out.println(collectionInputSplitIterator.next());
        }
        System.out.println("---------------------------------------------------------------------------\n\n\n");
    }
}

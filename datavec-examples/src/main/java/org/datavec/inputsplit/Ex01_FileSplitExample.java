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
import org.datavec.api.split.FileSplit;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.Iterator;
import java.util.Random;


/**
 * {@link org.datavec.api.split.InputSplit} and its implementations are utility classes for defining and managing a catalog of
 * loadable locations (paths/files), in memory, that can later be exposed through an {{@link java.util.Iterator}}.
 * It also provides methods for exposing the locations through URIs. InputSplit also contains utilities for
 * opening up {@link java.io.InputStream} and {@link java.io.OutputStream}, given the location.
 * <p>
 * In simple terms, they define where your data comes from or should be saved to, when building a data pipeline with DataVec
 * <p>
 * In this example, we'll see the basic implementation and usages of the {@link org.datavec.api.split.FileSplit},
 * which is implemented from {@link org.datavec.api.split.BaseInputSplit}, which is further implemented from
 * {@link org.datavec.api.split.InputSplit}
 */

public class Ex01_FileSplitExample {


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
        File directoryToLook = new File(DATA_LOCAL_PATH, "files");

        //=====================================================================
        //                 Example 1: Loading everything within
        //=====================================================================

        /*
          This will gather all the loadable files within the specified directory. By default it will load all the files
          regardless of the extensions they have. Also, it will search for the inner directories recursively for
          further loadable files.
         */
        FileSplit fileSplit1 = new FileSplit(directoryToLook);

        /*
          We can view the files in the file split by using the FileSplit#locations function
         */

        System.out.println("--------------- Example 1: Loading every file ---------------");
        URI[] fileSplit1Uris = fileSplit1.locations();
        for (URI uri : fileSplit1Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 2: Loading non-recursively
        //=====================================================================

        /*
          To load the files without the recursive search, you can specify 'false' for the recursive flag in the
          FileSplit's constructor
         */
        FileSplit fileSplit2 = new FileSplit(directoryToLook, null, false);

        /*
          Now the loaded locations will just contain the files in the immediate directory that was specified
         */
        System.out.println("--------------- Example 2: Loading non-recursively ---------------");
        URI[] fileSplit2Uris = fileSplit2.locations();
        for (URI uri : fileSplit2Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 3: Loading with filters
        //=====================================================================

        /*
          You can also filter the files by their extensions. Just specify the files extensions or specify the substring
          of the files to which they are ending with. Here, we'll keep the recursive flag as 'false'.
         */
        String[] extensionsToFilter = new String[]{".jpg"};
        FileSplit fileSplit3 = new FileSplit(directoryToLook, extensionsToFilter, false);

        /*
          This will load all the image files with just the 'jpg' extension
         */
        System.out.println("--------------- Example 3: Loading with filters ---------------");
        URI[] fileSplit3Uris = fileSplit3.locations();
        for (URI uri : fileSplit3Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 4: Loading with a random seed
        //=====================================================================

        /*
          You can also load the files with a random seed. This is a good practice for keeping consistency while loading
          files.
         */
        FileSplit fileSplit4 = new FileSplit(directoryToLook, extensionsToFilter, new Random(222));

        /*
          This will load all the jpg files recursively and randomly, specified by the random seed. Beware that this
          randomization will only be reflected when you use the Iterator.
         */
        System.out.println("--------------- Example 4: Loading with a random seed ---------------");
        Iterator<URI> fileSplit4UrisIterator = fileSplit4.locationsIterator();
        while (fileSplit4UrisIterator.hasNext()) {
            System.out.println(fileSplit4UrisIterator.next());
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 5: FileSplit with a single file
        //=====================================================================

        /*
          This example will show that you can point a FileSplit to a single file and have
          it do basically what you'd expect.
         */
        FileSplit fileSplit5 = new FileSplit(new File(directoryToLook.getAbsolutePath(), "cats/domestic_cat_s_001970.jpg"));

        /*
          This will print the single file uri we've specified through the class path resource.
         */
        System.out.println("--------------- Example 5: FileSplit with a single file ---------------");
        Iterator<URI> fileSplit5UrisIterator = fileSplit5.locationsIterator();
        while (fileSplit5UrisIterator.hasNext()) {
            System.out.println(fileSplit5UrisIterator.next());
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }

}

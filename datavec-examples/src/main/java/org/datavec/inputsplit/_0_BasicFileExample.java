package org.datavec.inputsplit;

import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.net.URI;
import java.util.Iterator;
import java.util.Random;

/**
 * {@link org.datavec.api.split.InputSplit} and its implementation are utility classes for managing a catalog of
 * loadable locations (paths/files), in memory, that can later be exposed through an {{@link java.util.Iterator}}.
 * It also provides methods for exposing the locations through URIs. InputSplit also contains utilities for
 * opening up {@link java.io.InputStream} and {@link java.io.OutputStream}, given the location.
 *
 * In this example, we'll see the basic implementation and usages of the {@link org.datavec.api.split.FileSplit},
 * which is implemented from {@link org.datavec.api.split.BaseInputSplit}, which is further implemented from
 * {@link org.datavec.api.split.InputSplit}
 */

public class _0_BasicFileExample {
    public static void main(String[] args) throws Exception {
        // Receive the class path resource from the resource folder
        ClassPathResource classPathResource1 = new ClassPathResource("inputsplit/files/");
        File directoryToLook = classPathResource1.getFile();

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
        for (URI uri: fileSplit1Uris) {
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
        for (URI uri: fileSplit2Uris) {
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
        for (URI uri: fileSplit3Uris) {
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
    }
}

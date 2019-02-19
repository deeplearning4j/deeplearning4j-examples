package org.datavec.inputsplit;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.net.URI;
import java.util.Iterator;

/**
 * {@link org.datavec.api.split.CollectionInputSplit} is a basic implementation of
 * {@link org.datavec.api.split.InputSplit} that's useful when we want to create InputSplit from a list/collection of
 * URIs
 */
public class Ex02_CollectionSplitExample {
    public static void main(String[] args) throws Exception{
        // Receive the class path resource from the resource folder
        ClassPathResource classPathResource1 = new ClassPathResource("inputsplit/files/cats");
        File directoryToLook = classPathResource1.getFile();

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

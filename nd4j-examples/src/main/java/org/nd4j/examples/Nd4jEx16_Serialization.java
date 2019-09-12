package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Objects;

/**
 * Nd4j provides serialization of INDArrays many formats. This example gives some examples for binary and text
 * serialization.
 */
public class Nd4jEx16_Serialization {

    public static void main(String[] args) throws IOException {
        ClassLoader loader = Nd4jEx16_Serialization.class.getClassLoader();  // used to read files from resources.

        // 1. binary format from stream.
        INDArray arrWrite = Nd4j.linspace(1,25,25).reshape(5,5);
        String pathname = "tmp.bin";
        DataOutputStream sWrite = new DataOutputStream(new FileOutputStream(new File(pathname )));
        Nd4j.write(arrWrite, sWrite);

        DataInputStream sRead = new DataInputStream(new FileInputStream(new File(pathname )));
        INDArray arrRead = Nd4j.read(sRead);
        // We now have our test matrix in arrRead
        System.out.println("Read from binary format:" );
        System.out.println(arrRead );


        // 2. Read the numpy npy (and npz) formats:
        File file =  new File( Objects.requireNonNull(loader.getResource("twentyfive.npy")).getFile());

        INDArray x = Nd4j.createFromNpyFile(file); // Nd4j.createFromNpzFile for npz Numpy files.
        System.out.println();
        System.out.println("Read from Numpy .npyformat:" );
        System.out.println(x);


        // 3. binary format from file.
        file =  new File(pathname);
        Nd4j.saveBinary(arrWrite, file);
        arrRead = Nd4j.readBinary(file );
        System.out.println();
        System.out.println("Read from Numpy .npyformat:" );
        System.out.println(arrRead);


        // 4. read a csv file.
        file =  new File( Objects.requireNonNull(loader.getResource("twentyfive.csv")).getFile());
        String Filename = file.getAbsolutePath();
        arrRead  = Nd4j.readNumpy(Filename, ",");
        System.out.println();
        System.out.println("Read from csv format:" );
        System.out.println(arrRead);
    }

}

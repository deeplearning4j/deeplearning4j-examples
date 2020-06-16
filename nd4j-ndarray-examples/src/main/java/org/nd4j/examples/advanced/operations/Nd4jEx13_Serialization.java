package org.nd4j.examples.advanced.operations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Map;
import java.util.Objects;

/**
 * Nd4j provides serialization of INDArrays many formats. This example gives some examples for binary and text
 * serialization.
 */
public class Nd4jEx13_Serialization {

    public static void main(String[] args) throws Exception {
        ClassLoader loader = Nd4jEx13_Serialization.class.getClassLoader();  // used to read files from resources.

        // 1. binary format from stream.
        INDArray arrWrite = Nd4j.linspace(1,25,25).reshape(5,5);
        String pathname = "tmp.bin";

        try(DataOutputStream sWrite = new DataOutputStream(new FileOutputStream(new File(pathname )))){
            Nd4j.write(arrWrite, sWrite);
        }

        INDArray arrRead;
        try(DataInputStream sRead = new DataInputStream(new FileInputStream(new File(pathname )))){
            arrRead = Nd4j.read(sRead);
        }

        // We now have our test matrix in arrRead
        System.out.println("Read from binary stream:" );
        System.out.println(arrRead );


        // 2. Write and read the numpy npy format:
        File file =  new File("nd4j.npy" );
        Nd4j.writeAsNumpy(arrRead, file ); // Try to read this file from Python:   y = np.load('nd4j.npy')

        arrRead = Nd4j.createFromNpyFile(file); // We can read these files from nd4j.
        System.out.println();
        System.out.println("Read from Numpy .npy format:" );
        System.out.println(arrRead);


        // 3. Read the numpy npz format:
        file =  new File( Objects.requireNonNull(loader.getResource("numpyz.npz")).getFile());

        Map<String, INDArray> arrayMap = Nd4j.createFromNpzFile(file); //We get a map reading an .npz file.
        System.out.println();
        System.out.println("Read from Numpy .npz format:" );
        System.out.println(arrayMap.get("arr_0")); //We know there are 2 arrays in the .npz file.
        System.out.println(arrayMap.get("arr_1"));


        // 4. binary format from file.
        file =  new File(pathname);
        Nd4j.saveBinary(arrWrite, file);
        arrRead = Nd4j.readBinary(file );
        System.out.println();
        System.out.println("Read from binary format:" );
        System.out.println(arrRead);


        // 5. read a csv file.
        file =  new File( Objects.requireNonNull(loader.getResource("twentyfive.csv")).getFile());
        String Filename = file.getAbsolutePath();
        arrRead  = Nd4j.readNumpy(Filename, ",");
        System.out.println();
        System.out.println("Read from csv format:" );
        System.out.println(arrRead);

    }

}

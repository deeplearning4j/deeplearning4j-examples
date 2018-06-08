package org.datavec.inputsplit;

import org.datavec.api.split.NumberedFileInputSplit;

import java.net.URI;

/**
 * {@link org.datavec.api.split.NumberedFileInputSplit} is useful when you have a sequence of numbered files, following
 * a common pattern. For example, if you have a directory containing files in the following form:
 * file1.txt
 * file2.txt
 * file3.txt
 * file4.txt
 * ...
 * file100.txt
 * Then {@link org.datavec.api.split.NumberedFileInputSplit} can be used in the following way:
 * {@code NumberedFileInputSplit numberedFileInputSplit = new NumberedFileInputSplit("file%d.txt",1,100);}.
 *
 * As you can see, both the indexes, starting/ending, in the constructor, are inclusive.
 */
public class Ex03_NumberedFileInputSplitExample {
    public static void main(String[] args) {

        /*
          A basic usage of the NumberedFileInputSplit is shown below:

          Specify a partial non-absolute path (such as "file%d.txt" will prepend your path with the absolute path
          (with a file:// schema) of your root working directory. Such as:
          "file:///E:/Projects/Java/dl4j-examples/datavec-examples/file5.txt"
         */
        NumberedFileInputSplit split1 = new NumberedFileInputSplit("file%d.txt",
            1,
            5);

        System.out.println("--------------- Example 1: Loading simple numbered files ---------------");
        URI[] split1Uris = split1.locations();
        for (URI uri: split1Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        /*
          Sometimes the files are numbered with leading zeros. For example,
          file001.txt
          file002.txt
          ...
          file425.txt

          This is usually done for easier readability of file names and for having them all of the same length.
          Since NumberedFileInputSplit uses String.format() internally we can follow a leading zeros string pattern such
           as:
          "prefix-%03d.suffix" where %07d will format the numbered files upto 3 leading zeros.
         */
        NumberedFileInputSplit split2 = new NumberedFileInputSplit("/path/to/files/prefix-%03d.suffix",
            1,
            15);

        System.out.println("--------------- Example 2: Loading files with leading zeros ---------------");
        URI[] split2Uris = split2.locations();
        for (URI uri: split2Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        /*
          For more examples on how to format with String.format(), you can visit this link:
          https://docs.oracle.com/javase/7/docs/api/java/util/Formatter.html#syntax
          Make sure you only use the formats with the '%d' type

          Also, you can see some more examples for the unit tests here:
          https://github.com/deeplearning4j/deeplearning4j/blob/79d4110eee96f6b7f331931f4775233c4264d999/datavec/datavec-api/src/test/java/org/datavec/api/split/NumberedFileInputSplitTests.java
         */
    }
}

package org.datavec.inputsplit;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.TransformSplit;

import java.net.URI;
import java.util.List;

import static java.util.Arrays.asList;

/**
 * {@link org.datavec.api.split.TransformSplit} is an InputSplit implementation that maps the URIs of a given
 * {@link org.datavec.api.split.BaseInputSplit} to new URIs. It's seful when features and labels are in different
 * files sharing a common naming scheme, and the name of the output file can be determined given the name of the
 * input file.
 *
 * It takes a {@link org.datavec.api.split.BaseInputSplit} or its implementation and transforms it based on the
 * given {@link org.datavec.api.split.TransformSplit.URITransform}
 */

public class _3_TransformSplitExample {
    public static void main(String[] args) throws Exception {
        List<URI> inputFiles1 = asList(new URI("file:///foo/bar/../0.csv"), new URI("file:///foo/1.csv"));

        /*
          For the above files list, we can normalize the URI through URI#normalize by implementing the URITransform
          interface as follows:
         */
        TransformSplit.URITransform normalizeUriTransform = uri -> uri.normalize();

        TransformSplit transformSplit1 = new TransformSplit(new CollectionInputSplit(inputFiles1), normalizeUriTransform);

        /*
          The following URIs will have their ".." substrings striped off of them.
         */
        System.out.println("--------------- Example 1: Normalizing URIs ---------------");
        URI[] transformSplit1Uris = transformSplit1.locations();
        for (int i = 0; i < transformSplit1Uris.length; i++) {
            System.out.println(String.format("%s -> %s", inputFiles1.get(i), transformSplit1Uris[i]));
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        /*----------------------------------------------*/

        List<URI> inputFiles2 = asList(new URI("file:///foo/1-in.csv"), new URI("file:///foo/2-in.csv"));

        /*
          TransformSplit also provides a default implementation of URITransform which replaces a search string with a
          replace string through TransformSplit#ofSearchReplace. It can be used as follows:
         */
        InputSplit transformSplit2 = TransformSplit.ofSearchReplace(new CollectionInputSplit(inputFiles2),
            "-in.csv",
            "-out.csv");

        /*
          The following URIs will have their "-in.csv" substrings replaced with "-out.csv"
         */
        System.out.println("--------------- Example 2: Replacing substrings in URIs ---------------");
        URI[] transformSplit2Uris = transformSplit2.locations();
        for (int i = 0; i < transformSplit2Uris.length; i++) {
            System.out.println(String.format("%s -> %s", inputFiles2.get(i), transformSplit2Uris[i]));
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }
}

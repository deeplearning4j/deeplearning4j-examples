package org.nd4j.examples;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * --- Nd4j Example 14: Normalizers ---
 *
 * In this example, we demonstrate how one can create and fit a new normalizer, and save and restore them.
 * The example uses the NormalizerStandardize, but the same approach works with any {@link Normalizer} implementation.
 *
 * @author Ede Meijer
 */
public class Nd4jEx14_Normalizers {
    public static void main(String[] args) throws Exception {
        // A new normalizer can just be instantiated without any arguments, as we will fit it separately
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);

        // Now we create a random DataSet - normally you would have your real data
        DataSet data = new DataSet(Nd4j.rand(10, 3), Nd4j.rand(10, 1));

        // Fit the normalizer to the data - in this case it will calculate the means and standard deviations
        normalizer.fit(data);

        // Output the feature means and standard deviations so we can compare them after restoring the normalizer
        System.out.println("Means original: " + normalizer.getMean());
        System.out.println("Stds original:  " + normalizer.getStd());

        // Now we want to save the normalizer to a binary file. For doing this, one can use the NormalizerSerializer.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();

        // Prepare a temporary file to save to and load from
        File tmpFile = File.createTempFile("nd4j-example", "normalizers");
        tmpFile.deleteOnExit();

        // Save the normalizer to a temporary file
        serializer.write(normalizer, tmpFile);

        // Now restore the normalizer from the temporary file.
        NormalizerStandardize restoredNormalizer = serializer.restore(tmpFile);

        // Output the feature means and standard deviations so we can verify it was restored correctly
        System.out.println("Means restored: " + restoredNormalizer.getMean());
        System.out.println("Stds restored:  " + restoredNormalizer.getStd());
    }
}

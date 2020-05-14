/* *****************************************************************************
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

package org.nd4j.examples.advanced.operations;

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

/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.examples.sdx;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

/**
 * Generates the shared {@code models/mlp.sdz} fixture used by the non-JVM SDX
 * runtime examples (Python, Rust, C#, Kotlin, Swift), which cannot build a
 * SameDiff graph themselves.
 *
 * <p>The model is the same deterministic MLP as
 * {@link SdxRuntimeEndToEndExample}: {@code probs = softmax(relu(x·W1+b1)·W2+b2)}
 * with linspace-initialized weights, so its outputs are stable across
 * regenerations. The canonical verification vector printed by this tool
 * (input {@code x = linspace(0.1, 0.8, 8).reshape(2,4)} and the expected
 * {@code probs}) is baked into each language example for output checking.</p>
 *
 * <p>Run with:
 * {@code mvn -q compile exec:java -Dexec.mainClass=org.nd4j.examples.sdx.GenerateExampleModel}</p>
 */
public class GenerateExampleModel {

    public static void main(String[] args) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.var("w1", Nd4j.linspace(-1.0, 1.0, 32, DataType.FLOAT).reshape('c', 4, 8));
        SDVariable b1 = sd.var("b1", Nd4j.linspace(0.0, 0.7, 8, DataType.FLOAT));
        SDVariable w2 = sd.var("w2", Nd4j.linspace(1.0, -1.0, 24, DataType.FLOAT).reshape('c', 8, 3));
        SDVariable b2 = sd.var("b2", Nd4j.linspace(-0.1, 0.1, 3, DataType.FLOAT));
        SDVariable hidden = sd.nn.relu(x.mmul(w1).add(b1), 0.0);
        sd.nn.softmax("probs", hidden.mmul(w2).add(b2), 1);

        File out = new File(args.length > 0 ? args[0] : "../models/mlp.sdz").getAbsoluteFile();
        out.getParentFile().mkdirs();
        SDZSerializer.save(sd, out, false, Collections.emptyMap());
        System.out.println("Wrote " + out + " (" + out.length() + " bytes)");

        INDArray canonicalX = Nd4j.linspace(0.1, 0.8, 8, DataType.FLOAT).reshape('c', 2, 4);
        Map<String, INDArray> result =
                sd.output(Collections.singletonMap("x", canonicalX), Collections.singletonList("probs"));
        System.out.println("Canonical input  x[2,4]      = "
                + Arrays.toString(canonicalX.dup('c').data().asFloat()));
        System.out.println("Expected output  probs[2,3]  = "
                + Arrays.toString(result.get("probs").dup('c').data().asFloat()));
    }
}

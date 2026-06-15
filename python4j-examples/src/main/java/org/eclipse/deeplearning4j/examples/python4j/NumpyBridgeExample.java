/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.eclipse.deeplearning4j.examples.python4j;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.python4j.PythonExecutioner;
import org.nd4j.python4j.PythonGIL;
import org.nd4j.python4j.PythonVariable;
import org.nd4j.python4j.PythonVariables;
import org.nd4j.python4j.numpy.NumpyArray;

/**
 * NumPy Bridge Example
 *
 * This example demonstrates zero-copy sharing of memory between Java INDArrays
 * and Python NumPy arrays using the Python4j NumpyArray bridge.
 *
 * Key concepts:
 *
 *   NumpyArray.INSTANCE
 *     A PythonType<INDArray> that serialises an INDArray as a NumPy array when
 *     passing to Python, and deserialises a NumPy array as an INDArray when
 *     reading back from Python.  Because both ND4J and NumPy can share the same
 *     off-heap memory buffer (via JavaCPP / DLPack / the NumPy C-API), no data
 *     copy is required for CPU arrays -- this is the "zero-copy" bridge.
 *
 *   Zero-copy semantics
 *     When an INDArray created with Nd4j.create() is passed into Python, NumPy
 *     sees the same memory.  Any in-place modification performed in Python (e.g.,
 *     arr *= 2) will be visible immediately on the Java side, and vice-versa.
 *     This makes the bridge very efficient for large tensors.
 *
 *   Data-type mapping
 *     ND4J DataType.FLOAT  <-->  numpy.float32
 *     ND4J DataType.DOUBLE <-->  numpy.float64
 *     ND4J DataType.INT    <-->  numpy.int32
 *     ND4J DataType.LONG   <-->  numpy.int64
 *
 * Thread safety: every Python4j call must be wrapped in try(PythonGIL gil = PythonGIL.lock()).
 */
public class NumpyBridgeExample {

    public static void main(String[] args) throws Exception {

        // -------------------------------------------------------------------------
        // 1. Pass an INDArray to Python and receive it as a NumPy array
        // -------------------------------------------------------------------------
        System.out.println("=== 1. Pass INDArray to Python as NumPy array ===");

        INDArray javaArray = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        System.out.println("Java INDArray (before Python): " + javaArray);

        try (PythonGIL gil = PythonGIL.lock()) {
            // Wrap the INDArray as a PythonVariable using NumpyArray.INSTANCE as the type.
            // Python4j will expose this to the interpreter as a numpy.ndarray.
            PythonVariable<INDArray> inputVar = new PythonVariable<>("arr", NumpyArray.INSTANCE, javaArray);

            PythonVariables inputs = new PythonVariables();
            inputs.add(inputVar);

            PythonVariables outputs = new PythonVariables();
            // Declare an output variable that will hold the NumPy result as an INDArray.
            outputs.add(new PythonVariable<>("arr_info", NumpyArray.INSTANCE));

            String code =
                "import numpy as np\n" +
                "print('NumPy array received in Python:', arr)\n" +
                "print('dtype:', arr.dtype, '  shape:', arr.shape)\n" +
                "arr_info = arr * 1  # make a copy so we can return it safely\n";

            PythonExecutioner.exec(code, inputs, outputs);
            INDArray result = outputs.<INDArray>get("arr_info").getValue();
            System.out.println("INDArray received back in Java : " + result);
        }

        // -------------------------------------------------------------------------
        // 2. Zero-copy: in-place Python modification visible in Java
        // -------------------------------------------------------------------------
        System.out.println("\n=== 2. Zero-copy sharing: in-place Python mutation visible in Java ===");

        INDArray sharedArray = Nd4j.create(new float[]{10.0f, 20.0f, 30.0f});
        System.out.println("Before Python in-place op: " + sharedArray);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("shared", NumpyArray.INSTANCE, sharedArray));

            // arr *= 2 modifies the underlying off-heap buffer IN PLACE.
            // Because Java's INDArray points to the same buffer, no copy happens.
            String code = "shared *= 2  # in-place scale";
            PythonExecutioner.exec(code, inputs, null);
        }

        // The Java INDArray now reflects the change made in Python -- no copy needed.
        System.out.println("After Python in-place '*= 2'  : " + sharedArray);

        // -------------------------------------------------------------------------
        // 3. Matrix operations: dot product
        // -------------------------------------------------------------------------
        System.out.println("\n=== 3. Matrix dot product via NumPy ===");

        // Create two 2x3 and 3x2 matrices
        INDArray matA = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        INDArray matB = Nd4j.create(new float[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});
        System.out.println("Matrix A (2x3):\n" + matA);
        System.out.println("Matrix B (3x2):\n" + matB);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("A", NumpyArray.INSTANCE, matA));
            inputs.add(new PythonVariable<>("B", NumpyArray.INSTANCE, matB));

            PythonVariables outputs = new PythonVariables();
            outputs.add(new PythonVariable<>("C", NumpyArray.INSTANCE));

            String code =
                "import numpy as np\n" +
                "C = np.dot(A, B)  # 2x3 @ 3x2 => 2x2\n";

            PythonExecutioner.exec(code, inputs, outputs);
            INDArray dotProduct = outputs.<INDArray>get("C").getValue();
            System.out.println("A @ B (dot product, shape 2x2):\n" + dotProduct);
        }

        // -------------------------------------------------------------------------
        // 4. Transpose
        // -------------------------------------------------------------------------
        System.out.println("\n=== 4. Transpose via NumPy ===");

        INDArray mat = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        System.out.println("Original (2x3):\n" + mat);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("mat", NumpyArray.INSTANCE, mat));

            PythonVariables outputs = new PythonVariables();
            outputs.add(new PythonVariable<>("mat_T", NumpyArray.INSTANCE));

            // np.ascontiguousarray is needed because a transposed NumPy view is not
            // contiguous in memory; we need a contiguous copy to hand back to Java.
            String code =
                "import numpy as np\n" +
                "mat_T = np.ascontiguousarray(mat.T)\n";

            PythonExecutioner.exec(code, inputs, outputs);
            INDArray transposed = outputs.<INDArray>get("mat_T").getValue();
            System.out.println("Transposed (3x2):\n" + transposed);
        }

        // -------------------------------------------------------------------------
        // 5. Reshape
        // -------------------------------------------------------------------------
        System.out.println("\n=== 5. Reshape via NumPy ===");

        INDArray flat = Nd4j.linspace(1, 12, 12, DataType.FLOAT);
        System.out.println("Flat array (12 elements): " + flat);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("flat", NumpyArray.INSTANCE, flat));

            PythonVariables outputs = new PythonVariables();
            outputs.add(new PythonVariable<>("reshaped", NumpyArray.INSTANCE));

            String code =
                "import numpy as np\n" +
                "reshaped = np.ascontiguousarray(flat.reshape(3, 4))\n";

            PythonExecutioner.exec(code, inputs, outputs);
            INDArray reshaped = outputs.<INDArray>get("reshaped").getValue();
            System.out.println("Reshaped to (3x4):\n" + reshaped);
        }

        // -------------------------------------------------------------------------
        // 6. Data type handling: float32, float64, int32
        // -------------------------------------------------------------------------
        System.out.println("\n=== 6. Data type handling ===");

        INDArray float32Array = Nd4j.create(new float[]{1.1f, 2.2f, 3.3f}).castTo(DataType.FLOAT);
        INDArray float64Array = Nd4j.create(new double[]{1.1, 2.2, 3.3}).castTo(DataType.DOUBLE);
        INDArray int32Array   = Nd4j.create(new int[]{1, 2, 3}, new long[]{3}).castTo(DataType.INT);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("f32", NumpyArray.INSTANCE, float32Array));
            inputs.add(new PythonVariable<>("f64", NumpyArray.INSTANCE, float64Array));
            inputs.add(new PythonVariable<>("i32", NumpyArray.INSTANCE, int32Array));

            String code =
                "import numpy as np\n" +
                "print(f'float32 array dtype: {f32.dtype}  values: {f32}')\n" +
                "print(f'float64 array dtype: {f64.dtype}  values: {f64}')\n" +
                "print(f'int32   array dtype: {i32.dtype}  values: {i32}')\n";

            PythonExecutioner.exec(code, inputs, null);
        }

        System.out.println("Java DataType for float32Array: " + float32Array.dataType());
        System.out.println("Java DataType for float64Array: " + float64Array.dataType());
        System.out.println("Java DataType for int32Array  : " + int32Array.dataType());

        // -------------------------------------------------------------------------
        // 7. Full round-trip: compute element-wise sigmoid in Python, return INDArray
        // -------------------------------------------------------------------------
        System.out.println("\n=== 7. Round-trip: element-wise sigmoid ===");

        INDArray logits = Nd4j.create(new float[]{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        System.out.println("Logits: " + logits);

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.add(new PythonVariable<>("logits", NumpyArray.INSTANCE, logits));

            PythonVariables outputs = new PythonVariables();
            outputs.add(new PythonVariable<>("sigmoid_out", NumpyArray.INSTANCE));

            String code =
                "import numpy as np\n" +
                "sigmoid_out = np.ascontiguousarray(1.0 / (1.0 + np.exp(-logits)))\n";

            PythonExecutioner.exec(code, inputs, outputs);
            INDArray sigmoid = outputs.<INDArray>get("sigmoid_out").getValue();
            System.out.println("Sigmoid: " + sigmoid);
        }

        System.out.println("\nNumpyBridgeExample complete.");
    }
}

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
package org.nd4j.examples.sdx.client;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * A host-resident float32 tensor, analogous to {@code OnnxTensor} in the
 * ONNX Runtime Java API.
 *
 * <p>Create via the static factory methods:
 * <pre>{@code
 * SdxTensor t = SdxTensor.fromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new long[]{1, 4});
 * SdxTensor t = SdxTensor.fromBuffer(floatBuffer, new long[]{2, 4});
 * }</pre>
 *
 * <p>The tensor owns a direct {@link FloatBuffer} backed by a native-endian
 * {@link ByteBuffer}; the same buffer is passed to JNA without a copy when
 * the session calls {@link #buffer()} internally.
 *
 * <p>Instances are reusable: call {@link #update(float[])} to replace the data
 * in-place for subsequent runs without allocating a new tensor.
 */
public final class SdxTensor {

    /** sd::DataType::FLOAT32 */
    static final int SDX_DTYPE_FLOAT = 5;

    private final FloatBuffer buffer;
    private final long[] shape;
    private final int numElements;

    private SdxTensor(FloatBuffer buffer, long[] shape) {
        this.buffer = buffer;
        this.shape = shape.clone();
        int n = 1;
        for (long d : shape) {
            n = Math.toIntExact(Math.multiplyExact(n, d));
        }
        this.numElements = n;
    }

    /**
     * Creates a tensor by copying {@code data} into a new direct buffer.
     * Equivalent to ONNX Runtime's {@code OnnxTensor.createTensor(env, data, shape)}.
     *
     * @param data  float values in row-major (C) order
     * @param shape dimension sizes; product must equal {@code data.length}
     * @return a new tensor backed by a direct native buffer
     */
    public static SdxTensor fromArray(float[] data, long[] shape) {
        int n = 1;
        for (long d : shape) {
            n = Math.toIntExact(Math.multiplyExact(n, d));
        }
        if (n != data.length) {
            throw new IllegalArgumentException(
                    "Shape product " + n + " != data.length " + data.length);
        }
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(data).rewind();
        return new SdxTensor(fb, shape);
    }

    /**
     * Creates a tensor that wraps an existing direct {@link FloatBuffer}.
     * The buffer must be direct and native-byte-ordered. No copy is made.
     *
     * @param buffer direct FloatBuffer containing values in row-major order
     * @param shape  dimension sizes; product must equal {@code buffer.remaining()}
     * @return a new tensor wrapping the supplied buffer
     */
    public static SdxTensor fromBuffer(FloatBuffer buffer, long[] shape) {
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException("FloatBuffer must be direct");
        }
        return new SdxTensor(buffer.slice(), shape);
    }

    /**
     * Replaces the tensor data in-place.  The array length must match the
     * original shape product.  This avoids allocating a new tensor for
     * repeated inference calls with fresh input values.
     *
     * @param data new float values in row-major order
     */
    public void update(float[] data) {
        if (data.length != numElements) {
            throw new IllegalArgumentException(
                    "data.length " + data.length + " != tensor elements " + numElements);
        }
        buffer.clear();
        buffer.put(data);
        buffer.rewind();
    }

    /** Returns a copy of the tensor shape. */
    public long[] shape() {
        return shape.clone();
    }

    /** Returns the rank (number of dimensions). */
    public int rank() {
        return shape.length;
    }

    /** Returns the total number of float elements. */
    public int numElements() {
        return numElements;
    }

    /**
     * Reads the tensor contents into a new float array.
     * Equivalent to {@code OnnxTensor.getFloatBuffer().array()} after ensuring
     * backing array availability — works for direct buffers too.
     *
     * @return a fresh float array copy of the tensor data
     */
    public float[] toFloatArray() {
        float[] out = new float[numElements];
        FloatBuffer view = buffer.duplicate();
        view.rewind();
        view.get(out);
        return out;
    }

    /** Internal: the underlying direct buffer, rewound. */
    FloatBuffer buffer() {
        FloatBuffer view = buffer.duplicate();
        view.rewind();
        return view;
    }

    /** Internal: byte count for the tensor data. */
    long byteCount() {
        return (long) numElements * Float.BYTES;
    }
}

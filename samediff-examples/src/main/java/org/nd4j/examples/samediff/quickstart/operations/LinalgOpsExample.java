/*
 *
 * This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package org.nd4j.examples.samediff.quickstart.operations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

/**
 * SameDiff Linear Algebra Operations (sd.linalg()) - Complete API Example
 *
 * The SDLinalg namespace provides differentiable linear algebra operations
 * for matrix decompositions, solvers, and transformations.
 *
 * Operations covered:
 *   Decompositions:
 *     - cholesky    - Cholesky decomposition (A = LL^T)
 *     - lu          - LU decomposition
 *     - svd         - Singular Value Decomposition
 *     - qr          - QR decomposition
 *     - eig         - Eigendecomposition
 *
 *   Solvers:
 *     - solve             - Solve Ax = b
 *     - triangularSolve   - Solve triangular system
 *     - lstsq             - Least squares solution
 *
 *   Matrix Operations:
 *     - matmul / mmul             - Matrix multiplication
 *     - matrixDeterminant         - Determinant
 *     - matrixInverse             - Matrix inverse
 *     - logdet                    - Log-determinant
 *     - cross                     - Cross product
 *     - einsum                    - Einstein summation
 *
 *   Matrix Construction:
 *     - diag / diag_part          - Diagonal operations
 *     - tri / triu                - Triangular matrices
 *     - matrixBandPart            - Band matrix extraction
 */
public class LinalgOpsExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. CHOLESKY DECOMPOSITION - A = LL^T
        // ============================================================
        System.out.println("=== Cholesky Decomposition ===");
        {
            SameDiff sd = SameDiff.create();

            // Create a symmetric positive-definite matrix: A = X^T * X + I
            INDArray x = Nd4j.randn(DataType.DOUBLE, 4, 4);
            INDArray spd = x.transpose().mmul(x).add(Nd4j.eye(4).castTo(DataType.DOUBLE));

            SDVariable a = sd.constant("A", spd);
            SDVariable l = sd.linalg().cholesky("L", a);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "L");
            INDArray lResult = result.get("L");
            System.out.println("  A (4x4 SPD matrix):");
            System.out.println("  L (lower triangular):");
            System.out.println("  " + lResult);
            // Verify: L * L^T should equal A
            INDArray reconstructed = lResult.mmul(lResult.transpose());
            System.out.println("  Reconstruction error: " + spd.sub(reconstructed).norm2Number());
        }

        // ============================================================
        // 2. LU DECOMPOSITION
        // ============================================================
        System.out.println("\n=== LU Decomposition ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray matrix = Nd4j.createFromArray(new double[][]{
                    {2, 1, 1},
                    {4, 3, 3},
                    {8, 7, 9}
            });

            SDVariable a = sd.constant("A", matrix);
            SDVariable lu = sd.linalg().lu("LU", a);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "LU");
            System.out.println("  Input matrix: " + matrix);
            System.out.println("  LU result: " + result.get("LU"));
        }

        // ============================================================
        // 3. SVD - Singular Value Decomposition
        // ============================================================
        System.out.println("\n=== Singular Value Decomposition ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray matrix = Nd4j.createFromArray(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
                    {10, 11, 12}
            });

            SDVariable a = sd.constant("A", matrix);

            // Full SVD: returns singular values
            // fullUV=true: compute full U and V matrices
            // computeUV=true: compute U and V (not just singular values)
            SDVariable svd = sd.linalg().svd("svd", a, true, true);

            // Compact SVD (without switchNum parameter)
            SDVariable svdCompact = sd.linalg().svd("svdCompact", a, false, true);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "svd");
            System.out.println("  Input: 4x3 matrix");
            System.out.println("  Singular values: " + result.get("svd"));
        }

        // ============================================================
        // 4. QR DECOMPOSITION
        // ============================================================
        System.out.println("\n=== QR Decomposition ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray matrix = Nd4j.createFromArray(new double[][]{
                    {12, -51, 4},
                    {6, 167, -68},
                    {-4, 24, -41}
            });

            SDVariable a = sd.constant("A", matrix);

            // QR decomposition returns [Q, R]
            SDVariable[] qr = sd.linalg().qr(new String[]{"Q", "R"}, a, true);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "Q", "R");
            System.out.println("  Q (orthogonal): " + result.get("Q").shapeInfoToString());
            System.out.println("  R (upper triangular): " + result.get("R").shapeInfoToString());
        }

        // ============================================================
        // 5. EIGENDECOMPOSITION
        // ============================================================
        System.out.println("\n=== Eigendecomposition ===");
        {
            SameDiff sd = SameDiff.create();
            // Symmetric matrix for real eigenvalues
            INDArray matrix = Nd4j.createFromArray(new double[][]{
                    {2, 1},
                    {1, 3}
            });

            SDVariable a = sd.constant("A", matrix);
            SDVariable[] eigResult = sd.linalg().eig(new String[]{"eigenvalues", "eigenvectors"}, a);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "eigenvalues", "eigenvectors");
            System.out.println("  Matrix: [[2,1],[1,3]]");
            System.out.println("  Eigenvalues: " + result.get("eigenvalues"));
            System.out.println("  Eigenvectors: " + result.get("eigenvectors"));
        }

        // ============================================================
        // 6. SOLVE - Linear system Ax = b
        // ============================================================
        System.out.println("\n=== Solve Linear System ===");
        {
            SameDiff sd = SameDiff.create();

            // Solve: 2x + y = 5, x + 3y = 7
            INDArray a = Nd4j.createFromArray(new double[][]{
                    {2, 1},
                    {1, 3}
            });
            INDArray b = Nd4j.createFromArray(new double[][]{{5}, {7}});

            SDVariable matA = sd.constant("A", a);
            SDVariable vecB = sd.constant("b", b);

            // solve(matrix, rhs, adjoint)
            SDVariable x = sd.linalg().solve("x", matA, vecB, false);

            // Without adjoint parameter
            SDVariable x2 = sd.linalg().solve("x2", matA, vecB);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "x");
            System.out.println("  2x + y = 5");
            System.out.println("  x + 3y = 7");
            System.out.println("  Solution [x, y]: " + result.get("x").transpose());
        }

        // ============================================================
        // 7. TRIANGULAR SOLVE
        // ============================================================
        System.out.println("\n=== Triangular Solve ===");
        {
            SameDiff sd = SameDiff.create();

            // Lower triangular matrix
            INDArray lower = Nd4j.createFromArray(new double[][]{
                    {3, 0, 0},
                    {1, 2, 0},
                    {4, 1, 5}
            });
            INDArray rhs = Nd4j.createFromArray(new double[][]{{9}, {8}, {25}});

            SDVariable l = sd.constant("L", lower);
            SDVariable b = sd.constant("b", rhs);

            // triangularSolve(matrix, rhs, lower, adjoint)
            SDVariable x = sd.linalg().triangularSolve("x", l, b, true, false);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "x");
            System.out.println("  Lower triangular solve Lx = b");
            System.out.println("  Solution: " + result.get("x").transpose());
        }

        // ============================================================
        // 8. LEAST SQUARES (lstsq) - Overdetermined systems
        // ============================================================
        System.out.println("\n=== Least Squares ===");
        {
            SameDiff sd = SameDiff.create();

            // Overdetermined system: 3 equations, 2 unknowns
            INDArray a = Nd4j.createFromArray(new double[][]{
                    {1, 1},
                    {1, 2},
                    {1, 3}
            });
            INDArray b = Nd4j.createFromArray(new double[][]{{1}, {2}, {2}});

            SDVariable matA = sd.constant("A", a);
            SDVariable vecB = sd.constant("b", b);

            // lstsq(matrix, rhs, l2_regularizer, fast)
            SDVariable x = sd.linalg().lstsq("x", matA, vecB, 0.0, true);

            // Without fast parameter
            SDVariable x2 = sd.linalg().lstsq("x2", matA, vecB, 1e-6);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "x");
            System.out.println("  Overdetermined system (3 eqs, 2 unknowns)");
            System.out.println("  Least squares solution: " + result.get("x").transpose());
        }

        // ============================================================
        // 9. MATRIX MULTIPLY - matmul and mmul
        // ============================================================
        System.out.println("\n=== Matrix Multiplication ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable a = sd.placeHolder("A", DataType.DOUBLE, -1, 3);
            SDVariable b = sd.placeHolder("B", DataType.DOUBLE, 3, -1);

            // matmul: general with alpha/beta scaling and transpose flags
            //   result = alpha * op(A) * op(B) + beta * C
            SDVariable matmulResult = sd.linalg().matmul("matmul", a, b,
                    1.0, 0.0, false, false);

            // Simple matmul (no scaling, no transpose)
            SDVariable simpleResult = sd.linalg().matmul("simple", a, b);

            // mmul: standard matrix multiply with transpose flags
            SDVariable mmulResult = sd.linalg().mmul("mmul", a, b, false, false, false);

            // Simple mmul
            SDVariable mmulSimple = sd.linalg().mmul("mmulSimple", a, b);

            INDArray aData = Nd4j.randn(DataType.DOUBLE, 2, 3);
            INDArray bData = Nd4j.randn(DataType.DOUBLE, 3, 4);
            java.util.HashMap<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("A", aData);
            ph.put("B", bData);

            Map<String, INDArray> result = sd.output(ph, "matmul", "mmulSimple");
            System.out.println("  [2x3] * [3x4] = " + result.get("matmul").shapeInfoToString());
        }

        // ============================================================
        // 10. DETERMINANT, INVERSE, LOG-DETERMINANT
        // ============================================================
        System.out.println("\n=== Determinant, Inverse, Log-Determinant ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray matrix = Nd4j.createFromArray(new double[][]{
                    {1, 2},
                    {3, 4}
            });

            SDVariable a = sd.constant("A", matrix);

            SDVariable det = sd.linalg().matrixDeterminant("det", a);
            SDVariable inv = sd.linalg().matrixInverse("inv", a);
            SDVariable logDet = sd.linalg().logdet("logdet", a);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "det", "inv", "logdet");
            System.out.println("  Matrix: [[1,2],[3,4]]");
            System.out.println("  Determinant: " + result.get("det"));
            System.out.println("  Inverse:\n  " + result.get("inv"));
            System.out.println("  Log-determinant: " + result.get("logdet"));
        }

        // ============================================================
        // 11. CROSS PRODUCT
        // ============================================================
        System.out.println("\n=== Cross Product ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray v1 = Nd4j.createFromArray(new double[]{1, 0, 0});
            INDArray v2 = Nd4j.createFromArray(new double[]{0, 1, 0});

            SDVariable a = sd.constant("a", v1);
            SDVariable b = sd.constant("b", v2);
            SDVariable cross = sd.linalg().cross("cross", a, b);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "cross");
            System.out.println("  [1,0,0] x [0,1,0] = " + result.get("cross"));
            System.out.println("  (should be [0,0,1])");
        }

        // ============================================================
        // 12. EINSUM - Einstein Summation
        // ============================================================
        System.out.println("\n=== Einstein Summation ===");
        {
            SameDiff sd = SameDiff.create();

            // Batch matrix multiply: "bij,bjk->bik"
            SDVariable a = sd.placeHolder("A", DataType.DOUBLE, -1, 3, 4);
            SDVariable b = sd.placeHolder("B", DataType.DOUBLE, -1, 4, 5);
            SDVariable batchMM = sd.linalg().einsum("batchMM",
                    new SDVariable[]{a, b}, "bij,bjk->bik");

            // Trace: "ii->"
            SDVariable m = sd.placeHolder("M", DataType.DOUBLE, 3, 3);
            SDVariable trace = sd.linalg().einsum("trace",
                    new SDVariable[]{m}, "ii->");

            // Outer product: "i,j->ij"
            SDVariable x = sd.placeHolder("x", DataType.DOUBLE, 3);
            SDVariable y = sd.placeHolder("y", DataType.DOUBLE, 4);
            SDVariable outer = sd.linalg().einsum("outer",
                    new SDVariable[]{x, y}, "i,j->ij");

            java.util.HashMap<String, INDArray> ph = new java.util.HashMap<>();
            ph.put("A", Nd4j.randn(DataType.DOUBLE, 2, 3, 4));
            ph.put("B", Nd4j.randn(DataType.DOUBLE, 2, 4, 5));
            ph.put("M", Nd4j.eye(3).castTo(DataType.DOUBLE));
            ph.put("x", Nd4j.createFromArray(1.0, 2.0, 3.0));
            ph.put("y", Nd4j.createFromArray(4.0, 5.0, 6.0, 7.0));

            Map<String, INDArray> result = sd.output(ph,
                    "batchMM", "trace", "outer");
            System.out.println("  Batch matmul bij,bjk->bik: " + result.get("batchMM").shapeInfoToString());
            System.out.println("  Trace ii->: " + result.get("trace"));
            System.out.println("  Outer product i,j->ij:\n  " + result.get("outer"));
        }

        // ============================================================
        // 13. DIAGONAL OPERATIONS
        // ============================================================
        System.out.println("\n=== Diagonal Operations ===");
        {
            SameDiff sd = SameDiff.create();

            // diag: vector -> diagonal matrix
            INDArray vec = Nd4j.createFromArray(1.0, 2.0, 3.0);
            SDVariable v = sd.constant("v", vec);
            SDVariable diagMat = sd.linalg().diag("diagMat", v);

            // diag_part: matrix -> diagonal vector
            INDArray mat = Nd4j.createFromArray(new double[][]{
                    {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
            });
            SDVariable m = sd.constant("M", mat);
            SDVariable diagVec = sd.linalg().diag_part("diagVec", m);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "diagMat", "diagVec");
            System.out.println("  diag([1,2,3]):\n  " + result.get("diagMat"));
            System.out.println("  diag_part([[1,2,3],[4,5,6],[7,8,9]]): " + result.get("diagVec"));
        }

        // ============================================================
        // 14. TRIANGULAR MATRIX CONSTRUCTION
        // ============================================================
        System.out.println("\n=== Triangular Matrices ===");
        {
            SameDiff sd = SameDiff.create();

            // tri: create lower triangular matrix of ones
            // tri(dataType, rows, cols, diagonal)
            SDVariable triMat = sd.linalg().tri("triMat", DataType.DOUBLE, 4, 4, 0);

            // tri with offset diagonal
            SDVariable triOffset = sd.linalg().tri("triOffset", DataType.DOUBLE, 4, 4, 1);

            // Simple tri (no dataType or diagonal)
            SDVariable triSimple = sd.linalg().tri("triSimple", 3, 3);

            // triu: extract upper triangular part
            INDArray fullMat = Nd4j.ones(DataType.DOUBLE, 4, 4);
            SDVariable full = sd.constant("full", fullMat);
            SDVariable upper = sd.linalg().triu("upper", full, 0);
            SDVariable upperOffset = sd.linalg().triu("upperOffset", full, 1);

            // triu with default diagonal
            SDVariable upperDefault = sd.linalg().triu("upperDefault", full);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(),
                    "triMat", "triOffset", "upper", "upperOffset");
            System.out.println("  tri(4,4,0) - lower triangular:\n  " + result.get("triMat"));
            System.out.println("  tri(4,4,1) - with super-diagonal:\n  " + result.get("triOffset"));
            System.out.println("  triu(ones, 0) - upper triangular:\n  " + result.get("upper"));
            System.out.println("  triu(ones, 1) - strict upper:\n  " + result.get("upperOffset"));
        }

        // ============================================================
        // 15. MATRIX BAND PART
        // ============================================================
        System.out.println("\n=== Matrix Band Part ===");
        {
            SameDiff sd = SameDiff.create();
            INDArray fullMat = Nd4j.ones(DataType.DOUBLE, 5, 5);
            SDVariable m = sd.constant("M", fullMat);

            // Extract band: keep minLower sub-diagonals and maxUpper super-diagonals
            // (1, 1) = tridiagonal
            SDVariable[] band = sd.linalg().matrixBandPart(new String[]{"band"}, m, 1, 1);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "band");
            System.out.println("  Tridiagonal band (lower=1, upper=1):");
            System.out.println("  " + result.get("band"));
        }

        System.out.println("\nAll linear algebra operations demonstrated successfully.");
    }
}

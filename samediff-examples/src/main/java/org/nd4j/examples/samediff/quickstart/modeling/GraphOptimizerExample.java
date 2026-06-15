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

package org.nd4j.examples.samediff.quickstart.modeling;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SameDiff Graph Optimizer - Complete API Reference
 *
 * The GraphOptimizer applies a pipeline of optimization passes to simplify,
 * fuse, and accelerate SameDiff computation graphs before execution.
 *
 * Optimization passes (in order):
 *
 *   1. Dead Code Elimination - Remove unused ops
 *   2. Constant Folding - Pre-compute constant expressions
 *   3. Broadcast Elimination - Remove redundant broadcasts, double negation
 *   4. Reordering - Reassociate constants, eliminate double transpose
 *   5. Algebraic Simplification - x+0->x, x*1->x, x*0->0
 *   6. Peephole Optimizations - relu(relu(x))->relu(x), exp(log(x))->x
 *   7. Arithmetic Chain Folding - add(add(x,c1),c2)->add(x,c1+c2)
 *   8. Strength Reduction - pow(x,2)->square, div(x,c)->mul(x,1/c)
 *   9. Concat/Split Optimization - Flatten nested concat
 *  10. Common Subexpression Elimination (CSE)
 *  11. Attention Fusion - softmax(Q@K^T)@V -> fused attention
 *  12. Horizontal Fusion - Parallel matmuls -> single fused matmul
 *  13. Activation Fusion - sigmoid(x)*x -> Swish, SwiGLU detection
 *  14. Normalization Fusion - RMSNorm pattern detection
 *  15. Linear Fusion - matmul(x,w)+bias -> XwPlusB
 *
 * Key API:
 *   - GraphOptimizer.optimize(SameDiff, String... outputs)
 *   - GraphOptimizer.optimize(SameDiff, List<String> outputs, List<OptimizerSet> passes)
 *   - GraphOptimizer.defaultOptimizations()
 *   - GraphOptimizer.defaultCorrectnessOptimizations()
 */
public class GraphOptimizerExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. ALGEBRAIC SIMPLIFICATION
        // ============================================================
        System.out.println("=== Algebraic Simplification ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

            // Build a graph with algebraic redundancies
            SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1, 4));
            SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1, 4));

            // x + 0 should simplify to x
            SDVariable addZero = x.add(zero);
            // (x + 0) * 1 should simplify to x
            SDVariable mulOne = addZero.mul(one);
            // result * 1 + 0 -> x
            SDVariable out = mulOne.mul(one).add("out", zero);

            int originalOpCount = sd.getOps().size();

            // Optimize
            SameDiff optimized = GraphOptimizer.optimize(sd, "out");
            int optimizedOpCount = optimized.getOps().size();

            // Verify correctness
            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");

            System.out.println("  x + 0 * 1 + 0 -> x");
            System.out.println("  Original ops:  " + originalOpCount);
            System.out.println("  Optimized ops: " + optimizedOpCount);
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-5));
        }

        // ============================================================
        // 2. PEEPHOLE OPTIMIZATIONS
        // ============================================================
        System.out.println("\n=== Peephole Optimizations ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

            // Idempotent: relu(relu(x)) -> relu(x)
            SDVariable relu1 = sd.nn().relu(x, 0);
            SDVariable relu2 = sd.nn().relu("out_relu", relu1, 0);

            int originalOps = sd.getOps().size();
            SameDiff optimized = GraphOptimizer.optimize(sd, "out_relu");
            int optimizedOps = optimized.getOps().size();

            System.out.println("  relu(relu(x)) -> relu(x)");
            System.out.println("  Original ops:  " + originalOps);
            System.out.println("  Optimized ops: " + optimizedOps);

            // Inverse pairs: exp(log(x)) -> x (for x > 0)
            SameDiff sd2 = SameDiff.create();
            SDVariable x2 = sd2.placeHolder("x", DataType.FLOAT, -1, 4);
            SDVariable logX = sd2.math().log(x2);
            SDVariable expLogX = sd2.math().exp("out_explog", logX);

            SameDiff opt2 = GraphOptimizer.optimize(sd2, "out_explog");
            System.out.println("  exp(log(x)) -> x");
            System.out.println("  Original ops:  " + sd2.getOps().size());
            System.out.println("  Optimized ops: " + opt2.getOps().size());
        }

        // ============================================================
        // 3. STRENGTH REDUCTION
        // ============================================================
        System.out.println("\n=== Strength Reduction ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

            // pow(x, 2) -> square(x) - much faster
            SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0));
            SDVariable powX = sd.math().pow("out", x, two);

            SameDiff optimized = GraphOptimizer.optimize(sd, "out");

            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");

            System.out.println("  pow(x, 2) -> square(x)");
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-5));

            // div(x, c) -> mul(x, 1/c) - multiplication is faster than division
            SameDiff sd2 = SameDiff.create();
            SDVariable x2 = sd2.placeHolder("x", DataType.FLOAT, -1, 4);
            SDVariable c = sd2.constant("c", Nd4j.scalar(DataType.FLOAT, 3.0));
            SDVariable divC = x2.div("out", c);

            SameDiff opt2 = GraphOptimizer.optimize(sd2, "out");
            System.out.println("  div(x, 3) -> mul(x, 0.333...)");
        }

        // ============================================================
        // 4. COMMON SUBEXPRESSION ELIMINATION (CSE)
        // ============================================================
        System.out.println("\n=== Common Subexpression Elimination ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

            // tanh(x) computed twice -> should be deduplicated
            SDVariable tanh1 = sd.math().tanh(x);
            SDVariable tanh2 = sd.math().tanh(x);
            SDVariable out = tanh1.add("out", tanh2);

            int originalOps = sd.getOps().size();
            SameDiff optimized = GraphOptimizer.optimize(sd, "out");
            int optimizedOps = optimized.getOps().size();

            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");

            System.out.println("  tanh(x) + tanh(x) -> 2 * tanh(x) [CSE deduplicates tanh]");
            System.out.println("  Original ops:  " + originalOps);
            System.out.println("  Optimized ops: " + optimizedOps);
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-5));
        }

        // ============================================================
        // 5. ACTIVATION FUSION (Swish Detection)
        // ============================================================
        System.out.println("\n=== Activation Fusion (Swish) ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

            // sigmoid(x) * x is the Swish activation
            SDVariable sigmoid = sd.nn().sigmoid(x);
            SDVariable swish = sigmoid.mul("out", x);

            int originalOps = sd.getOps().size();
            SameDiff optimized = GraphOptimizer.optimize(sd, "out");
            int optimizedOps = optimized.getOps().size();

            // Check that Swish op was fused
            boolean hasSwish = false;
            for (SameDiffOp op : optimized.getOps().values()) {
                String opName = op.getOp().getClass().getSimpleName();
                if (opName.contains("Swish")) {
                    hasSwish = true;
                    break;
                }
            }

            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");

            System.out.println("  sigmoid(x) * x -> Swish(x)");
            System.out.println("  Original ops:  " + originalOps);
            System.out.println("  Optimized ops: " + optimizedOps);
            System.out.println("  Swish fused:   " + hasSwish);
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-5));
        }

        // ============================================================
        // 6. LINEAR FUSION (MatMul + Bias -> XwPlusB)
        // ============================================================
        System.out.println("\n=== Linear Fusion (MatMul + Bias) ===");
        {
            SameDiff sd = SameDiff.create();
            int nIn = 8, nOut = 4;

            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, nIn);
            SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, nIn, nOut).muli(0.1));
            SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, nOut));

            // Manual matmul + bias (common in hand-written graphs)
            SDVariable mm = x.mmul(w);
            SDVariable out = mm.add("out", b);

            int originalOps = sd.getOps().size();
            SameDiff optimized = GraphOptimizer.optimize(sd, "out");
            int optimizedOps = optimized.getOps().size();

            // Check for XwPlusB fusion
            boolean hasXwPlusB = false;
            for (SameDiffOp op : optimized.getOps().values()) {
                String opName = op.getOp().getClass().getSimpleName();
                if (opName.contains("XwPlusB")) {
                    hasXwPlusB = true;
                    break;
                }
            }

            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, nIn));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");

            System.out.println("  matmul(x, w) + b -> XwPlusB(x, w, b)");
            System.out.println("  Original ops:  " + originalOps);
            System.out.println("  Optimized ops: " + optimizedOps);
            System.out.println("  XwPlusB fused: " + hasXwPlusB);
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-5));
        }

        // ============================================================
        // 7. MULTI-LAYER MLP OPTIMIZATION
        // ============================================================
        System.out.println("\n=== Multi-Layer MLP Optimization ===");
        {
            SameDiff sd = SameDiff.create();
            int nIn = 16, nHidden = 32, nOut = 10;

            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, nIn);

            // Layer 1: matmul + bias + relu
            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, nIn, nHidden).muli(0.1));
            SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, nHidden));
            SDVariable z1 = x.mmul(w1).add(b1);
            SDVariable a1 = sd.nn().relu(z1, 0);

            // Layer 2: matmul + bias + sigmoid * x (Swish)
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, nHidden, nHidden).muli(0.1));
            SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, nHidden));
            SDVariable z2 = a1.mmul(w2).add(b2);
            SDVariable sig2 = sd.nn().sigmoid(z2);
            SDVariable a2 = sig2.mul(z2); // Swish

            // Output layer: matmul + bias + softmax
            SDVariable w3 = sd.var("w3", Nd4j.randn(DataType.FLOAT, nHidden, nOut).muli(0.1));
            SDVariable b3 = sd.var("b3", Nd4j.zeros(DataType.FLOAT, nOut));
            SDVariable z3 = a2.mmul(w3).add(b3);
            SDVariable out = sd.nn().softmax("out", z3);

            int originalOps = sd.getOps().size();

            // Optimize the entire graph at once
            SameDiff optimized = GraphOptimizer.optimize(sd, "out");
            int optimizedOps = optimized.getOps().size();

            // Enumerate remaining ops
            System.out.println("  3-layer MLP: linear->relu, linear->swish, linear->softmax");
            System.out.println("  Original ops:  " + originalOps);
            System.out.println("  Optimized ops: " + optimizedOps);
            System.out.println("  Optimization: " + String.format("%.0f%%", (1.0 - (double) optimizedOps / originalOps) * 100) + " reduction");

            System.out.println("  Remaining ops:");
            for (SameDiffOp op : optimized.getOps().values()) {
                DifferentialFunction fn = op.getOp();
                System.out.println("    " + fn.getClass().getSimpleName());
            }

            Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 4, nIn));
            INDArray expected = sd.outputSingle(ph, "out");
            INDArray actual = optimized.outputSingle(ph, "out");
            System.out.println("  Results match: " + expected.equalsWithEps(actual, 1e-4));
        }

        // ============================================================
        // 8. CUSTOM OPTIMIZATION PASSES
        // ============================================================
        System.out.println("\n=== Custom Optimization Passes ===");
        {
            // Get the full default optimization pipeline
            List<OptimizerSet> allPasses = GraphOptimizer.defaultOptimizations();
            System.out.println("  Total default passes: " + allPasses.size());

            // Correctness-safe passes (excludes precision-changing optimizations)
            List<OptimizerSet> safePasses = GraphOptimizer.defaultCorrectnessOptimizations();
            System.out.println("  Correctness-safe passes: " + safePasses.size());

            // You can also suppress individual passes via system property:
            // -Dnd4j.optimizer.skip=QuantizationOptimizations
            // -Dnd4j.optimizer.maxIterations=5

            System.out.println("\n  Pass categories:");
            System.out.println("    Algebraic: x+0, x*1, x*0 simplification");
            System.out.println("    Peephole: idempotent ops, inverse pairs");
            System.out.println("    Strength: pow->square, div->mul");
            System.out.println("    CSE: deduplicate identical subexpressions");
            System.out.println("    Fusion: Swish, XwPlusB, attention, RMSNorm");
            System.out.println("    DCE: remove ops not contributing to output");
            System.out.println("    Remat: rematerialize cheap ops to shorten live ranges");
        }

        // ============================================================
        // 9. INSPECTING OPTIMIZED GRAPHS
        // ============================================================
        System.out.println("\n=== Graph Inspection ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
            SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
            SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 4));
            SDVariable mm = x.mmul(w);
            SDVariable linear = mm.add(b);
            SDVariable sig = sd.nn().sigmoid(linear);
            SDVariable out = sig.mul("out", linear); // Swish pattern

            SameDiff optimized = GraphOptimizer.optimize(sd, "out");

            // Inspect ops
            Map<String, SameDiffOp> ops = optimized.getOps();
            System.out.println("  Optimized graph ops:");
            for (Map.Entry<String, SameDiffOp> entry : ops.entrySet()) {
                SameDiffOp op = entry.getValue();
                DifferentialFunction fn = op.getOp();
                System.out.println("    " + entry.getKey() + " -> " +
                        fn.getClass().getSimpleName() +
                        " (inputs: " + Arrays.toString(op.getInputsToOp().toArray()) +
                        ", outputs: " + Arrays.toString(op.getOutputsOfOp().toArray()) + ")");
            }

            // Inspect variables
            System.out.println("  Variables:");
            for (String varName : optimized.variableNames()) {
                System.out.println("    " + varName + " type=" +
                        optimized.getVariable(varName).getVariableType());
            }
        }

        System.out.println("\nAll graph optimization patterns demonstrated successfully.");
    }
}

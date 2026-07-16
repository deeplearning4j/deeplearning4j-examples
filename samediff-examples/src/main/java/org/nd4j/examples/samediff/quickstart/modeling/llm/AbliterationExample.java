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
package org.nd4j.examples.samediff.quickstart.modeling.llm;

import org.eclipse.deeplearning4j.llm.editing.AbliterationConfig;
import org.eclipse.deeplearning4j.llm.editing.AbliterationConfig.LayerSelection;
import org.eclipse.deeplearning4j.llm.editing.AbliterationConfig.Method;
import org.eclipse.deeplearning4j.llm.editing.AbliterationResult;
import org.eclipse.deeplearning4j.llm.editing.AbliterationWorkflow;
import org.eclipse.deeplearning4j.llm.editing.RefusalDirection;
import org.eclipse.deeplearning4j.llm.editing.RefusalDirectionFinder;
import org.eclipse.deeplearning4j.llm.editing.WeightOrthogonalizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Activation ablation (a.k.a. "abliteration") — a mechanistic-interpretability technique
 * for directed model editing, from the residual-stream editing APIs in samediff-llm.
 *
 * Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS
 * 2024), showed that a behaviour encoded in a model's residual stream can often be traced
 * to a SINGLE linear direction, and removed WITHOUT retraining by orthogonalizing the
 * weight matrices against that direction. The pipeline is pure linear algebra:
 *
 *   1. Collect residual-stream activations on two contrasting prompt sets (A vs B).
 *   2. Estimate the direction that separates them (diff-in-means / PCA / projected).
 *   3. Orthogonalize the target weights against it:  W' = W - alpha * (W @ d) d^T.
 *
 * Because it is training-free and interpretable, it is a standard tool for probing which
 * directions in activation space carry a given behaviour and for measuring how much of a
 * model's capability rides on a single direction. (Abliteration was popularized as a
 * safety-alignment-removal method; use the technique responsibly and only on models you
 * are authorized to modify.)
 *
 * This example is fully SELF-CONTAINED: it builds a tiny synthetic model and feeds
 * synthetic activation matrices whose two classes are deliberately separated along a known
 * planted direction, so every API surface runs in milliseconds with no model download and
 * the math can be VERIFIED (recovered direction ~= planted direction; edited weights become
 * orthogonal to it). The exact same calls apply to a real imported GGUF/ONNX decoder — see
 * the commented AbliterationWorkflow block at the end for the end-to-end shape.
 *
 * Run: mvn exec:java -Dexec.mainClass=org.nd4j.examples.samediff.quickstart.modeling.llm.AbliterationExample
 */
public class AbliterationExample {

    private static final int HIDDEN = 16;     // residual-stream / hidden dimension
    private static final int SAMPLES = 128;    // activations collected per class
    private static final int LAYERS = 3;       // synthetic transformer layers

    public static void main(String[] args) {

        // ============================================================
        // 1. A TINY SYNTHETIC MODEL WITH REAL WEIGHT NAMES
        // ============================================================
        // The finder/orthogonalizer locate weights by regex on variable names
        // (o_proj / down_proj / lm_head by default), so we name ours accordingly.
        System.out.println("=== 1. Synthetic model ===");
        SameDiff model = SameDiff.create();
        for (int layer = 0; layer < LAYERS; layer++) {
            model.var("model.layers." + layer + ".self_attn.o_proj.weight",
                    Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.1));
            model.var("model.layers." + layer + ".mlp.down_proj.weight",
                    Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.1));
        }
        model.var("lm_head.weight", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.1));
        long editableWeights = model.variables().stream()
                .filter(v -> v.name().matches(".*(o_proj|down_proj|lm_head).*")).count();
        System.out.println("  Model variables: " + model.variables().size()
                + " (" + editableWeights + " match the default target-weight patterns)");

        // ============================================================
        // 2. SYNTHETIC ACTIVATIONS SEPARATED ALONG A PLANTED DIRECTION
        // ============================================================
        // Class A = mean + planted, Class B = mean - planted, plus isotropic noise.
        // A real run would instead collect these from forward passes on two contrasting
        // prompt sets; here we plant the direction so we can check the recovery.
        System.out.println("\n=== 2. Synthetic activations (planted direction) ===");
        INDArray planted = Nd4j.randn(DataType.FLOAT, HIDDEN);
        planted.divi(planted.norm2Number().doubleValue());   // unit vector

        String probeVar = "model.layers.1.mlp.down_proj.weight";
        INDArray baseMean = Nd4j.randn(DataType.FLOAT, HIDDEN).muli(0.5);
        INDArray classA = baseMean.reshape(1, HIDDEN).broadcast(SAMPLES, HIDDEN)
                .add(planted.reshape(1, HIDDEN).mul(2.0))
                .add(Nd4j.randn(DataType.FLOAT, SAMPLES, HIDDEN).muli(0.3));
        INDArray classB = baseMean.reshape(1, HIDDEN).broadcast(SAMPLES, HIDDEN)
                .sub(planted.reshape(1, HIDDEN).mul(2.0))
                .add(Nd4j.randn(DataType.FLOAT, SAMPLES, HIDDEN).muli(0.3));

        Map<String, INDArray> activationsA = new HashMap<>();
        Map<String, INDArray> activationsB = new HashMap<>();
        activationsA.put(probeVar, classA);
        activationsB.put(probeVar, classB);
        System.out.println("  Probe point: " + probeVar);
        System.out.println("  Activations per class: " + Arrays.toString(classA.shape()));

        // ============================================================
        // 3. FIND THE SEPARATING DIRECTION
        // ============================================================
        System.out.println("\n=== 3. RefusalDirectionFinder ===");
        AbliterationConfig config = AbliterationConfig.builder()
                .method(Method.DIFF_IN_MEANS)               // also: PCA, PROJECTED
                .layerSelectionStrategy(LayerSelection.BEST_SINGLE)
                .ablationStrength(1.0)                        // 1.0 = full removal
                .build();

        RefusalDirectionFinder finder = new RefusalDirectionFinder();
        List<RefusalDirection> candidates =
                finder.findRefusalDirections(model, activationsA, activationsB, config);

        RefusalDirection found = candidates.get(0);
        System.out.println("  Candidates found: " + candidates.size());
        System.out.println("  Top: " + found);

        // Verify recovery: |cos(found, planted)| should be ~1 (sign is arbitrary).
        INDArray dir = found.getDirection().castTo(DataType.FLOAT);
        double cos = Nd4j.getBlasWrapper().dot(dir, planted)
                / (dir.norm2Number().doubleValue() * planted.norm2Number().doubleValue());
        System.out.println("  |cosine(recovered, planted)| = " + String.format("%.4f", Math.abs(cos)));
        check(Math.abs(cos) > 0.9, "recovered direction should align with the planted one");

        // ============================================================
        // 4. ORTHOGONALIZE THE WEIGHTS AGAINST IT
        // ============================================================
        System.out.println("\n=== 4. WeightOrthogonalizer ===");
        // Snapshot one weight's alignment with the direction before editing.
        INDArray wBefore = model.getVariable(probeVar).getArr().dup();
        double alignBefore = rowSpaceAlignment(wBefore, dir);

        WeightOrthogonalizer orthogonalizer = new WeightOrthogonalizer();
        List<String> modified = orthogonalizer.orthogonalizeWeights(
                model, candidates.subList(0, 1), config);

        INDArray wAfter = model.getVariable(probeVar).getArr();
        double alignAfter = rowSpaceAlignment(wAfter, dir);
        System.out.println("  Weights modified: " + modified.size());
        System.out.println("  Row-space alignment with direction: "
                + String.format("%.4f -> %.4f", alignBefore, alignAfter));
        check(alignAfter < alignBefore * 0.05,
                "edited weights should be (near-)orthogonal to the ablated direction");

        // ============================================================
        // 5. METHOD / STRENGTH VARIANTS
        // ============================================================
        System.out.println("\n=== 5. Config variants ===");
        System.out.println("  defaults():        method=" + AbliterationConfig.defaults().getMethod()
                + " strength=" + AbliterationConfig.defaults().getAblationStrength());
        System.out.println("  conservative():    method=" + AbliterationConfig.conservative().getMethod()
                + " strength=" + AbliterationConfig.conservative().getAblationStrength()
                + "  (PROJECTED, 0.8 — minimizes collateral capability loss)");
        System.out.println("  withWinsorization(): winsorize="
                + AbliterationConfig.withWinsorization().isWinsorize()
                + "  (clips activation outliers — needed for e.g. Gemma)");

        // Partial-strength ablation on a fresh copy: alpha=0.5 removes half the component.
        SameDiff model2 = model.dup();
        AbliterationConfig half = AbliterationConfig.builder().ablationStrength(0.5).build();
        INDArray freshW = Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.1);
        model2.getVariable(probeVar).setArray(freshW.dup());
        new WeightOrthogonalizer().orthogonalizeWeights(model2, candidates.subList(0, 1), half);
        double partial = rowSpaceAlignment(model2.getVariable(probeVar).getArr(), dir)
                / rowSpaceAlignment(freshW, dir);
        System.out.println("  alpha=0.5 leaves ~" + String.format("%.0f%%", partial * 100)
                + " of the direction component (partial ablation)");

        // ============================================================
        // 6. END-TO-END WORKFLOW ON A REAL MODEL (reference)
        // ============================================================
        System.out.println("\n=== 6. AbliterationWorkflow (end-to-end shape) ===");
        System.out.println("  On an imported GGUF/ONNX decoder the whole pipeline is one call.");
        System.out.println("  DefaultPromptSets provides representative contrasting prompt sets;");
        System.out.println("  the workflow collects activations, finds directions, and edits weights:");
        System.out.println();
        System.out.println("    AbliterationResult r = AbliterationWorkflow.builder()");
        System.out.println("        .model(importedDecoder)");
        System.out.println("        .config(AbliterationConfig.conservative())");
        System.out.println("        .harmfulPrompts(DefaultPromptSets.getDefaultHarmfulPrompts())");
        System.out.println("        .harmlessPrompts(DefaultPromptSets.getDefaultHarmlessPrompts())");
        System.out.println("        .modelInputs(inputMap)");
        System.out.println("        .build().execute();");
        System.out.println();
        // Drive the workflow object with our pre-computed synthetic activations so the
        // orchestration path itself is exercised (no model download / forward pass needed).
        AbliterationResult result = AbliterationWorkflow.builder()
                .model(model.dup())
                .config(config)
                .harmfulActivations(activationsA)
                .harmlessActivations(activationsB)
                .build()
                .execute();
        System.out.println("  Workflow result: " + result);
        check(result.getNumWeightsModified() > 0, "workflow should modify at least one weight");

        System.out.println("\nAblation pipeline verified: direction recovered and weights orthogonalized.");
    }

    /**
     * Alignment of a weight matrix's row space with a direction: mean magnitude of the
     * projection of each row onto the unit direction. Goes to ~0 after orthogonalization.
     */
    private static double rowSpaceAlignment(INDArray weight, INDArray unitDir) {
        INDArray proj = weight.mmul(unitDir.reshape(unitDir.length(), 1)); // [rows, 1]
        return Nd4j.math().abs(proj).meanNumber().doubleValue();
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("FAILED: " + message);
        }
    }
}

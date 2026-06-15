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

package org.nd4j.examples.samediff.quickstart.training;

import org.nd4j.autodiff.samediff.config.DistillationConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Knowledge Distillation Configuration Examples.
 *
 * Knowledge Distillation (KD) transfers knowledge from a large "teacher" model
 * to a smaller "student" model. SameDiff supports three distillation types:
 *
 *   1. Logit-KD  (LOGIT_KD)     - Soft label distillation via KL divergence on softmax outputs
 *   2. Feature-KD (FEATURE_KD)  - Match intermediate hidden states between teacher and student
 *   3. Attention-KD (ATTN_KD)   - Match attention weight distributions across layers
 *   4. Combined (COMBINED)       - Weighted combination of all three
 *
 * Key concepts:
 *   - Temperature (T): higher T softens probability distributions, revealing more information.
 *     Typical range: 2-10. Loss = alpha * T^2 * KL(soft_teacher || soft_student)
 *   - Alpha: weight between distillation loss and hard-label cross-entropy loss.
 *     alpha=1.0 means pure distillation; alpha=0.0 means pure hard-label training.
 *   - Temperature annealing: start with high T (soft) and decay to low T (hard) during training.
 */
public class KnowledgeDistillationConfigExample {
    private static final Logger log = LoggerFactory.getLogger(KnowledgeDistillationConfigExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Logit-KD — Soft Label Distillation (most common)
        // =====================================================================
        log.info("=== 1. Logit Knowledge Distillation ===");

        // Standard KD from Hinton et al. 2015.
        // Loss = alpha * T^2 * KL(student_soft || teacher_soft) + (1-alpha) * CE(student, hard_labels)
        // where soft = softmax(logits / T)
        DistillationConfig logitKD = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.LOGIT_KD)
                .studentLogitVariable("student_logits")   // SameDiff variable name in student model
                .teacherLogitVariable("teacher_logits")   // SameDiff variable name from teacher
                .temperature(4.0)                          // T=4: softens teacher distribution considerably
                .alpha(0.5)                                // 50% distillation, 50% hard labels
                .build();

        logitKD.validate();
        log.info("  Logit KD config:");
        log.info("    Type: {}", logitKD.getDistillationType());
        log.info("    Temperature: {}", logitKD.getTemperature());
        log.info("    Alpha (distillation weight): {}", logitKD.getAlpha());
        log.info("    Student logit variable: {}", logitKD.getStudentLogitVariable());
        log.info("    Teacher logit variable: {}", logitKD.getTeacherLogitVariable());

        // Factory shortcut: (studentVar, teacherVar, temperature, alpha)
        DistillationConfig logitKDFactory = DistillationConfig.logitKD(
                "student_logits", "teacher_logits", 4.0, 0.5
        );
        log.info("  From factory: type={}, T={}, alpha={}",
                logitKDFactory.getDistillationType(),
                logitKDFactory.getTemperature(),
                logitKDFactory.getAlpha());

        // Temperature sensitivity
        log.info("  --- Temperature Effect ---");
        log.info("  T=1:   hard student distribution (equivalent to cross-entropy)");
        log.info("  T=4:   moderately soft (reveals class relationships, common default)");
        log.info("  T=10:  very soft (maximum information from teacher dark knowledge)");
        log.info("  T=20:  extremely soft (used when teacher is much larger than student)");

        // =====================================================================
        // 2. Feature-KD — Intermediate Layer Matching
        // =====================================================================
        log.info("=== 2. Feature Knowledge Distillation ===");

        // Match hidden state activations between teacher and student intermediate layers.
        // Useful when teacher and student have different depth/width
        // (student layers are mapped to teacher layers).
        Map<String, String> featureMappings = new LinkedHashMap<>();
        featureMappings.put("student_layer_4_hidden", "teacher_layer_8_hidden");   // Student L4 -> Teacher L8
        featureMappings.put("student_layer_8_hidden", "teacher_layer_16_hidden");  // Student L8 -> Teacher L16
        featureMappings.put("student_layer_12_hidden", "teacher_layer_24_hidden"); // Student L12 -> Teacher L24

        DistillationConfig featureKD = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.FEATURE_KD)
                .studentLogitVariable("student_logits")
                .teacherLogitVariable("teacher_logits")
                .temperature(4.0)
                .alpha(0.5)
                .featureLayerMappings(featureMappings)   // student_var -> teacher_var
                .featureLossWeight(1.0)                  // Weight for feature matching loss
                .build();

        featureKD.validate();
        log.info("  Feature KD:");
        log.info("    Layer mappings (student -> teacher):");
        for (Map.Entry<String, String> entry : featureKD.getFeatureLayerMappings().entrySet()) {
            log.info("      {} -> {}", entry.getKey(), entry.getValue());
        }
        log.info("    Feature loss weight: {}", featureKD.getFeatureLossWeight());

        // Factory shortcut
        DistillationConfig featureKDFactory = DistillationConfig.featureKD(featureMappings);
        log.info("  From factory: {} layer mappings", featureKDFactory.getFeatureLayerMappings().size());

        // =====================================================================
        // 3. Attention-KD — Attention Distribution Matching
        // =====================================================================
        log.info("=== 3. Attention Knowledge Distillation ===");

        // Match attention weight distributions (attention score matrices) between models.
        // AKD from BERT-PKD and TinyBERT papers.
        // Forces student to attend to similar positions as the teacher.
        Map<String, String> attentionMappings = new LinkedHashMap<>();
        attentionMappings.put("student_attn_layer_4", "teacher_attn_layer_8");
        attentionMappings.put("student_attn_layer_8", "teacher_attn_layer_16");
        attentionMappings.put("student_attn_layer_12", "teacher_attn_layer_24");

        DistillationConfig attentionKD = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.ATTENTION_KD)
                .studentLogitVariable("student_logits")
                .teacherLogitVariable("teacher_logits")
                .temperature(4.0)
                .alpha(0.5)
                .attentionLayerMappings(attentionMappings)  // student_attn -> teacher_attn
                .attentionLossWeight(1.0)                   // Weight for attention matching loss
                .build();

        attentionKD.validate();
        log.info("  Attention KD:");
        log.info("    Attention mappings (student -> teacher):");
        for (Map.Entry<String, String> entry : attentionKD.getAttentionLayerMappings().entrySet()) {
            log.info("      {} -> {}", entry.getKey(), entry.getValue());
        }
        log.info("    Attention loss weight: {}", attentionKD.getAttentionLossWeight());

        // =====================================================================
        // 4. Combined — All Three Types Together
        // =====================================================================
        log.info("=== 4. Combined Distillation ===");

        // Use logit + feature + attention distillation simultaneously.
        // Total loss = logitWeight * logit_loss
        //            + featureWeight * feature_loss
        //            + attentionWeight * attention_loss
        //            + (1 - alpha) * hard_label_CE
        DistillationConfig combinedKD = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.COMBINED)
                .studentLogitVariable("student_logits")
                .teacherLogitVariable("teacher_logits")
                .temperature(4.0)
                .alpha(0.7)                         // 70% distillation, 30% hard labels
                .featureLayerMappings(featureMappings)
                .attentionLayerMappings(attentionMappings)
                .logitLossWeight(1.0)               // Equal weight for all three losses
                .featureLossWeight(1.0)
                .attentionLossWeight(1.0)
                .build();

        combinedKD.validate();
        log.info("  Combined KD:");
        log.info("    Type: {}", combinedKD.getDistillationType());
        log.info("    Logit weight: {}", combinedKD.getLogitLossWeight());
        log.info("    Feature weight: {}", combinedKD.getFeatureLossWeight());
        log.info("    Attention weight: {}", combinedKD.getAttentionLossWeight());

        // =====================================================================
        // 5. Temperature Annealing
        // =====================================================================
        log.info("=== 5. Temperature Annealing ===");

        // Temperature annealing: start with high T (soft distributions, more information)
        // and linearly decay to low T (hard distributions, sharper gradients) over training.
        DistillationConfig annealedKD = DistillationConfig.builder()
                .distillationType(DistillationConfig.DistillationType.LOGIT_KD)
                .studentLogitVariable("student_logits")
                .teacherLogitVariable("teacher_logits")
                .temperature(1.0)                   // Final temperature (used if annealing disabled)
                .alpha(0.5)
                .temperatureAnnealing(true)          // Enable temperature annealing
                .initialTemperature(10.0)            // Start temperature (high = very soft)
                .finalTemperature(1.0)               // End temperature (low = near-hard labels)
                .build();

        annealedKD.validate();
        log.info("  Annealed KD config:");
        log.info("    Annealing enabled: {}", annealedKD.isTemperatureAnnealing());
        log.info("    Initial T: {}", annealedKD.getInitialTemperature());
        log.info("    Final T: {}", annealedKD.getFinalTemperature());

        // Check effective temperature at various progress points
        log.info("  Effective temperature during training:");
        for (double progress : new double[]{0.0, 0.25, 0.5, 0.75, 1.0}) {
            double effectiveT = annealedKD.getEffectiveTemperature(progress);
            log.info("    Progress {:.0f}%: T = {:.2f}", progress * 100, effectiveT);
        }

        // =====================================================================
        // SUMMARY
        // =====================================================================
        log.info("=== Knowledge Distillation Guide ===");
        log.info("  Recommended configurations:");
        log.info("  - General compression:          Logit-KD, T=4, alpha=0.5");
        log.info("  - BERT/GPT to smaller:          Combined KD with 1:1:1 weights");
        log.info("  - Very large teacher (GPT-4):   Logit-KD, T=10-20, alpha=0.9");
        log.info("  - Feature extraction tasks:     Feature-KD (match specific layers)");
        log.info("  - Attention-heavy tasks:        Attention-KD or Combined");
        log.info("  - Dynamic curriculum:           Temperature annealing (T: 10 -> 1)");
        log.info("  Tips:");
        log.info("  - Temperature > 1 always: T=1 degenerates to cross-entropy on argmax");
        log.info("  - alpha=1.0: pure distillation (no hard label supervision)");
        log.info("  - alpha=0.0: ignore teacher entirely (useless; just use CE directly)");
        log.info("  - Student and teacher variable names must match SameDiff graph variable names");
        log.info("**************** Knowledge Distillation Config Example finished ********************");
    }
}

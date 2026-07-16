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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.DistillationTrainer;
import org.nd4j.autodiff.samediff.config.DistillationConfig;
import org.nd4j.autodiff.samediff.config.DistillationConfig.DistillationType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.loss.AttentionDistillationLoss;
import org.nd4j.linalg.api.ops.impl.loss.DistillationKLLoss;
import org.nd4j.linalg.api.ops.impl.loss.FeatureDistillationLoss;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Knowledge Distillation Example.
 *
 * Knowledge distillation transfers knowledge from a large "teacher" model
 * to a smaller "student" model, achieving near-teacher accuracy with
 * significantly fewer parameters.
 *
 * <h3>Distillation Types ({@link DistillationType}):</h3>
 * <ul>
 *   <li>LOGIT_KD — Soft target matching (Hinton et al. 2015). The student learns
 *       to match the teacher's output probability distribution, which contains
 *       "dark knowledge" about inter-class relationships.</li>
 *   <li>FEATURE_KD — Intermediate feature matching. The student learns to
 *       reproduce the teacher's intermediate representations, potentially
 *       using a projection matrix when dimensions differ.</li>
 *   <li>ATTENTION_KD — Attention map matching. The student learns to mimic
 *       the teacher's attention patterns across layers.</li>
 *   <li>COMBINED — Combines logit, feature, and attention distillation with
 *       configurable loss weights.</li>
 * </ul>
 *
 * <h3>Key Classes:</h3>
 * <ul>
 *   <li>{@link DistillationConfig} — Configuration: type, temperature, alpha, layer mappings</li>
 *   <li>{@link DistillationTrainer} — Training loop for teacher-student distillation</li>
 *   <li>{@link DistillationKLLoss} — KL divergence loss with temperature scaling</li>
 *   <li>{@link FeatureDistillationLoss} — Feature-matching MSE loss with optional projection</li>
 *   <li>{@link AttentionDistillationLoss} — Attention map matching loss</li>
 *   <li>{@link CosineWarmupSchedule} — Cosine annealing with linear warmup</li>
 * </ul>
 *
 * <h3>Temperature Scaling:</h3>
 * Higher temperature (e.g., T=4.0) produces softer probability distributions,
 * making inter-class relationships more visible to the student. Temperature
 * annealing starts high and decreases during training, shifting focus from
 * exploration to precision.
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.training.KnowledgeDistillationExample"
 */
public class KnowledgeDistillationExample {
    private static final Logger log = LoggerFactory.getLogger(KnowledgeDistillationExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build Teacher and Student Models
        // =====================================================================
        log.info("=== 1. Building Teacher and Student Models ===");

        // Teacher: larger model (128→256→128→10)
        SameDiff teacher = SameDiff.create();
        SDVariable tInput = teacher.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable tW1 = teacher.var("t_w1", Nd4j.randn(128, 256).muli(0.01));
        SDVariable tB1 = teacher.var("t_b1", Nd4j.zeros(256));
        SDVariable tW2 = teacher.var("t_w2", Nd4j.randn(256, 128).muli(0.01));
        SDVariable tB2 = teacher.var("t_b2", Nd4j.zeros(128));
        SDVariable tW3 = teacher.var("t_w3", Nd4j.randn(128, 10).muli(0.01));
        SDVariable tB3 = teacher.var("t_b3", Nd4j.zeros(10));

        SDVariable tZ1 = tInput.mmul(tW1).add(tB1);
        SDVariable tH1 = teacher.nn.relu("t_hidden1", tZ1, 0);
        SDVariable tZ2 = tH1.mmul(tW2).add(tB2);
        SDVariable tH2 = teacher.nn.relu("t_hidden2", tZ2, 0);
        SDVariable tLogits = tH2.mmul(tW3).add(tB3).rename("t_logits");
        SDVariable tOutput = teacher.nn.softmax("t_output", tLogits, -1);

        log.info("Teacher model: {} variables, {} ops",
                teacher.variables().size(), teacher.ops().length);

        // Student: smaller model (128→64→10)
        SameDiff student = SameDiff.create();
        SDVariable sInput = student.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable sW1 = student.var("s_w1", Nd4j.randn(128, 64).muli(0.01));
        SDVariable sB1 = student.var("s_b1", Nd4j.zeros(64));
        SDVariable sW2 = student.var("s_w2", Nd4j.randn(64, 10).muli(0.01));
        SDVariable sB2 = student.var("s_b2", Nd4j.zeros(10));

        SDVariable sZ1 = sInput.mmul(sW1).add(sB1);
        SDVariable sH1 = student.nn.relu("s_hidden1", sZ1, 0);
        SDVariable sLogits = sH1.mmul(sW2).add(sB2).rename("s_logits");
        SDVariable sOutput = student.nn.softmax("s_output", sLogits, -1);

        log.info("Student model: {} variables, {} ops",
                student.variables().size(), student.ops().length);

        // =====================================================================
        // 2. Logit Knowledge Distillation (Hinton et al.)
        // =====================================================================
        log.info("\n=== 2. Logit Knowledge Distillation ===");

        // Static factory for logit-based KD
        // Parameters: studentLogitVar, teacherLogitVar, temperature, alpha
        // alpha controls the balance: (1-alpha)*hardLoss + alpha*softLoss
        DistillationConfig logitConfig = DistillationConfig.logitKD(
                "s_logits",   // student logit variable name
                "t_logits",   // teacher logit variable name
                4.0,          // temperature (higher = softer distributions)
                0.7           // alpha (0.7 = 70% soft loss, 30% hard loss)
        );

        log.info("Logit KD config:");
        log.info("  Type: {}", logitConfig.getDistillationType());
        log.info("  Temperature: {}", logitConfig.getTemperature());
        log.info("  Alpha: {}", logitConfig.getAlpha());
        log.info("  Student logit var: {}", logitConfig.getStudentLogitVariable());
        log.info("  Teacher logit var: {}", logitConfig.getTeacherLogitVariable());

        // Validate the config
        logitConfig.validate();
        log.info("  Config validated successfully");

        // Create DistillationTrainer
        DistillationTrainer logitTrainer = new DistillationTrainer(student, teacher, logitConfig);
        log.info("  Trainer created: student={}, teacher={}", "student model", "teacher model");

        // Run a training step
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.randn(32, 128));  // same input for both models
        double loss = logitTrainer.trainStep(inputs);
        log.info("  Training step loss: {}", loss);

        // =====================================================================
        // 3. Feature Knowledge Distillation
        // =====================================================================
        log.info("\n=== 3. Feature Knowledge Distillation ===");

        // Feature KD matches intermediate layer representations
        // Maps: student layer name → teacher layer name
        Map<String, String> featureMappings = new LinkedHashMap<>();
        featureMappings.put("s_hidden1", "t_hidden1");  // student 64d → teacher 256d

        DistillationConfig featureConfig = DistillationConfig.featureKD(featureMappings);
        log.info("Feature KD config:");
        log.info("  Type: {}", featureConfig.getDistillationType());
        log.info("  Feature mappings: {}", featureConfig.getFeatureLayerMappings());
        log.info("  Feature loss weight: {}", featureConfig.getFeatureLossWeight());

        // =====================================================================
        // 4. Attention Knowledge Distillation
        // =====================================================================
        log.info("\n=== 4. Attention Knowledge Distillation ===");

        // Attention KD matches attention patterns across layers.
        // Useful for transformer models — maps student attention layers to teacher's.
        Map<String, String> attentionMappings = new LinkedHashMap<>();
        attentionMappings.put("s_attn_layer_0", "t_attn_layer_0");
        attentionMappings.put("s_attn_layer_1", "t_attn_layer_3");

        DistillationConfig attnConfig = DistillationConfig.builder()
                .distillationType(DistillationType.ATTENTION_KD)
                .attentionLayerMappings(attentionMappings)
                .attentionLossWeight(1.0)
                .build();

        log.info("Attention KD config:");
        log.info("  Type: {}", attnConfig.getDistillationType());
        log.info("  Attention mappings: {}", attnConfig.getAttentionLayerMappings());
        log.info("  Attention loss weight: {}", attnConfig.getAttentionLossWeight());

        // =====================================================================
        // 5. Combined Distillation
        // =====================================================================
        log.info("\n=== 5. Combined Distillation ===");

        // Combines all three distillation types with configurable loss weights
        DistillationConfig combinedConfig = DistillationConfig.builder()
                .distillationType(DistillationType.COMBINED)
                .studentLogitVariable("s_logits")
                .teacherLogitVariable("t_logits")
                .temperature(4.0)
                .alpha(0.5)
                .featureLayerMappings(featureMappings)
                .featureLossWeight(0.5)       // Weight for feature matching loss
                .attentionLayerMappings(attentionMappings)
                .attentionLossWeight(0.3)     // Weight for attention matching loss
                .logitLossWeight(1.0)         // Weight for logit KD loss
                .build();

        log.info("Combined config:");
        log.info("  Type: {}", combinedConfig.getDistillationType());
        log.info("  Logit loss weight: {}", combinedConfig.getLogitLossWeight());
        log.info("  Feature loss weight: {}", combinedConfig.getFeatureLossWeight());
        log.info("  Attention loss weight: {}", combinedConfig.getAttentionLossWeight());

        // =====================================================================
        // 6. Temperature Annealing
        // =====================================================================
        log.info("\n=== 6. Temperature Annealing ===");

        // Temperature annealing starts with a high temperature (wide exploration)
        // and decreases to a low temperature (precise matching) during training.
        DistillationConfig annealingConfig = DistillationConfig.builder()
                .distillationType(DistillationType.LOGIT_KD)
                .studentLogitVariable("s_logits")
                .teacherLogitVariable("t_logits")
                .temperatureAnnealing(true)     // Enable annealing
                .initialTemperature(10.0)       // Start high (very soft)
                .finalTemperature(1.0)          // End low (sharp)
                .alpha(0.7)
                .build();

        log.info("Temperature annealing schedule:");
        // progress goes from 0.0 (start) to 1.0 (end)
        for (double progress = 0.0; progress <= 1.0; progress += 0.25) {
            double effectiveTemp = annealingConfig.getEffectiveTemperature(progress);
            log.info("  Progress {}: temperature = {}", String.format("%.0f%%", progress * 100), effectiveTemp);
        }

        // Without annealing, temperature is constant
        log.info("  Without annealing: temperature = {} (constant)", logitConfig.getTemperature());

        // =====================================================================
        // 7. Self-Distillation with EMA
        // =====================================================================
        log.info("\n=== 7. Self-Distillation with EMA ===");

        // Self-distillation uses the SAME model as both teacher and student.
        // The teacher is an Exponential Moving Average (EMA) of the student weights.
        // This is simpler than traditional KD — no separate teacher model needed.
        DistillationConfig selfDistillConfig = DistillationConfig.builder()
                .distillationType(DistillationType.LOGIT_KD)
                .studentLogitVariable("s_logits")
                .teacherLogitVariable("s_logits")  // same variable
                .temperature(4.0)
                .alpha(0.5)
                .build();

        DistillationTrainer selfTrainer = DistillationTrainer.selfDistillation(
                student, selfDistillConfig);

        log.info("Self-distillation trainer created");
        log.info("  Student and teacher are the SAME model");
        log.info("  Teacher weights = EMA of student weights");

        // During training, periodically update the teacher's EMA weights
        double emaDecay = 0.999;  // how fast old weights decay (higher = slower update)
        selfTrainer.refreshTeacherEMA(emaDecay);
        log.info("  EMA refreshed with decay={}", emaDecay);

        // Typical self-distillation training loop:
        log.info("\n  Self-distillation training loop pattern:");
        log.info("    for each batch:");
        log.info("      loss = selfTrainer.trainStep(batchInputs);");
        log.info("      selfTrainer.refreshTeacherEMA(0.999);  // update teacher EMA");

        // =====================================================================
        // 8. Distillation Loss Ops — Direct Usage
        // =====================================================================
        log.info("\n=== 8. Distillation Loss Ops ===");

        // These ops can also be used directly in custom SameDiff graphs,
        // outside of the DistillationTrainer framework.

        // --- DistillationKLLoss ---
        log.info("DistillationKLLoss (op: 'distillation_kl_loss'):");
        log.info("  Computes: (1-alpha)*CE(student,hard) + alpha*T^2*KL(soft_student||soft_teacher)");
        log.info("  where soft = softmax(logits/T)");

        // INDArray version
        INDArray studentLogits = Nd4j.randn(32, 10);  // [batch, classes]
        INDArray teacherLogits = Nd4j.randn(32, 10);
        DistillationKLLoss klLoss = new DistillationKLLoss(
                studentLogits, teacherLogits,
                4.0,    // temperature
                0.7     // alpha
        );
        log.info("  Created with INDArrays: temp={}, alpha={}", 4.0, 0.7);

        // With hard labels (3-input form)
        INDArray hardLabels = Nd4j.zeros(32, 10);
        // Set one-hot labels
        for (int i = 0; i < 32; i++) {
            hardLabels.putScalar(i, i % 10, 1.0);
        }
        DistillationKLLoss klLossWithHard = new DistillationKLLoss(
                studentLogits, teacherLogits, hardLabels,
                4.0, 0.7
        );
        log.info("  With hard labels: includes cross-entropy term");

        // SameDiff version (for use in computation graphs)
        SameDiff lossGraph = SameDiff.create();
        SDVariable sLog = lossGraph.var("sLogits", studentLogits);
        SDVariable tLog = lossGraph.var("tLogits", teacherLogits);
        DistillationKLLoss sdKlLoss = new DistillationKLLoss(
                lossGraph, sLog, tLog, 4.0, 0.7);
        log.info("  SameDiff version: builds into computation graph");

        // --- FeatureDistillationLoss ---
        log.info("\nFeatureDistillationLoss (op: 'feature_distillation_loss'):");
        log.info("  Computes: MSE(project(student_features), teacher_features)");

        INDArray studentFeatures = Nd4j.randn(32, 64);   // [batch, student_dim]
        INDArray teacherFeatures = Nd4j.randn(32, 256);  // [batch, teacher_dim]

        // Without projection (dimensions must match)
        INDArray matchingStudentFeatures = Nd4j.randn(32, 256);
        FeatureDistillationLoss featureLoss = new FeatureDistillationLoss(
                matchingStudentFeatures, teacherFeatures);
        log.info("  Without projection: student [32,256] → teacher [32,256]");

        // With projection matrix (when dimensions differ)
        INDArray projectionWeight = Nd4j.randn(64, 256);  // [student_dim, teacher_dim]
        FeatureDistillationLoss featureLossWithProj = new FeatureDistillationLoss(
                studentFeatures, teacherFeatures, projectionWeight);
        log.info("  With projection: student [32,64] × proj [64,256] → teacher [32,256]");

        // --- AttentionDistillationLoss ---
        log.info("\nAttentionDistillationLoss (op: 'attention_distillation_loss'):");
        log.info("  Computes: MSE(student_attention, avg(teacher_attention))");
        log.info("  Averages teacher heads if counts differ");

        INDArray studentAttn = Nd4j.randn(32, 4, 16, 16);   // [batch, s_heads, seq, seq]
        INDArray teacherAttn = Nd4j.randn(32, 8, 16, 16);   // [batch, t_heads, seq, seq]
        AttentionDistillationLoss attnLoss = new AttentionDistillationLoss(
                studentAttn, teacherAttn);
        log.info("  Student: 4 heads, Teacher: 8 heads → teacher heads averaged");

        // =====================================================================
        // 9. CosineWarmupSchedule — Learning Rate Scheduling
        // =====================================================================
        log.info("\n=== 9. CosineWarmupSchedule ===");

        // Linear warmup followed by cosine decay
        // Commonly used for distillation and fine-tuning
        CosineWarmupSchedule schedule = new CosineWarmupSchedule(
                1e-4,    // maxLR — peak learning rate after warmup
                1e-6,    // minLR — minimum learning rate at end
                500,     // warmupSteps — linear warmup steps
                10000    // totalSteps — total training steps
        );

        log.info("CosineWarmupSchedule:");
        log.info("  maxLR={}, minLR={}, warmupSteps={}, totalSteps={}",
                1e-4, 1e-6, 500, 10000);

        // Show LR values at various points
        for (int step : new int[]{0, 100, 250, 500, 2500, 5000, 7500, 10000}) {
            double lr = schedule.valueAt(step, 0);
            log.info("  Step {}: LR = {}", String.format("%5d", step), String.format("%.6f", lr));
        }

        // Simplified constructor (minLR defaults to 0.0)
        CosineWarmupSchedule simpleSchedule = new CosineWarmupSchedule(
                1e-4, 500, 10000);
        log.info("\nSimplified (minLR=0): step 0 LR = {}", simpleSchedule.valueAt(0, 0));

        // From warmup ratio (e.g., 10% warmup)
        CosineWarmupSchedule ratioSchedule = CosineWarmupSchedule.fromRatio(
                1e-4,    // maxLR
                1e-6,    // minLR
                0.1,     // warmupRatio (10% of total steps)
                10000    // totalSteps
        );
        log.info("From ratio (10% warmup): warmupSteps={}",
                (int)(0.1 * 10000));

        // =====================================================================
        // 10. Complete Distillation Training Pattern
        // =====================================================================
        log.info("\n=== 10. Complete Distillation Training Pattern ===");

        log.info("Typical knowledge distillation workflow:");
        log.info("");
        log.info("  // 1. Load or build teacher (pre-trained, larger model)");
        log.info("  SameDiff teacher = SameDiff.load(teacherFile, false);");
        log.info("");
        log.info("  // 2. Build student (smaller architecture)");
        log.info("  SameDiff student = buildStudentModel();");
        log.info("");
        log.info("  // 3. Configure distillation");
        log.info("  DistillationConfig config = DistillationConfig.logitKD(");
        log.info("      \"student_logits\", \"teacher_logits\", 4.0, 0.7);");
        log.info("");
        log.info("  // 4. Create trainer");
        log.info("  DistillationTrainer trainer = new DistillationTrainer(");
        log.info("      student, teacher, config);");
        log.info("");
        log.info("  // 5. Training loop with cosine schedule");
        log.info("  CosineWarmupSchedule schedule = new CosineWarmupSchedule(");
        log.info("      1e-4, 1e-6, 500, totalSteps);");
        log.info("");
        log.info("  for (int step = 0; step < totalSteps; step++) {");
        log.info("      Map<String,INDArray> batch = dataIterator.next();");
        log.info("      double loss = trainer.trainStep(batch);");
        log.info("      double lr = schedule.valueAt(step, 0);");
        log.info("      // apply lr to optimizer...");
        log.info("      if (step % 100 == 0) log.info(\"Step {} loss {}\", step, loss);");
        log.info("  }");
        log.info("");
        log.info("  // 6. Export the trained student");
        log.info("  student.save(outputFile, true);");

        // =====================================================================
        // 11. Distillation for LLMs
        // =====================================================================
        log.info("\n=== 11. Distillation for LLMs ===");
        log.info("For LLM distillation, feature and attention KD are most effective:");
        log.info("");
        log.info("  // Map student transformer layers to teacher layers");
        log.info("  // (e.g., 6-layer student from 24-layer teacher)");
        log.info("  Map<String,String> features = new LinkedHashMap<>();");
        log.info("  features.put(\"student.layer.0.output\", \"teacher.layer.3.output\");");
        log.info("  features.put(\"student.layer.1.output\", \"teacher.layer.7.output\");");
        log.info("  features.put(\"student.layer.2.output\", \"teacher.layer.11.output\");");
        log.info("  features.put(\"student.layer.3.output\", \"teacher.layer.15.output\");");
        log.info("  features.put(\"student.layer.4.output\", \"teacher.layer.19.output\");");
        log.info("  features.put(\"student.layer.5.output\", \"teacher.layer.23.output\");");
        log.info("");
        log.info("  Map<String,String> attention = new LinkedHashMap<>();");
        log.info("  attention.put(\"student.layer.0.attn\", \"teacher.layer.3.attn\");");
        log.info("  // ... similar mapping ...");
        log.info("");
        log.info("  DistillationConfig config = DistillationConfig.builder()");
        log.info("      .distillationType(DistillationType.COMBINED)");
        log.info("      .studentLogitVariable(\"lm_head_logits\")");
        log.info("      .teacherLogitVariable(\"lm_head_logits\")");
        log.info("      .temperature(4.0)");
        log.info("      .alpha(0.5)");
        log.info("      .featureLayerMappings(features)");
        log.info("      .featureLossWeight(0.5)");
        log.info("      .attentionLayerMappings(attention)");
        log.info("      .attentionLossWeight(0.3)");
        log.info("      .temperatureAnnealing(true)");
        log.info("      .initialTemperature(10.0)");
        log.info("      .finalTemperature(2.0)");
        log.info("      .build();");

        log.info("\n**************** Knowledge Distillation Example finished ********************");
    }
}

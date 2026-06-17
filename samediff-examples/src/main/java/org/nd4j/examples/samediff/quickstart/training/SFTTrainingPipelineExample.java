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
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.QLoraConfig;
import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.config.TaskType;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.training.SFTTrainingPipeline;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.curation.batching.LengthBucketingIterator;
import org.nd4j.linalg.dataset.curation.dedup.TextDeduplicator;
import org.nd4j.linalg.dataset.curation.filtering.FilterResult;
import org.nd4j.linalg.dataset.curation.filtering.TextQualityFilter;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter;
import org.nd4j.linalg.dataset.curation.mixing.WeightedDataMixer;
import org.nd4j.linalg.dataset.curation.splitting.SplitResult;
import org.nd4j.linalg.dataset.curation.splitting.StratifiedSplitter;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Complete Supervised Fine-Tuning (SFT) Pipeline Example.
 *
 * This example demonstrates the full SFT workflow in DL4J/ND4J, covering:
 *
 * 1.  Build base language model     - SameDiff model with weight matrices, softmax, loss
 * 2.  SFTConfig basics              - defaultSFT(), all fields, custom config with LoRA
 * 3.  SFTConfig presets             - loraDefaults(16), qloraDefaults()
 * 4.  Conversation data formatting  - InstructionDataFormatter with multi-turn conversations
 *                                     and character-level loss masks
 * 5.  Data quality filtering        - TextQualityFilter with custom thresholds
 * 6.  Data deduplication            - TextDeduplicator with exact and MinHash
 * 7.  Data splitting                - StratifiedSplitter 80/10/10
 * 8.  Length bucketing              - LengthBucketingIterator with token budget
 * 9.  Weighted data mixing          - WeightedDataMixer across 3 domains
 * 10. SFT Training Pipeline         - SFTTrainingPipeline.trainFromPairs()
 * 11. Export merged model           - mergeAndExport()
 * 12. Workflow summary              - full pipeline step-by-step recap
 * 13. DSP-Accelerated SFT Training  - DSP plan phase progression during SFT fit() loop
 *
 * SFT differs from continued pretraining in one critical way: loss is computed only
 * on assistant response tokens, not on system prompt or user turns. The
 * InstructionDataFormatter produces character-level loss masks (1=train, 0=ignore).
 * SFTTrainingPipeline converts those to token-level masks before feeding batches
 * to SameDiff.fit().
 *
 * @see SFTConfig
 * @see SFTTrainingPipeline
 * @see InstructionDataFormatter
 */
public class SFTTrainingPipelineExample {
    private static final Logger log = LoggerFactory.getLogger(SFTTrainingPipelineExample.class);

    // Toy model dimensions
    private static final int VOCAB_SIZE  = 128;
    private static final int EMBED_DIM   = 32;
    private static final int HIDDEN_DIM  = 64;

    public static void main(String[] args) {

        // =====================================================================
        // 1. Build base language model
        // =====================================================================
        log.info("=== 1. Build Base Language Model ===");

        // A minimal language model: placeholder -> embed projection -> hidden layer
        // -> softmax logits -> cross-entropy loss. In a real scenario this would be
        // a loaded transformer checkpoint; here we build from scratch to keep the
        // example self-contained.
        SameDiff baseModel = buildBaseLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM);

        // Summarise the variable count for transparency
        long numVariables = baseModel.variables().size();
        log.info("  Base model variables: {}", numVariables);
        log.info("  Placeholders: input_ids, labels");
        log.info("  Trainable weights: W_embed, W_hidden, b_hidden, W_out, b_out");
        log.info("  Loss variable: xent_loss (marked as loss)");

        // =====================================================================
        // 2. SFTConfig basics
        // =====================================================================
        log.info("=== 2. SFTConfig Basics ===");

        // Default: full fine-tune, CHATML template, lr=2e-5, 3 epochs, gradAccum=4
        SFTConfig defaultConfig = SFTConfig.defaultSFT();
        log.info("  defaultSFT():");
        log.info("    chatTemplate          = {}", defaultConfig.getChatTemplate());
        log.info("    maxSeqLength          = {}", defaultConfig.getMaxSeqLength());
        log.info("    learningRate          = {}", defaultConfig.getLearningRate());
        log.info("    minLearningRate       = {}", defaultConfig.getMinLearningRate());
        log.info("    warmupRatio           = {}", defaultConfig.getWarmupRatio());
        log.info("    weightDecay           = {}", defaultConfig.getWeightDecay());
        log.info("    maxGradNorm           = {}", defaultConfig.getMaxGradNorm());
        log.info("    numEpochs             = {}", defaultConfig.getNumEpochs());
        log.info("    gradientAccumulation  = {}", defaultConfig.getGradientAccumulationSteps());
        log.info("    computeDataType       = {}", defaultConfig.getComputeDataType());
        log.info("    packSequences         = {}", defaultConfig.isPackSequences());
        log.info("    peftConfig            = {}", defaultConfig.getPeftConfig());

        // Custom config: LLAMA3 template, custom learning rate, LoRA PEFT
        LoraConfig customLora = LoraConfig.builder()
                .r(16)
                .loraAlpha(32)
                .loraDropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .taskType(TaskType.CAUSAL_LM)
                .build();

        SFTConfig customConfig = SFTConfig.builder()
                .chatTemplate(ChatTemplate.LLAMA3)
                .systemMessage("You are a precise and helpful coding assistant.")
                .maxSeqLength(4096)
                .learningRate(1e-4)
                .minLearningRate(1e-6)
                .warmupRatio(0.05)
                .weightDecay(0.01)
                .maxGradNorm(1.0)
                .numEpochs(5)
                .gradientAccumulationSteps(8)
                .computeDataType(DataType.BFLOAT16)
                .peftConfig(customLora)
                .build();

        customConfig.validate();
        log.info("  customConfig:");
        log.info("    chatTemplate          = {}", customConfig.getChatTemplate());
        log.info("    systemMessage         = '{}'", customConfig.getSystemMessage());
        log.info("    learningRate          = {}", customConfig.getLearningRate());
        log.info("    warmupRatio           = {}", customConfig.getWarmupRatio());
        log.info("    gradientAccumulation  = {}", customConfig.getGradientAccumulationSteps());
        log.info("    PEFT type             = {}", customConfig.getPeftConfig().getPeftType());
        log.info("    LoRA rank             = {}", customLora.getR());
        log.info("    LoRA scaling          = {}", customLora.getScaling());
        log.info("    LoRA target modules   = {}", customLora.getTargetModules());

        // =====================================================================
        // 3. SFTConfig presets
        // =====================================================================
        log.info("=== 3. SFTConfig Presets ===");

        // Full fine-tune preset
        SFTConfig fullFT = SFTConfig.defaultSFT();
        log.info("  SFTConfig.defaultSFT() -> peft={}", fullFT.getPeftConfig());

        // LoRA preset with rank=16 (suitable for 7B-13B models on a single GPU)
        SFTConfig loraSFT = SFTConfig.loraDefaults(16);
        loraSFT.validate();
        LoraConfig loraPreset = (LoraConfig) loraSFT.getPeftConfig();
        log.info("  SFTConfig.loraDefaults(16):");
        log.info("    peft type    = {}", loraPreset.getPeftType());
        log.info("    rank         = {}", loraPreset.getR());
        log.info("    alpha        = {}", loraPreset.getLoraAlpha());
        log.info("    dropout      = {}", loraPreset.getLoraDropout());
        log.info("    targets      = {}", loraPreset.getTargetModules());

        // QLoRA preset (4-bit quantization + LoRA, for consumer hardware)
        SFTConfig qloraSFT = SFTConfig.qloraDefaults();
        qloraSFT.validate();
        QLoraConfig qloraPreset = (QLoraConfig) qloraSFT.getPeftConfig();
        log.info("  SFTConfig.qloraDefaults():");
        log.info("    peft type              = {}", qloraPreset.getPeftType());
        log.info("    rank                   = {}", qloraPreset.getR());
        log.info("    alpha                  = {}", qloraPreset.getLoraAlpha());
        log.info("    gradientAccumulation   = {}", qloraSFT.getGradientAccumulationSteps());

        // =====================================================================
        // 4. Conversation data formatting with loss masks
        // =====================================================================
        log.info("=== 4. Conversation Data Formatting ===");

        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        // Build a representative multi-turn SFT dataset.
        // System turns and user turns get loss mask=0 (ignored during training).
        // Assistant turns get loss mask=1 (trained on).
        List<List<ConversationTurn>> conversations = buildSampleConversations();
        log.info("  Formatting {} conversations with CHATML template", conversations.size());

        for (int i = 0; i < conversations.size(); i++) {
            List<ConversationTurn> conv = conversations.get(i);
            FormattedExample example = formatter.format(conv);

            // Analyse the loss mask
            int[] mask = example.getLossMask();
            int trainableChars = countOnes(mask);
            int totalChars     = mask.length;
            double trainRatio  = (double) trainableChars / totalChars * 100;

            log.info("  Conversation {}: {} chars total, {} trainable ({}%)",
                    i + 1, totalChars, trainableChars, String.format("%.1f", trainRatio));

            // Print the first conversation's formatted text in full to show the template
            if (i == 0) {
                log.info("  Formatted text (conversation 1):\n{}", example.getText());
                log.info("  Loss mask preview (first 80 positions): {}",
                        maskPreview(mask, 80));
                log.info("  (1=assistant content trained on, 0=system/user ignored)");
            }
        }

        // Demonstrate token-level mask conversion (char mask -> token mask)
        FormattedExample firstEx = formatter.format(conversations.get(0));
        int seqLen     = firstEx.getText().length() / 4; // rough token count
        int[] tokenMask = SFTTrainingPipeline.convertCharMaskToTokenMask(
                firstEx.getLossMask(), seqLen);
        int trainableTokens = countOnes(tokenMask);
        log.info("  Token mask: seqLen={}, trainable tokens={}/{}", seqLen, trainableTokens, seqLen);

        // =====================================================================
        // 5. Data quality filtering
        // =====================================================================
        log.info("=== 5. Data Quality Filtering ===");

        TextQualityFilter qualityFilter = TextQualityFilter.builder()
                .minChars(40)
                .maxChars(50_000)
                .minWords(8)
                .minAlphaRatio(0.50)
                .maxSpecialCharRatio(0.20)
                .maxRepetitionRatio(0.35)
                .maxAllCapsRatio(0.40)
                .maxWhitespaceRatio(0.35)
                .build();

        List<String> candidateTexts = buildCandidateTexts();
        log.info("  Evaluating {} candidate texts:", candidateTexts.size());

        List<String> accepted = new ArrayList<>();
        List<String> rejected = new ArrayList<>();

        for (String text : candidateTexts) {
            FilterResult result = qualityFilter.evaluate(text);
            String preview = text.length() > 55 ? text.substring(0, 55) + "..." : text;
            if (result.isAccepted()) {
                accepted.add(text);
                log.info("  ACCEPT  \"{}\"", preview);
            } else {
                rejected.add(text);
                log.info("  REJECT  \"{}\"", preview);
                log.info("          reasons: {}", result.getRejectionReasons());
            }
        }
        log.info("  Result: {}/{} texts accepted", accepted.size(), candidateTexts.size());

        // =====================================================================
        // 6. Data deduplication
        // =====================================================================
        log.info("=== 6. Data Deduplication ===");

        // Exact deduplication using SHA-256 (case- and whitespace-normalised)
        TextDeduplicator exactDedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(false)
                .build();

        List<String> withExactDups = buildTextsWithDuplicates();
        List<String> afterExactDedup = exactDedup.deduplicate(withExactDups);
        log.info("  Exact dedup: {} -> {} texts", withExactDups.size(), afterExactDedup.size());
        for (int i = 0; i < afterExactDedup.size(); i++) {
            log.info("    Kept [{}]: \"{}\"", i + 1, afterExactDedup.get(i));
        }

        // Near-duplicate detection using MinHash LSH (Jaccard similarity)
        TextDeduplicator minHashDedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(true)
                .shingleSize(4)
                .numBands(16)
                .rowsPerBand(8)
                .jaccardThreshold(0.75)
                .seed(42L)
                .build();

        List<String> withNearDups = buildTextsWithNearDuplicates();
        log.info("  Near-dedup input ({} texts):", withNearDups.size());
        for (String t : withNearDups) {
            log.info("    - \"{}\"", t);
        }

        // Streaming mode: process one document at a time
        minHashDedup.reset();
        List<String> uniqueTexts = new ArrayList<>();
        for (String doc : withNearDups) {
            if (minHashDedup.addIfUnique(doc)) {
                uniqueTexts.add(doc);
            }
        }
        log.info("  Near-dedup (Jaccard>=0.75): {} -> {} unique texts",
                withNearDups.size(), uniqueTexts.size());
        for (String t : uniqueTexts) {
            log.info("    UNIQUE: \"{}\"", t);
        }

        // =====================================================================
        // 7. Data splitting
        // =====================================================================
        log.info("=== 7. Data Splitting (80/10/10) ===");

        List<String> allAccepted = buildLargerDataset(60); // 60 examples for clear split demo

        StratifiedSplitter<String> splitter = new StratifiedSplitter<>(42L);

        // Random split 80 / 10 / 10
        SplitResult<String> randomSplit = splitter.split(allAccepted, 0.80, 0.10);
        log.info("  Random split from {} examples:", allAccepted.size());
        log.info("    train : {} ({}%)",
                randomSplit.getTrain().size(),
                String.format("%.1f", 100.0 * randomSplit.getTrain().size() / allAccepted.size()));
        log.info("    val   : {} ({}%)",
                randomSplit.getValidation().size(),
                String.format("%.1f", 100.0 * randomSplit.getValidation().size() / allAccepted.size()));
        log.info("    test  : {} ({}%)",
                randomSplit.getTest().size(),
                String.format("%.1f", 100.0 * randomSplit.getTest().size() / allAccepted.size()));

        // Stratified split by domain label (preserves domain proportions)
        List<LabeledText> labeledData = buildLabeledDataset();
        StratifiedSplitter<LabeledText> stratSplitter = new StratifiedSplitter<>(42L);
        SplitResult<LabeledText> stratSplit = stratSplitter.splitStratified(
                labeledData, 0.80, 0.10, item -> item.domain);

        log.info("  Stratified split from {} labeled examples (3 domains):", labeledData.size());
        log.info("    train : {}", stratSplit.getTrain().size());
        log.info("    val   : {}", stratSplit.getValidation().size());
        log.info("    test  : {}", stratSplit.getTest().size());

        // Verify all domains appear in train split
        java.util.Set<String> trainDomains = new java.util.HashSet<>();
        for (LabeledText item : stratSplit.getTrain()) trainDomains.add(item.domain);
        log.info("    domains in train: {}", trainDomains);

        // =====================================================================
        // 8. Length bucketing
        // =====================================================================
        log.info("=== 8. Length Bucketing Iterator ===");

        List<String> bucketingData = buildVariableLengthTexts(80);

        // Fixed batch size: group by token length, 8 sequences per batch
        LengthBucketingIterator<String> fixedBatchIter = LengthBucketingIterator.<String>builder()
                .bucketBoundaries(64, 128, 256, 512, 1024)
                .fixedBatchSize(8)
                .lengthFunction(text -> text.length() / 4)  // chars/4 -> approx tokens
                .shuffle(42L)
                .build(bucketingData);

        int fixedBatches = 0;
        int totalSequences = 0;
        while (fixedBatchIter.hasNext()) {
            List<String> batch = fixedBatchIter.next();
            fixedBatches++;
            totalSequences += batch.size();
        }
        log.info("  Fixed-batch iterator (batch=8): {} batches, {} sequences total",
                fixedBatches, totalSequences);

        // Token budget: limit total tokens per batch (shorter seqs -> larger batch)
        LengthBucketingIterator<String> tokenBudgetIter = LengthBucketingIterator.<String>builder()
                .bucketBoundaries(64, 128, 256, 512, 1024)
                .tokenBudget(2048)          // at most 2048 tokens per batch
                .lengthFunction(text -> text.length() / 4)
                .shuffle(42L)
                .build(bucketingData);

        int budgetBatches = 0;
        while (tokenBudgetIter.hasNext()) {
            List<String> batch = tokenBudgetIter.next();
            int batchTokens = batch.stream().mapToInt(t -> t.length() / 4).sum();
            log.info("  Token-budget batch {}: {} seqs, ~{} tokens",
                    budgetBatches + 1, batch.size(), batchTokens);
            budgetBatches++;
        }
        log.info("  Token-budget iterator (budget=2048): {} batches from {} sequences",
                budgetBatches, bucketingData.size());

        // =====================================================================
        // 9. Weighted data mixing
        // =====================================================================
        log.info("=== 9. Weighted Data Mixing ===");

        // Three domains with different desired proportions.
        // WeightedDataMixer samples proportionally to the specified weights.
        List<String> codeDomain     = buildDomainData("code",     40);
        List<String> mathDomain     = buildDomainData("math",     30);
        List<String> reasonDomain   = buildDomainData("reasoning", 20);

        Map<String, WeightedDataMixer.WeightedSource<String>> sources = new LinkedHashMap<>();
        sources.put("code",      new WeightedDataMixer.WeightedSource<>(codeDomain.iterator(),   0.50));
        sources.put("math",      new WeightedDataMixer.WeightedSource<>(mathDomain.iterator(),   0.30));
        sources.put("reasoning", new WeightedDataMixer.WeightedSource<>(reasonDomain.iterator(), 0.20));

        // temperature=1.0 preserves exact weights; < 1 makes mixing more uniform;
        // > 1 concentrates sampling on the largest domain
        WeightedDataMixer<String> mixer = new WeightedDataMixer<>(sources, 1.0, 42L);

        int sampleCount = 0;
        while (mixer.hasNext() && sampleCount < 50) {
            mixer.next();
            sampleCount++;
        }
        Map<String, Long> mixStats = mixer.getStats();
        log.info("  After {} samples (target: code=50%, math=30%, reasoning=20%):", sampleCount);
        log.info("    code drawn      : {} ({}%)",
                mixStats.get("code"),
                String.format("%.1f", 100.0 * mixStats.get("code") / sampleCount));
        log.info("    math drawn      : {} ({}%)",
                mixStats.get("math"),
                String.format("%.1f", 100.0 * mixStats.get("math") / sampleCount));
        log.info("    reasoning drawn : {} ({}%)",
                mixStats.get("reasoning"),
                String.format("%.1f", 100.0 * mixStats.get("reasoning") / sampleCount));

        // Temperature smoothing example
        List<String> codeT   = buildDomainData("code",     40);
        List<String> mathT   = buildDomainData("math",     30);
        List<String> reasonT = buildDomainData("reasoning", 20);
        Map<String, WeightedDataMixer.WeightedSource<String>> sourcesT = new LinkedHashMap<>();
        sourcesT.put("code",      new WeightedDataMixer.WeightedSource<>(codeT.iterator(),   0.50));
        sourcesT.put("math",      new WeightedDataMixer.WeightedSource<>(mathT.iterator(),   0.30));
        sourcesT.put("reasoning", new WeightedDataMixer.WeightedSource<>(reasonT.iterator(), 0.20));
        WeightedDataMixer<String> mixerLowTemp = new WeightedDataMixer<>(sourcesT, 0.5, 42L);
        int sampledLow = 0;
        while (mixerLowTemp.hasNext() && sampledLow < 50) { mixerLowTemp.next(); sampledLow++; }
        Map<String, Long> lowTStats = mixerLowTemp.getStats();
        log.info("  temperature=0.5 (more uniform mixing after {} samples):", sampledLow);
        log.info("    code drawn      : {} ({}%)",
                lowTStats.get("code"),
                String.format("%.1f", 100.0 * lowTStats.get("code") / sampledLow));
        log.info("    math drawn      : {} ({}%)",
                lowTStats.get("math"),
                String.format("%.1f", 100.0 * lowTStats.get("math") / sampledLow));
        log.info("    reasoning drawn : {} ({}%)",
                lowTStats.get("reasoning"),
                String.format("%.1f", 100.0 * lowTStats.get("reasoning") / sampledLow));

        // =====================================================================
        // 10. SFT Training Pipeline
        // =====================================================================
        log.info("=== 10. SFT Training Pipeline ===");

        // Build a fresh model for training. In practice you would load a pre-trained
        // checkpoint here. We rebuild the toy model so the example is self-contained.
        SameDiff trainModel = buildBaseLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM);

        // Use full fine-tune with minimal config for a fast toy-model demonstration.
        // Change to SFTConfig.loraDefaults(16) for LoRA on a real transformer.
        SFTConfig sftConfig = SFTConfig.builder()
                .chatTemplate(ChatTemplate.CHATML)
                .systemMessage("You are a helpful assistant.")
                .maxSeqLength(512)
                .learningRate(2e-5)
                .minLearningRate(0.0)
                .warmupRatio(0.03)
                .numEpochs(1)          // 1 epoch for example speed; use 3-5 in production
                .gradientAccumulationSteps(2)
                .computeDataType(DataType.FLOAT)  // FLOAT for CPU toy demo; BFLOAT16 for GPU
                .build();

        SFTTrainingPipeline pipeline = new SFTTrainingPipeline(trainModel, sftConfig);
        log.info("  Pipeline constructed (peft={})", pipeline.getPeftModel() != null ? "LoRA" : "full FT");

        // Show the TrainingConfig that the pipeline would build for 100 steps
        TrainingConfig trainingConfig = pipeline.buildTrainingConfig(100);
        log.info("  buildTrainingConfig(100):");
        log.info("    updater              = {}", trainingConfig.getUpdater().getClass().getSimpleName());
        log.info("    mixed precision      = {}", trainingConfig.isMixedPrecision());
        log.info("    gradient accumulation= {}", trainingConfig.isGradientAccumulationEnabled());

        // trainFromPairs: provide raw (instruction, response) string pairs.
        // The pipeline formats them using the configured chat template, builds
        // MultiDataSet batches with token-level loss masks, and calls model.fit().
        List<Map.Entry<String, String>> pairs = buildInstructionPairs();
        log.info("  Calling trainFromPairs() with {} instruction/response pairs...", pairs.size());

        pipeline.trainFromPairs(pairs);
        log.info("  trainFromPairs() completed.");

        // Alternatively: provide pre-built conversation lists
        List<List<ConversationTurn>> trainConversations = buildSampleConversations();
        log.info("  Calling train(conversations) with {} conversations...", trainConversations.size());
        pipeline.train(trainConversations);
        log.info("  train(conversations) completed.");

        // =====================================================================
        // 11. Export merged model
        // =====================================================================
        log.info("=== 11. Export Merged Model ===");

        // When PEFT was configured, mergeAndExport() folds the LoRA adapter weights
        // back into the base weight matrices and returns a standalone SameDiff model
        // that can be used for inference without any adapter overhead.
        // When no PEFT was configured it returns the trained model unchanged.
        SameDiff mergedModel = pipeline.mergeAndExport();
        log.info("  mergeAndExport() returned model with {} variables",
                mergedModel.variables().size());

        // In a real pipeline you would save to disk:
        //   mergedModel.save(new File("merged_sft_model.bin"), true);
        log.info("  (In production: mergedModel.save(new File(\"merged_sft.bin\"), true))");

        // =====================================================================
        // 12. Complete workflow summary
        // =====================================================================
        log.info("=== 12. Complete SFT Workflow Summary ===");
        log.info("  Step  1  Load/build base model (SameDiff with loss variable)");
        log.info("  Step  2  Collect raw instruction/response data and conversations");
        log.info("  Step  3  TextQualityFilter -> reject too-short, low-alpha, spam, repetitive");
        log.info("  Step  4  TextDeduplicator  -> exact SHA-256 + MinHash near-dedup (Jaccard)");
        log.info("  Step  5  InstructionDataFormatter -> chat template + char-level loss mask");
        log.info("  Step  6  StratifiedSplitter -> 80% train / 10% val / 10% test");
        log.info("  Step  7  LengthBucketingIterator -> group by length, minimize padding");
        log.info("  Step  8  WeightedDataMixer -> sample proportionally across domains");
        log.info("  Step  9  SFTConfig -> choose full FT / LoRA / QLoRA + hyperparameters");
        log.info("  Step 10  SFTTrainingPipeline -> trainFromPairs() or train(conversations)");
        log.info("  Step 11  pipeline.mergeAndExport() -> adapter-free inference model");
        log.info("  Step 12  model.save() -> checkpoint for deployment");

        // =====================================================================
        // 13. DSP-Accelerated SFT Training
        // =====================================================================
        log.info("=== 13. DSP-Accelerated SFT Training ===");

        // Build a fresh model — same architecture as section 1.
        SameDiff dspSftModel = buildBaseLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM);

        // Enable DSP explicitly (both default to true; shown here for documentation).
        dspSftModel.setDspAutoCompileEnabled(true);
        dspSftModel.setDspNativeAutoCompileEnabled(true);
        log.info("  dspAutoCompileEnabled:       {}", dspSftModel.isDspAutoCompileEnabled());
        log.info("  dspNativeAutoCompileEnabled: {}", dspSftModel.isDspNativeAutoCompileEnabled());

        // TrainingConfig with Adam(2e-5) — typical SFT learning rate.
        // Map DataSet features -> "input_ids" and labels -> "labels" to match
        // the placeholder names used by buildBaseLanguageModel().
        TrainingConfig dspTrainingConfig = TrainingConfig.builder()
                .updater(new org.nd4j.linalg.learning.config.Adam(2e-5))
                .dataSetFeatureMapping("input_ids")
                .dataSetLabelMapping("labels")
                .build();
        dspSftModel.setTrainingConfig(dspTrainingConfig);
        log.info("  TrainingConfig: Adam(lr=2e-5), features->input_ids, labels->labels");

        // Fixed-batch synthetic DataSet: batch=4, seqLen=8.
        // input_ids: LONG [4, 8], labels: LONG [4, 8]
        int dspBatch  = 4;
        int dspSeqLen = 8;
        INDArray dspInputIds = Nd4j.ones(DataType.LONG, dspBatch, dspSeqLen);
        INDArray dspLabels   = Nd4j.ones(DataType.LONG, dspBatch, dspSeqLen);
        DataSet dspDs = new DataSet(dspInputIds, dspLabels);

        // 10 training steps — observe DSP plan phase progression.
        int dspSteps = 10;
        log.info("  Running {} SFT training steps with DSP...", dspSteps);

        for (int step = 0; step < dspSteps; step++) {
            long t0 = System.nanoTime();
            dspSftModel.fit(dspDs);
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

            DspHandle dspH = dspSftModel.dsp();
            if (dspH.isCompiled()) {
                int phaseCode = dspH.planPhase();
                PlanPhase phase = PlanPhase.fromNativeCode(phaseCode);
                String phaseName = phase != null ? phase.name() : "UNKNOWN(" + phaseCode + ")";
                int segsReplayed    = dspH.lastExecSegmentsReplayed();
                int segsSlotBySlot  = dspH.lastExecSegmentsSlotBySlot();
                int segsTotal       = dspH.lastExecSegmentsTotal();
                log.info("  Step {}: {}ms  phase={}  segs[replay={}/sbs={}/total={}]",
                        String.format("%2d", step), String.format("%4d", elapsedMs), phaseName,
                        segsReplayed, segsSlotBySlot, segsTotal);
            } else {
                log.info("  Step {}: {}ms  (plan not yet compiled)",
                        String.format("%2d", step), String.format("%4d", elapsedMs));
            }
        }

        // DspHandle summary after the training loop.
        DspHandle dspFinal = dspSftModel.dsp();
        if (dspFinal.isCompiled()) {
            log.info("  DspHandle summary after {} SFT steps:", dspSteps);
            log.info("    totalSlots              = {}", dspFinal.totalSlots());
            log.info("    numSegments             = {}", dspFinal.numSegments());
            log.info("    numCapturedGraphSegments= {}", dspFinal.numCapturedGraphSegments());
            log.info("    totalGraphReplays       = {}", dspFinal.totalGraphReplays());
            log.info("    planPhase               = {}", PlanPhase.fromNativeCode(dspFinal.planPhase()));
            log.info("    pointersStable          = {}", dspFinal.pointersStable());
            log.info("    compilationSealed       = {}", dspFinal.isCompilationSealed());
        } else {
            log.info("  Plan not compiled after {} steps (no DSP executor on this backend/config).", dspSteps);
        }

        log.info("  Key insight: DSP compiles the full SFT training graph (forward + loss"
                + " + backward + updater) into a flat-slot plan");

        log.info("**************** SFT Training Pipeline Example finished ********************");
    }

    // =========================================================================
    // Model builder
    // =========================================================================

    /**
     * Build a minimal language model graph in SameDiff.
     *
     * Architecture:
     *   input_ids [batch, seqLen] (INT64)
     *   -> one-hot encode -> W_embed [vocabSize, embedDim]
     *   -> W_hidden [embedDim, hiddenDim] + b_hidden -> relu
     *   -> W_out [hiddenDim, vocabSize] + b_out -> softmax logits
     *   -> cross-entropy loss with labels [batch, seqLen]
     *
     * In a real SFT pipeline the model would be loaded from a checkpoint and
     * may be a transformer. The graph structure here is intentionally simple
     * so the example executes quickly on CPU.
     */
    private static SameDiff buildBaseLanguageModel(int vocabSize, int embedDim, int hiddenDim) {
        SameDiff sd = SameDiff.create();

        // Placeholders for tokenised input and next-token prediction targets
        SDVariable inputIds = sd.placeHolder("input_ids",  DataType.LONG,  -1, -1); // [batch, seqLen]
        SDVariable labels   = sd.placeHolder("labels",     DataType.LONG,  -1, -1); // [batch, seqLen]

        // Embedding lookup projection (one-hot matmul is equivalent to embedding table lookup)
        SDVariable wEmbed  = sd.var("W_embed",  Nd4j.rand(DataType.FLOAT, vocabSize, embedDim).muli(0.02));
        SDVariable wHidden = sd.var("W_hidden", Nd4j.rand(DataType.FLOAT, embedDim,  hiddenDim).muli(0.02));
        SDVariable bHidden = sd.var("b_hidden", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable wOut    = sd.var("W_out",    Nd4j.rand(DataType.FLOAT, hiddenDim,  vocabSize).muli(0.02));
        SDVariable bOut    = sd.var("b_out",    Nd4j.zeros(DataType.FLOAT, vocabSize));

        // Cast input_ids to float for one-hot, then embed
        // one_hot: [batch, seqLen, vocabSize]
        SDVariable onHot       = sd.oneHot("one_hot", inputIds, vocabSize, -1, 1.0, 0.0, DataType.FLOAT);
        // Reshape to [batch*seqLen, vocabSize] for matmul
        SDVariable batchSeq    = sd.reshape(onHot, new long[]{-1, vocabSize});
        SDVariable embedded    = sd.mmul(batchSeq, wEmbed);                        // [batch*seqLen, embedDim]
        SDVariable hidden      = sd.nn.relu(embedded.mmul(wHidden).add(bHidden), 0); // [batch*seqLen, hiddenDim]
        SDVariable logits      = hidden.mmul(wOut).add(bOut);                       // [batch*seqLen, vocabSize]
        logits.rename("logits");

        // Softmax cross-entropy loss against flattened labels
        SDVariable labelsFlat  = sd.reshape(labels, new long[]{-1});               // [batch*seqLen]
        SDVariable xentLoss    = sd.loss.sparseSoftmaxCrossEntropy("xent_loss", logits, labelsFlat);
        SDVariable meanLoss    = sd.mean("loss", xentLoss);
        meanLoss.markAsLoss();

        return sd;
    }

    // =========================================================================
    // Sample data builders
    // =========================================================================

    /**
     * Build a realistic set of multi-turn conversations covering three domains:
     * coding, scientific explanation, and general Q&A.
     */
    private static List<List<ConversationTurn>> buildSampleConversations() {
        List<List<ConversationTurn>> conversations = new ArrayList<>();

        // Conversation 1: multi-turn coding assistant (Python)
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are an expert Python programming assistant."),
            ConversationTurn.user("How do I sort a list of dictionaries by a specific key in Python?"),
            ConversationTurn.assistant(
                "Use the sorted() function with a key argument:\n\n" +
                "    records = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\n" +
                "    sorted_records = sorted(records, key=lambda x: x['age'])\n\n" +
                "For in-place sorting use list.sort(key=...). Add reverse=True for descending order."),
            ConversationTurn.user("How do I handle missing keys safely?"),
            ConversationTurn.assistant(
                "Use dict.get() with a default value in the key function:\n\n" +
                "    sorted_records = sorted(records, key=lambda x: x.get('age', 0))\n\n" +
                "This returns 0 when 'age' is absent, preventing a KeyError.")
        ));

        // Conversation 2: scientific explanation
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are a knowledgeable science educator."),
            ConversationTurn.user("Can you explain how CRISPR-Cas9 gene editing works?"),
            ConversationTurn.assistant(
                "CRISPR-Cas9 is a molecular scissors system derived from bacterial immune defence. " +
                "A guide RNA (gRNA) is engineered to match the target DNA sequence. The Cas9 protein " +
                "binds to the gRNA and scans the genome until it finds the complementary sequence. " +
                "Once located, Cas9 makes a double-strand break in the DNA. The cell's natural repair " +
                "machinery then either disrupts the gene (via NHEJ) or incorporates a new DNA template " +
                "(via HDR), allowing precise gene knockout or correction."),
            ConversationTurn.user("What are the main off-target concerns?"),
            ConversationTurn.assistant(
                "Off-target edits occur when the guide RNA binds to sequences similar but not identical " +
                "to the target. Key concerns include: (1) unintended cuts causing mutations or genomic " +
                "rearrangements; (2) mosaicism in embryo editing where only some cells are corrected; " +
                "(3) immunogenicity from bacterial Cas9 proteins. High-fidelity Cas9 variants (eSpCas9, " +
                "HiFi Cas9) and truncated guide RNAs reduce off-target rates significantly.")
        ));

        // Conversation 3: general knowledge
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are a helpful and accurate general-knowledge assistant."),
            ConversationTurn.user("What caused the 2008 financial crisis?"),
            ConversationTurn.assistant(
                "The 2008 global financial crisis was triggered by the collapse of the US housing " +
                "bubble. Key factors: (1) widespread issuance of subprime mortgages to borrowers " +
                "with poor creditworthiness; (2) repackaging of these mortgages into complex " +
                "derivatives (CDOs) rated AAA by credit agencies; (3) excessive leverage at major " +
                "investment banks; (4) inadequate regulatory oversight of shadow banking. When " +
                "housing prices fell, default rates spiked, CDOs lost value rapidly, and " +
                "inter-bank lending froze, triggering a systemic banking crisis.")
        ));

        // Conversation 4: coding – Java
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are a senior Java developer."),
            ConversationTurn.user("What is the difference between HashMap and ConcurrentHashMap?"),
            ConversationTurn.assistant(
                "HashMap is not thread-safe. Concurrent modification from multiple threads can cause " +
                "data corruption or infinite loops during rehashing. ConcurrentHashMap (Java 5+) " +
                "provides thread-safe operations using segment-level locking (Java 7) or compare-and-swap " +
                "with volatile reads (Java 8+). ConcurrentHashMap allows concurrent reads without locks " +
                "and partitions writes, giving far better throughput than a fully synchronised " +
                "Collections.synchronizedMap(). Use ConcurrentHashMap whenever the map is shared " +
                "across threads; prefer HashMap in single-threaded contexts for lower overhead.")
        ));

        // Conversation 5: mathematics explanation
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are a patient and rigorous mathematics tutor."),
            ConversationTurn.user("Explain the intuition behind gradient descent."),
            ConversationTurn.assistant(
                "Gradient descent minimises a function by iteratively stepping in the direction of " +
                "steepest descent. Imagine standing on a hilly landscape with no visibility: the " +
                "gradient tells you which direction is currently steepest uphill, so you take a " +
                "small step in the opposite direction. Formally, for parameters θ and loss L: " +
                "θ_{t+1} = θ_t - η ∇L(θ_t), where η is the learning rate. With a suitable η " +
                "this converges to a local (or global, for convex L) minimum. Stochastic gradient " +
                "descent approximates ∇L using a random mini-batch, which adds noise but dramatically " +
                "reduces compute per step and often escapes sharp minima.")
        ));

        // Conversation 6: data engineering
        conversations.add(Arrays.asList(
            ConversationTurn.system("You are a data engineering expert."),
            ConversationTurn.user("When should I use Apache Kafka over a traditional message queue?"),
            ConversationTurn.assistant(
                "Choose Kafka when you need: (1) high-throughput durable log storage (millions of " +
                "messages/sec); (2) multiple independent consumers reading the same stream at their " +
                "own pace; (3) replay of past events by rewinding offsets; (4) long retention periods " +
                "(days to forever). Traditional queues like RabbitMQ or ActiveMQ are better for: " +
                "complex routing topologies, per-message acknowledgement with retry semantics, " +
                "low-latency task queuing, and cases where messages should be deleted after " +
                "consumption. The fundamental difference is that Kafka is a distributed commit log " +
                "while a queue is a transient message buffer.")
        ));

        return conversations;
    }

    /**
     * Build a list of instruction/response pairs for trainFromPairs().
     * Each entry is a simple (instruction, response) tuple covering diverse topics.
     */
    private static List<Map.Entry<String, String>> buildInstructionPairs() {
        List<Map.Entry<String, String>> pairs = new ArrayList<>();
        pairs.add(new AbstractMap.SimpleEntry<>(
            "Summarise the key benefits of transfer learning in NLP.",
            "Transfer learning allows NLP models to leverage representations learned on large corpora " +
            "and adapt them to downstream tasks with far less labelled data. Key benefits: reduced " +
            "training time, better generalisation on small datasets, and access to rich linguistic " +
            "knowledge (syntax, semantics, world facts) captured during pre-training."));
        pairs.add(new AbstractMap.SimpleEntry<>(
            "What is the vanishing gradient problem and how is it addressed?",
            "During backpropagation in deep networks, gradients are multiplied layer by layer. " +
            "When activation derivatives are < 1 (e.g. sigmoid, tanh), gradients shrink " +
            "exponentially towards the input layers, making early layers train very slowly. " +
            "Solutions include ReLU activations (derivative=1 for positive inputs), residual " +
            "connections (skip connections let gradients flow directly), batch/layer normalisation, " +
            "and careful weight initialisation (He, Xavier)."));
        pairs.add(new AbstractMap.SimpleEntry<>(
            "Describe the attention mechanism used in transformer models.",
            "Attention computes a weighted sum of value vectors, where weights are determined by " +
            "the compatibility between a query and a set of keys. In scaled dot-product attention: " +
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. Multi-head attention applies this " +
            "h times in parallel with learned projections, allowing the model to attend to different " +
            "representation subspaces simultaneously. This replaces recurrence, enabling full " +
            "parallelism during training."));
        pairs.add(new AbstractMap.SimpleEntry<>(
            "How does dropout regularisation work?",
            "During training, dropout randomly sets neuron activations to zero with probability p " +
            "(typically 0.1-0.5). This forces the network to learn redundant representations and " +
            "prevents co-adaptation of neurons, acting as implicit ensemble averaging. At inference " +
            "time all neurons are active but outputs are scaled by (1-p) to maintain expected " +
            "activation magnitude. Dropout is most effective in large networks and less useful " +
            "in convolutional or batch-normalised architectures."));
        pairs.add(new AbstractMap.SimpleEntry<>(
            "Explain the difference between precision and recall.",
            "Precision = TP / (TP + FP): of all predicted positives, what fraction are actually " +
            "positive. Recall = TP / (TP + FN): of all actual positives, what fraction were " +
            "detected. High precision means few false alarms; high recall means few misses. " +
            "The F1 score (harmonic mean) balances both. Choose high precision when false positives " +
            "are costly (e.g. spam filters); high recall when false negatives are costly (e.g. cancer " +
            "screening)."));
        pairs.add(new AbstractMap.SimpleEntry<>(
            "What is LoRA and why is it useful for fine-tuning large language models?",
            "LoRA (Low-Rank Adaptation) freezes the pre-trained weight matrices and injects trainable " +
            "rank-decomposition matrices alongside them: W' = W + BA, where B and A have rank r << d. " +
            "This reduces trainable parameters by 10,000x for a 175B model while achieving performance " +
            "comparable to full fine-tuning. Practically this means fine-tuning fits on a single GPU, " +
            "adapter weights can be merged at inference for zero overhead, and multiple task-specific " +
            "adapters can be swapped on the same frozen backbone."));
        return pairs;
    }

    /** Build sample texts for quality filtering. */
    private static List<String> buildCandidateTexts() {
        return Arrays.asList(
            // ACCEPT: well-formed, informative
            "Neural networks learn hierarchical representations of data, transforming raw inputs " +
            "through successive layers into increasingly abstract features useful for the task.",
            // ACCEPT: good length, clear content
            "The transformer architecture replaced recurrent networks in most NLP tasks because " +
            "self-attention allows all positions to interact directly, enabling full parallelism " +
            "during training and capturing long-range dependencies more effectively.",
            // REJECT: too short (< 40 chars and < 8 words)
            "hi there",
            // REJECT: all caps / spam
            "BUY NOW!!! AMAZING DEAL CLICK HERE FREE OFFER!!!",
            // REJECT: high repetition
            "the model trains the model trains the model trains the model trains the model trains",
            // ACCEPT: technical content, good length
            "Gradient checkpointing trades compute for memory by recomputing intermediate " +
            "activations during the backward pass rather than storing them, enabling training " +
            "of larger models within a fixed GPU memory budget.",
            // ACCEPT: clear explanation
            "Knowledge distillation compresses a large teacher model into a smaller student by " +
            "training the student to match the teacher's soft probability outputs, which carry " +
            "richer supervision than hard one-hot labels.",
            // REJECT: mostly special characters, low alpha ratio
            "=== ### @@@ !!! ??? &&& %%% $$$ ^^^ *** ~~~ ::: --- +++ /// \\\\",
            // ACCEPT: good quality prose
            "Sparse mixture-of-experts models conditionally activate a small subset of parameters " +
            "for each token, scaling model capacity without proportionally increasing compute per " +
            "forward pass. Switch Transformer and GLaM demonstrated this approach at trillion-parameter scale.",
            // ACCEPT: data science topic
            "Cross-validation estimates model generalisation by partitioning data into k folds, " +
            "training on k-1 folds and evaluating on the held-out fold, then averaging results " +
            "across all k iterations to reduce variance in the performance estimate."
        );
    }

    /** Build texts with exact duplicates for dedup demo. */
    private static List<String> buildTextsWithDuplicates() {
        return Arrays.asList(
            "Attention is all you need: the transformer architecture relies solely on attention mechanisms.",
            "Convolutional neural networks are the backbone of modern computer vision systems.",
            "Attention is all you need: the transformer architecture relies solely on attention mechanisms.", // exact dup
            "Recurrent neural networks process sequences by maintaining a hidden state over time.",
            "  Attention is all you need: the transformer architecture relies solely on attention mechanisms.  " // whitespace dup
        );
    }

    /** Build texts with near-duplicates for MinHash demo. */
    private static List<String> buildTextsWithNearDuplicates() {
        return Arrays.asList(
            // Pair A: highly similar (>75% Jaccard)
            "Deep learning models learn hierarchical representations by stacking multiple processing layers.",
            "Deep learning models learn hierarchical feature representations by stacking multiple processing layers.",
            // Unique
            "Reinforcement learning trains agents to maximise cumulative reward through environment interaction.",
            // Pair B: similar phrasing
            "Batch normalisation accelerates training by normalising layer inputs across a mini-batch.",
            "Batch normalisation speeds up training by standardising the layer inputs within each mini-batch.",
            // Unique
            "Graph neural networks extend deep learning to graph-structured data by aggregating neighbourhood features."
        );
    }

    /** Build a larger plain-text dataset for split demo. */
    private static List<String> buildLargerDataset(int size) {
        List<String> data = new ArrayList<>(size);
        String[] templates = {
            "Training example %d about machine learning: models learn from data to make predictions.",
            "Data science example %d: statistical methods extract insights from structured datasets.",
            "Example %d on natural language processing: text is tokenised and embedded before modelling.",
            "Software engineering example %d: clean code is tested, reviewed, and documented thoroughly."
        };
        for (int i = 0; i < size; i++) {
            data.add(String.format(templates[i % templates.length], i));
        }
        return data;
    }

    /** Build labeled data with domain annotations for stratified splitting. */
    private static List<LabeledText> buildLabeledDataset() {
        List<LabeledText> data = new ArrayList<>();
        String[][] items = {
            {"code",      "Implement a binary search tree insertion operation in Java."},
            {"code",      "Write a Python function to compute the Fibonacci sequence iteratively."},
            {"code",      "Explain the difference between an interface and an abstract class in Java."},
            {"code",      "How do you handle null pointer exceptions safely in modern Java?"},
            {"code",      "Describe how HashMap collision resolution works in Java 8 and later."},
            {"science",   "Explain how mRNA vaccines trigger an immune response."},
            {"science",   "What is quantum entanglement and why is it significant for computing?"},
            {"science",   "Describe the process of photosynthesis in C3 and C4 plants."},
            {"science",   "How does the Krebs cycle produce ATP in cellular respiration?"},
            {"science",   "Explain the general and special theories of relativity at a high level."},
            {"general",   "What are the main differences between democracy and republicanism?"},
            {"general",   "Summarise the causes and consequences of the Industrial Revolution."},
            {"general",   "Explain the philosophical concept of Occam's Razor with an example."},
            {"general",   "What distinguishes a recession from a depression in macroeconomics?"},
            {"general",   "Describe how central banks use interest rates to control inflation."}
        };
        for (String[] item : items) {
            data.add(new LabeledText(item[0], item[1]));
        }
        return data;
    }

    /** Build strings with variable lengths for bucket batching demo. */
    private static List<String> buildVariableLengthTexts(int count) {
        List<String> texts = new ArrayList<>(count);
        // Create texts in three rough length bands:
        //   short  ~60-80 tokens  => ~240-320 chars
        //   medium ~130-160 tokens => ~520-640 chars
        //   long   ~400-512 tokens => ~1600-2048 chars
        String shortTemplate  = "Short text example %d. Machine learning models require curated data.";
        String mediumTemplate = "Medium length example %d. Transfer learning leverages pre-trained " +
                "representations to improve performance on downstream tasks with limited labelled data. " +
                "Fine-tuning adjusts the model parameters on the target task dataset.";
        String longTemplate   = "Long example %d. The development of large language models represents " +
                "a significant milestone in artificial intelligence research. These models, trained on " +
                "vast corpora of text using self-supervised objectives such as next-token prediction, " +
                "acquire broad linguistic and world knowledge that can be transferred to diverse " +
                "downstream tasks through fine-tuning or prompting. Scaling laws suggest that model " +
                "capability improves predictably with both parameter count and training compute, " +
                "motivating continued investment in ever-larger architectures and training runs. " +
                "Key challenges include alignment with human values, factual accuracy, and reducing " +
                "the carbon footprint of training at scale.";
        for (int i = 0; i < count; i++) {
            int group = i % 3;
            if (group == 0) texts.add(String.format(shortTemplate,  i));
            else if (group == 1) texts.add(String.format(mediumTemplate, i));
            else texts.add(String.format(longTemplate, i));
        }
        return texts;
    }

    /** Build domain-specific data for the weighted mixing demo. */
    private static List<String> buildDomainData(String domain, int count) {
        List<String> data = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            data.add(domain + "_example_" + i + ": training text for the " + domain + " domain.");
        }
        return data;
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /** Count the number of 1s in an int mask array. */
    private static int countOnes(int[] mask) {
        int n = 0;
        for (int v : mask) n += v;
        return n;
    }

    /**
     * Render the first {@code len} positions of a loss mask as a compact string,
     * e.g. "000000011111111110000" for easy visual inspection.
     */
    private static String maskPreview(int[] mask, int len) {
        int end = Math.min(len, mask.length);
        StringBuilder sb = new StringBuilder(end);
        for (int i = 0; i < end; i++) sb.append(mask[i]);
        if (mask.length > len) sb.append("...");
        return sb.toString();
    }

    /** Simple container used for stratified splitting demo. */
    private static class LabeledText {
        final String domain;
        final String text;
        LabeledText(String domain, String text) {
            this.domain = domain;
            this.text   = text;
        }
        @Override public String toString() { return domain + ": " + text.substring(0, 40) + "..."; }
    }
}

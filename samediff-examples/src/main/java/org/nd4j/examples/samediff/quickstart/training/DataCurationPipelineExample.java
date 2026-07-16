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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Data Curation Pipeline for LLM Training.
 *
 * This example demonstrates the full data curation API in
 * org.nd4j.linalg.dataset.curation, covering:
 *
 * 1. Text quality filtering (length, alpha ratio, repetition)
 * 2. Deduplication (exact SHA-256 and near-duplicate MinHash LSH)
 * 3. Instruction formatting with chat templates and loss masking
 * 4. Train/validation/test splitting with stratification support
 * 5. Length-based batching to reduce padding waste
 * 6. Weighted domain mixing with temperature scaling
 *
 * These utilities prepare raw text corpora for SFT, RLHF, and
 * continued pretraining pipelines.
 */
public class DataCurationPipelineExample {
    private static final Logger log = LoggerFactory.getLogger(DataCurationPipelineExample.class);

    public static void main(String[] args) {

        // =====================================================================
        // 1. Text Quality Filtering
        // =====================================================================
        log.info("=== 1. Text Quality Filtering ===");

        // Configure a heuristic quality filter chain (AND logic by default).
        // Each check that fires adds a rejection reason.
        TextQualityFilter qualityFilter = TextQualityFilter.builder()
                .minChars(50)                // Reject very short texts
                .maxChars(100_000)           // Reject unusually long texts
                .minWords(10)               // Require at least 10 words
                .minAlphaRatio(0.5)         // At least 50% alphabetic characters
                .maxSpecialCharRatio(0.2)   // At most 20% special characters
                .maxRepetitionRatio(0.3)    // At most 30% repeated n-grams (3-grams default)
                .maxAllCapsRatio(0.3)       // At most 30% uppercase letters
                .maxWhitespaceRatio(0.3)    // At most 30% whitespace
                .build();

        // Evaluate individual texts with detailed reasons
        String[] testTexts = {
            "The quick brown fox jumps over the lazy dog. This is a complete English sentence with good quality.",
            "hi",                                // Too short
            "BUY NOW!!! AMAZING DEAL!!! 100% FREE!!! CLICK HERE NOW!!!",  // All caps, spam
            "word word word word word word word word word word word word",   // High repetition
        };

        for (String text : testTexts) {
            FilterResult result = qualityFilter.evaluate(text);
            log.info("  Text: '{}'...", text.substring(0, Math.min(40, text.length())));
            log.info("    Accepted: {}", result.isAccepted());
            if (!result.isAccepted()) {
                log.info("    Reasons: {}", result.getRejectionReasons());
            }
        }

        // Batch filtering: keep only accepted texts
        List<String> rawTexts = new ArrayList<>(Arrays.asList(testTexts));
        List<String> filteredTexts = new ArrayList<>();
        for (String text : rawTexts) {
            if (qualityFilter.accept(text)) {
                filteredTexts.add(text);
            }
        }
        log.info("  Kept {}/{} texts after quality filtering", filteredTexts.size(), rawTexts.size());

        // =====================================================================
        // 2. Text Deduplication
        // =====================================================================
        log.info("=== 2. Text Deduplication ===");

        // Exact deduplication using SHA-256 hashes (case/whitespace normalized)
        TextDeduplicator exactDedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(false)
                .build();

        List<String> withDuplicates = Arrays.asList(
            "The cat sat on the mat.",
            "A completely different sentence about natural language processing.",
            "The cat sat on the mat.",         // exact duplicate
            "  The cat   sat on the  mat.  "   // whitespace-normalized duplicate
        );
        List<String> deduplicated = exactDedup.deduplicate(withDuplicates);
        log.info("  Exact dedup: {} -> {} texts", withDuplicates.size(), deduplicated.size());

        // Near-duplicate detection using MinHash LSH (Jaccard similarity)
        TextDeduplicator minHashDedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(true)
                .shingleSize(5)             // 5-word shingles
                .numBands(16)               // LSH bands (more = higher recall)
                .rowsPerBand(8)             // Rows per band (numBands * rowsPerBand = total hashes)
                .jaccardThreshold(0.8)      // Similarity threshold (0.8 = 80% similar = duplicate)
                .seed(42L)
                .build();

        // Streaming deduplication: process one document at a time
        List<String> corpus = Arrays.asList(
            "Machine learning is a subset of artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",  // exact dup
            "Machine learning is a type of artificial intelligence.",    // near-dup (>80% similar)
            "Deep learning uses neural networks with many layers.",       // unique
            "Reinforcement learning trains agents through reward signals." // unique
        );

        int uniqueCount = 0;
        minHashDedup.reset(); // clear streaming state
        for (String doc : corpus) {
            if (minHashDedup.addIfUnique(doc)) {
                uniqueCount++;
            }
        }
        log.info("  MinHash near-dedup: {} -> {} unique documents", corpus.size(), uniqueCount);

        // =====================================================================
        // 3. Instruction Formatting with Chat Templates
        // =====================================================================
        log.info("=== 3. Instruction Formatting ===");

        // ChatML template (used by Qwen, Mistral, etc.)
        InstructionDataFormatter chatMLFormatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        // Format a simple instruction-response pair
        FormattedExample simple = chatMLFormatter.format(
            "What is the capital of France?",
            "The capital of France is Paris."
        );
        log.info("  Simple format:\n{}", simple.getText());
        log.info("  Loss mask length: {}", simple.getLossMask().length);
        int trainableChars = countTrainable(simple.getLossMask());
        log.info("  Trainable chars: {} / {} total (assistant response only)",
                trainableChars, simple.getLossMask().length);

        // Format with system prompt
        FormattedExample withSystem = chatMLFormatter.format(
            "You are a helpful and concise AI assistant.",
            "Explain recursion in one sentence.",
            "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem."
        );
        log.info("  With system prompt:\n{}", withSystem.getText());

        // Format a multi-turn conversation
        List<ConversationTurn> conversation = Arrays.asList(
            ConversationTurn.system("You are a coding assistant."),
            ConversationTurn.user("How do I reverse a list in Python?"),
            ConversationTurn.assistant("You can use `my_list[::-1]` or `list(reversed(my_list))`."),
            ConversationTurn.user("Which is faster?"),
            ConversationTurn.assistant("Slicing `my_list[::-1]` creates a new list and is generally faster for most sizes.")
        );
        FormattedExample multiTurn = chatMLFormatter.format(conversation);
        log.info("  Multi-turn conversation length: {} chars", multiTurn.getText().length());

        // Compare different chat templates
        log.info("  --- Chat Template Comparison ---");
        for (ChatTemplate template : ChatTemplate.values()) {
            InstructionDataFormatter fmt = InstructionDataFormatter.builder()
                    .template(template)
                    .build();
            FormattedExample ex = fmt.format("Hello", "Hi there!");
            log.info("    {}: {} chars, template='{}'",
                    template.name(), ex.getText().length(), template.name());
        }

        // Llama 3 template example
        InstructionDataFormatter llama3Formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.LLAMA3)
                .build();
        FormattedExample llama3Ex = llama3Formatter.format(
                "You are Llama 3.",
                "What are you?",
                "I am Llama 3, a large language model by Meta AI."
        );
        log.info("  Llama 3 format:\n{}", llama3Ex.getText());

        // =====================================================================
        // 4. Train/Validation/Test Splitting
        // =====================================================================
        log.info("=== 4. Dataset Splitting ===");

        List<String> dataset = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            dataset.add("Example " + i + ": This is a training example with some content.");
        }

        // Random split: 80% train, 10% val, 10% test
        StratifiedSplitter<String> splitter = new StratifiedSplitter<>(42L);
        SplitResult<String> splits = splitter.split(dataset, 0.8, 0.1);
        log.info("  Random split (80/10/10):");
        log.info("    Train: {} examples", splits.getTrain().size());
        log.info("    Val:   {} examples", splits.getValidation().size());
        log.info("    Test:  {} examples", splits.getTest().size());

        // Stratified split by category label
        List<String> labeledData = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            labeledData.add("Example " + i);
        }

        // Stratify by length bucket (short vs long)
        SplitResult<String> stratSplits = splitter.splitStratified(
            labeledData, 0.8, 0.1,
            text -> text.length() > 12 ? "long" : "short"  // stratify by length
        );
        log.info("  Stratified split (80/10/10 by length):");
        log.info("    Train: {}, Val: {}, Test: {}",
                stratSplits.getTrain().size(), stratSplits.getValidation().size(), stratSplits.getTest().size());

        // =====================================================================
        // 5. Length-Based Batching (bucket batching)
        // =====================================================================
        log.info("=== 5. Length-Based Batching ===");

        // Bucket batching groups sequences by length to minimize padding.
        // Sequences in [0,128], (128,256], (256,512], (512,1024], (1024,2048] are batched together.
        List<String> sequences = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            int len = (i % 5 + 1) * 100;  // lengths: 100, 200, 300, 400, 500
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < len; j++) sb.append('x');
            sequences.add(sb.toString());
        }

        // Fixed batch size per bucket
        LengthBucketingIterator<String> bucketIter = LengthBucketingIterator.<String>builder()
                .bucketBoundaries(128, 256, 512, 1024, 2048)  // bucket size limits
                .fixedBatchSize(8)                              // sequences per batch
                .lengthFunction(String::length)
                .shuffle(42L)
                .build(sequences);

        int batchCount = 0;
        while (bucketIter.hasNext()) {
            List<String> batch = bucketIter.next();
            batchCount++;
        }
        log.info("  Created {} batches from {} sequences (batch size=8)", batchCount, sequences.size());

        // Token budget batching: limit total tokens per batch (variable batch size)
        LengthBucketingIterator<String> tokenBudgetIter = LengthBucketingIterator.<String>builder()
                .bucketBoundaries(128, 256, 512, 1024, 2048)
                .tokenBudget(2048)                  // At most 2048 tokens per batch
                .lengthFunction(String::length)
                .shuffle(42L)
                .build(sequences);

        int tokenBudgetBatches = 0;
        while (tokenBudgetIter.hasNext()) {
            tokenBudgetIter.next();
            tokenBudgetBatches++;
        }
        log.info("  Token budget (2048): {} batches from {} sequences", tokenBudgetBatches, sequences.size());

        // =====================================================================
        // 6. Weighted Domain Mixing
        // =====================================================================
        log.info("=== 6. Weighted Domain Mixing ===");

        // Mix data from different domains with specified proportions.
        // Temperature < 1 makes mixing more uniform; > 1 makes it more concentrated.
        List<String> codeDomain = new ArrayList<>();
        for (int i = 0; i < 500; i++) codeDomain.add("Code example " + i);

        List<String> mathDomain = new ArrayList<>();
        for (int i = 0; i < 300; i++) mathDomain.add("Math problem " + i);

        List<String> languageDomain = new ArrayList<>();
        for (int i = 0; i < 200; i++) languageDomain.add("Language text " + i);

        Map<String, WeightedDataMixer.WeightedSource<String>> sources = new LinkedHashMap<>();
        sources.put("code", new WeightedDataMixer.WeightedSource<>(codeDomain.iterator(), 0.5));
        sources.put("math", new WeightedDataMixer.WeightedSource<>(mathDomain.iterator(), 0.3));
        sources.put("language", new WeightedDataMixer.WeightedSource<>(languageDomain.iterator(), 0.2));

        // Temperature=1.0 preserves specified weights exactly
        WeightedDataMixer<String> mixer = new WeightedDataMixer<>(sources, 1.0, 42L);

        int totalSampled = 0;
        while (mixer.hasNext() && totalSampled < 100) {
            mixer.next();
            totalSampled++;
        }
        log.info("  Sampled {} items from weighted mix (code=50%, math=30%, language=20%)", totalSampled);
        Map<String, Long> stats = mixer.getStats();
        log.info("  Consumption: code={}, math={}, language={}",
                stats.get("code"), stats.get("math"), stats.get("language"));

        log.info("  --- Temperature Scaling ---");
        log.info("  temperature=0.5: weights more uniform (code~38%, math~34%, language~28%)");
        log.info("  temperature=1.0: exact weights     (code=50%, math=30%, language=20%)");
        log.info("  temperature=2.0: weights more concentrated on dominant domain");

        // =====================================================================
        // SUMMARY
        // =====================================================================
        log.info("=== Data Curation Pipeline Summary ===");
        log.info("  Steps for LLM training data preparation:");
        log.info("  1. TextQualityFilter   - heuristic quality gates (length, alpha, repetition)");
        log.info("  2. TextDeduplicator    - exact SHA-256 + near-dup MinHash LSH");
        log.info("  3. InstructionDataFormatter - chat templates + loss masking");
        log.info("  4. StratifiedSplitter  - train/val/test split with label stratification");
        log.info("  5. LengthBucketingIterator - group by length to minimize padding");
        log.info("  6. WeightedDataMixer   - domain mixing with temperature control");
        log.info("**************** Data Curation Pipeline Example finished ********************");
    }

    private static int countTrainable(int[] mask) {
        int count = 0;
        for (int m : mask) count += m;
        return count;
    }
}

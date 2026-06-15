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

package org.nd4j.examples.samediff.advanced.evaluation;

import org.eclipse.deeplearning4j.llm.eval.EvalConfig;
import org.eclipse.deeplearning4j.llm.eval.EvalResults;
import org.eclipse.deeplearning4j.llm.eval.EvalRunner;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.eval.benchmark.ArcBenchmark;
import org.eclipse.deeplearning4j.llm.eval.benchmark.Gsm8kBenchmark;
import org.eclipse.deeplearning4j.llm.eval.benchmark.HellaSwagBenchmark;
import org.eclipse.deeplearning4j.llm.eval.benchmark.MMLUBenchmark;
import org.eclipse.deeplearning4j.llm.eval.benchmark.TruthfulQABenchmark;
import org.eclipse.deeplearning4j.llm.eval.benchmark.WinograndeBenchmark;
import org.eclipse.deeplearning4j.llm.eval.metrics.BleuMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.RougeMetric;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

/**
 * LLM Evaluation Harness Reference Example
 *
 * This example demonstrates the standard evaluation pipeline for large language
 * models (LLMs) using SameDiff's built-in evaluation harness.
 *
 * Topics covered:
 *   1. EvalConfig: wiring a model to the evaluation harness
 *   2. Standard benchmarks: MMLU, ARC, GSM8K, HellaSwag, TruthfulQA, WinoGrande
 *   3. Running the evaluation harness: EvalRunner.run()
 *   4. Inspecting results: per-benchmark accuracy, overall score
 *   5. Perplexity evaluation: measuring language model quality on custom text
 *   6. Generation metrics: BLEU, ROUGE, Exact Match
 *   7. Custom evaluation: combining benchmarks with generation metrics
 *
 * Benchmark descriptions:
 *
 *   MMLU (Massive Multitask Language Understanding):
 *     57 academic subjects, 4-way multiple-choice.
 *     Tests broad knowledge (science, history, law, medicine, math, etc.)
 *     Standard shot count: 5-shot.
 *
 *   ARC (AI2 Reasoning Challenge):
 *     Science questions, 4-way multiple-choice.
 *     Two subsets: ARC-Easy and ARC-Challenge (harder, adversarially filtered).
 *     Standard: 25-shot.
 *
 *   GSM8K (Grade School Math 8K):
 *     8500 grade-school math word problems, free-form answer generation.
 *     Tests multi-step arithmetic reasoning.
 *     Standard: 5-shot chain-of-thought.
 *
 *   HellaSwag:
 *     4-way multiple-choice sentence completion (commonsense NLI).
 *     Tests physical situation understanding.
 *     Standard: 10-shot.
 *
 *   TruthfulQA:
 *     Measures how often a model produces truthful vs. plausible-sounding false answers.
 *     Tests against model hallucination tendencies.
 *     Standard: 0-shot.
 *
 *   WinoGrande:
 *     Commonsense pronoun resolution in two-choice Winograd schema problems.
 *     Tests physical/social commonsense reasoning.
 *     Standard: 5-shot.
 *
 * Key classes:
 *   - EvalConfig:           model + tokenizer + benchmark list + shot count config
 *   - EvalRunner:           executes the full evaluation pipeline
 *   - EvalResults:          results container with per-benchmark scores
 *   - PerplexityEvaluator:  computes perplexity on a text corpus
 *   - BleuMetric:           BLEU score for generation quality
 *   - RougeMetric:          ROUGE-L / ROUGE-N for summarization quality
 *   - ExactMatchMetric:     exact string match for QA tasks
 */
public class LLMEvalBenchmarkExample {

    public static void main(String[] args) {

        // ============================================================
        // 0. SETUP: Load a model (stub - replace with real model path)
        // ============================================================
        System.out.println("=== LLM Evaluation Harness ===");
        System.out.println("NOTE: This example shows the evaluation API.");
        System.out.println("Replace 'sd' and 'tokenizer' with your actual model.\n");

        // In a real scenario:
        //   SameDiff sd = SameDiff.fromFlatBuffers(new File("model.fb"));
        //   Tokenizer tokenizer = Tokenizer.load(new File("tokenizer.json"));
        //
        // Here we use a placeholder SameDiff instance and null tokenizer
        // to illustrate the API structure.
        SameDiff sd = SameDiff.create();
        Object tokenizer = null;  // placeholder - use real Tokenizer in production

        // ============================================================
        // 1. EVALCONFIG - Wiring model + benchmarks
        // ============================================================
        System.out.println("=== 1. EvalConfig ===");
        {
            // EvalConfig.builder() assembles the evaluation harness configuration.
            //
            // numFewShot: globally sets the default few-shot count (N).
            //   N-shot means the model is given N worked examples in its prompt
            //   before being asked the actual question.
            //   0-shot = no examples (pure zero-shot).
            //   5-shot = 5 examples prepended (most standard benchmarks use 5-shot).
            //
            // Per-benchmark shot counts can override the global default.

            EvalConfig evalConfig = EvalConfig.builder()
                    .model(sd)
                    .tokenizer(tokenizer)
                    .numFewShot(5)                              // global default: 5-shot
                    .benchmarks(Arrays.asList(
                            new MMLUBenchmark(),                // 57 subjects, 4-choice
                            new ArcBenchmark(),                 // science reasoning
                            new Gsm8kBenchmark(),               // math word problems
                            new HellaSwagBenchmark(),           // commonsense NLI
                            new TruthfulQABenchmark(),          // hallucination test
                            new WinograndeBenchmark()))         // pronoun resolution
                    .build();

            System.out.println("  Benchmarks configured: " + evalConfig.getBenchmarks().size());
            System.out.println("  Default few-shot:      " + evalConfig.getNumFewShot());
            System.out.println("  Model:                 " + (evalConfig.getModel() != null ? "set" : "null"));
            System.out.println("  Tokenizer:             " + (evalConfig.getTokenizer() != null ? "set" : "null (placeholder)"));
        }

        // ============================================================
        // 2. INDIVIDUAL BENCHMARKS - Configuration and properties
        // ============================================================
        System.out.println("\n=== 2. Individual Benchmark Configurations ===");
        {
            // Each benchmark can be configured independently.
            // numShot on a benchmark overrides the EvalConfig global default.

            MMLUBenchmark mmlu = MMLUBenchmark.builder()
                    .numShot(5)          // 5-shot is standard for MMLU
                    .subjects(null)      // null = all 57 subjects; or List.of("mathematics", "law")
                    .build();
            System.out.println("  MMLU:");
            System.out.println("    Shots:     " + mmlu.getNumShot());
            System.out.println("    Subjects:  " + (mmlu.getSubjects() == null ? "all 57" : mmlu.getSubjects()));
            System.out.println("    Task type: multiple-choice (A/B/C/D)");
            System.out.println("    Scoring:   accuracy (fraction of correct answers)");

            ArcBenchmark arc = ArcBenchmark.builder()
                    .numShot(25)              // 25-shot is standard for ARC
                    .useChallengeSet(true)    // false = ARC-Easy, true = ARC-Challenge (harder)
                    .build();
            System.out.println("\n  ARC:");
            System.out.println("    Shots:         " + arc.getNumShot());
            System.out.println("    Challenge set: " + arc.isUseChallengeSet());
            System.out.println("    Task type:     multiple-choice science questions");

            Gsm8kBenchmark gsm8k = Gsm8kBenchmark.builder()
                    .numShot(5)
                    .useChainOfThought(true)  // chain-of-thought prompting improves GSM8K significantly
                    .build();
            System.out.println("\n  GSM8K:");
            System.out.println("    Shots:            " + gsm8k.getNumShot());
            System.out.println("    Chain of thought: " + gsm8k.isUseChainOfThought());
            System.out.println("    Task type:        free-form math answer generation");
            System.out.println("    Scoring:          exact match on final numeric answer");

            HellaSwagBenchmark hellaswag = HellaSwagBenchmark.builder()
                    .numShot(10)
                    .build();
            System.out.println("\n  HellaSwag:");
            System.out.println("    Shots:     " + hellaswag.getNumShot());
            System.out.println("    Task type: 4-way sentence completion");

            TruthfulQABenchmark truthfulqa = TruthfulQABenchmark.builder()
                    .numShot(0)          // 0-shot is standard for TruthfulQA
                    .build();
            System.out.println("\n  TruthfulQA:");
            System.out.println("    Shots:     " + truthfulqa.getNumShot());
            System.out.println("    Task type: factual accuracy vs plausible hallucination");

            WinograndeBenchmark winogrande = WinograndeBenchmark.builder()
                    .numShot(5)
                    .build();
            System.out.println("\n  WinoGrande:");
            System.out.println("    Shots:     " + winogrande.getNumShot());
            System.out.println("    Task type: 2-way pronoun resolution");
        }

        // ============================================================
        // 3. RUNNING THE EVALUATION HARNESS
        // ============================================================
        System.out.println("\n=== 3. EvalRunner ===");
        {
            EvalConfig config = EvalConfig.builder()
                    .model(sd)
                    .tokenizer(tokenizer)
                    .numFewShot(5)
                    .benchmarks(Arrays.asList(
                            new MMLUBenchmark(),
                            new ArcBenchmark(),
                            new Gsm8kBenchmark()))
                    .build();

            // EvalRunner.run(config) executes all benchmarks sequentially.
            // For each benchmark it:
            //   1. Loads the benchmark dataset
            //   2. Constructs N-shot prompts for each question
            //   3. Runs model inference (tokenize -> forward pass -> decode)
            //   4. Scores the outputs (accuracy for MCQ, exact match for math)
            //   5. Accumulates results into EvalResults
            //
            // NOTE: This call requires a real model and tokenizer to produce
            // meaningful results. With a stub model, results will be random/empty.
            System.out.println("  EvalRunner.run(config) executes all configured benchmarks.");
            System.out.println("  (with a real model, this would run inference on each question)");
            System.out.println();
            System.out.println("  Usage:");
            System.out.println("    EvalResults results = EvalRunner.run(config);");
            System.out.println("    double mmlAccuracy   = results.getAccuracy(\"mmlu\");");
            System.out.println("    double arcAccuracy   = results.getAccuracy(\"arc\");");
            System.out.println("    double gsm8kAccuracy = results.getAccuracy(\"gsm8k\");");
            System.out.println("    double overall       = results.getOverallScore();");

            EvalResults stubResults = EvalResults.empty();
            System.out.println("\n  EvalResults.empty() stub created.");
        }

        // ============================================================
        // 4. INSPECTING EVAL RESULTS
        // ============================================================
        System.out.println("\n=== 4. EvalResults API ===");
        {
            EvalResults stubResults = EvalResults.empty();

            System.out.println("  EvalResults methods:");
            System.out.println("    results.getAccuracy(\"mmlu\")        - MMLU accuracy [0,1]");
            System.out.println("    results.getAccuracy(\"arc\")         - ARC accuracy [0,1]");
            System.out.println("    results.getAccuracy(\"gsm8k\")       - GSM8K accuracy [0,1]");
            System.out.println("    results.getAccuracy(\"hellaswag\")   - HellaSwag accuracy [0,1]");
            System.out.println("    results.getAccuracy(\"truthfulqa\")  - TruthfulQA accuracy [0,1]");
            System.out.println("    results.getAccuracy(\"winogrande\")  - WinoGrande accuracy [0,1]");
            System.out.println("    results.getOverallScore()          - average across all benchmarks");
            System.out.println("    results.getBenchmarkNames()        - list of evaluated benchmarks");
            System.out.println("    results.toSummaryString()          - formatted table output");
            System.out.println();

            // Typical results from open LLMs (approximate, for reference)
            System.out.println("  Reference scores (approximate):");
            System.out.println("  +-------------+----------+--------+----------+");
            System.out.println("  | Benchmark   | LLaMA2-7B|  13B   |   70B    |");
            System.out.println("  +-------------+----------+--------+----------+");
            System.out.println("  | MMLU        |   45.3%  | 54.8%  |  68.9%   |");
            System.out.println("  | ARC-C       |   53.1%  | 59.4%  |  67.3%   |");
            System.out.println("  | GSM8K       |   14.6%  | 28.7%  |  56.8%   |");
            System.out.println("  | HellaSwag   |   77.2%  | 80.7%  |  87.3%   |");
            System.out.println("  | TruthfulQA  |   38.8%  | 41.9%  |  44.9%   |");
            System.out.println("  | WinoGrande  |   69.2%  | 72.8%  |  80.0%   |");
            System.out.println("  +-------------+----------+--------+----------+");
        }

        // ============================================================
        // 5. PERPLEXITY EVALUATION
        // ============================================================
        System.out.println("\n=== 5. PerplexityEvaluator ===");
        {
            // Perplexity measures how well a language model predicts a text corpus.
            //
            //   PPL = exp(-1/N * sum_i log P(x_i | x_1, ..., x_{i-1}))
            //
            // Lower perplexity = better language model.
            //
            // Common baselines on WikiText-2:
            //   GPT-2 (1.5B):   ~18 PPL
            //   LLaMA-7B:       ~12 PPL
            //   LLaMA-13B:      ~11 PPL
            //   LLaMA-70B:       ~8 PPL
            //
            // Perplexity is especially useful for:
            //   - Comparing model checkpoints during training (lower is better)
            //   - Evaluating domain adaptation (lower = better domain fit)
            //   - Measuring quantization quality loss (higher PPL = degraded model)

            PerplexityEvaluator pplEval = PerplexityEvaluator.builder()
                    .model(sd)
                    .tokenizer(tokenizer)
                    .strideLength(512)   // sliding window stride (< maxSeqLen for full coverage)
                    .build();

            System.out.println("  PerplexityEvaluator configured:");
            System.out.println("    strideLength: " + pplEval.getStrideLength());
            System.out.println("    (stride < maxSeqLen: overlapping windows for full text coverage)");
            System.out.println();
            System.out.println("  Usage:");
            System.out.println("    String text = Files.readString(Path.of(\"wikitext2.txt\"));");
            System.out.println("    double ppl  = pplEval.evaluate(text);");
            System.out.println("    System.out.println(\"Perplexity: \" + ppl);");
            System.out.println();
            System.out.println("  Standard corpora for PPL benchmarking:");
            System.out.println("    WikiText-2    - clean Wikipedia articles");
            System.out.println("    WikiText-103  - larger Wikipedia set (1B tokens)");
            System.out.println("    Penn Treebank - WSJ news articles");
            System.out.println("    C4            - web text (for instruction-tuned models)");
        }

        // ============================================================
        // 6. GENERATION METRICS - BLEU, ROUGE, Exact Match
        // ============================================================
        System.out.println("\n=== 6. Generation Quality Metrics ===");
        {
            // --- BLEU ---
            // BLEU (Bilingual Evaluation Understudy):
            //   Measures n-gram precision between generated and reference text.
            //   Range: [0, 1], higher is better.
            //   BLEU-4 (unigram through 4-gram) is the standard.
            //   Limitation: penalizes valid paraphrases (exact n-gram match only).
            //   Standard for: machine translation, text generation.

            BleuMetric bleu = BleuMetric.builder()
                    .maxNgram(4)           // BLEU-4 (standard; covers 1-, 2-, 3-, 4-grams)
                    .smoothing(true)       // Chen-Cherry smoothing (better for short texts)
                    .build();

            List<String> hypotheses = Arrays.asList(
                    "The cat sat on the mat",
                    "The quick brown fox jumped"
            );
            List<String> references = Arrays.asList(
                    "The cat is sitting on the mat",
                    "A quick brown fox jumped over the fence"
            );

            double bleuScore = bleu.compute(hypotheses, references);
            System.out.println("  BLEU-4 example:");
            System.out.println("    hypothesis[0]: \"" + hypotheses.get(0) + "\"");
            System.out.println("    reference[0]:  \"" + references.get(0) + "\"");
            System.out.println("    BLEU-4 score:  " + bleuScore);
            System.out.println("    Max n-gram:    " + bleu.getMaxNgram());
            System.out.println("    Smoothing:     " + bleu.isSmoothing());

            // --- ROUGE ---
            // ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
            //   Measures recall of n-grams and longest common subsequence.
            //   ROUGE-1: unigram overlap
            //   ROUGE-2: bigram overlap
            //   ROUGE-L: longest common subsequence (order-aware, more flexible)
            //   Standard for: summarization evaluation.

            RougeMetric rouge = RougeMetric.builder()
                    .variant(RougeMetric.Variant.ROUGE_L)   // LCS-based (most common)
                    .build();

            double rougeScore = rouge.compute(hypotheses, references);
            System.out.println("\n  ROUGE-L example:");
            System.out.println("    ROUGE-L score: " + rougeScore);
            System.out.println("    Variant:       " + rouge.getVariant());
            System.out.println("    Use ROUGE for: summarization, abstractive generation");

            // --- Exact Match ---
            // ExactMatchMetric:
            //   Binary: 1 if generated answer exactly matches reference (after normalization).
            //   Normalization: lowercase, strip punctuation, collapse whitespace.
            //   Standard for: extractive QA (SQuAD), GSM8K math answers.

            ExactMatchMetric em = new ExactMatchMetric();
            List<String> predicted = Arrays.asList("Paris", "42", "Abraham Lincoln");
            List<String> gold      = Arrays.asList("Paris", "43", "Abraham Lincoln");
            double emScore = em.compute(predicted, gold);
            System.out.println("\n  Exact Match example:");
            System.out.println("    predicted: " + predicted);
            System.out.println("    gold:      " + gold);
            System.out.println("    EM score:  " + emScore + "  (2/3 correct: Paris and Lincoln match, 42 != 43)");
        }

        // ============================================================
        // 7. FULL EVALUATION PIPELINE
        // ============================================================
        System.out.println("\n=== 7. Full Evaluation Pipeline ===");
        {
            // A complete evaluation run for an LLM typically combines:
            //   1. Academic benchmarks (MMLU, ARC, ...) for knowledge/reasoning
            //   2. Perplexity on held-out text for language model quality
            //   3. Generation metrics (BLEU/ROUGE) for conditional generation
            //
            // This section shows how to configure and conceptually run all three.

            System.out.println("  Step 1: Academic benchmarks");
            EvalConfig fullConfig = EvalConfig.builder()
                    .model(sd)
                    .tokenizer(tokenizer)
                    .numFewShot(5)
                    .benchmarks(Arrays.asList(
                            MMLUBenchmark.builder().numShot(5).build(),
                            ArcBenchmark.builder().numShot(25).useChallengeSet(true).build(),
                            Gsm8kBenchmark.builder().numShot(5).useChainOfThought(true).build(),
                            HellaSwagBenchmark.builder().numShot(10).build(),
                            TruthfulQABenchmark.builder().numShot(0).build(),
                            WinograndeBenchmark.builder().numShot(5).build()))
                    .build();

            System.out.println("    Configured " + fullConfig.getBenchmarks().size() + " benchmarks");
            System.out.println("    // EvalResults results = EvalRunner.run(fullConfig);");
            System.out.println("    // System.out.println(results.toSummaryString());");

            System.out.println("\n  Step 2: Perplexity on WikiText-2");
            PerplexityEvaluator ppl = PerplexityEvaluator.builder()
                    .model(sd)
                    .tokenizer(tokenizer)
                    .strideLength(512)
                    .build();
            System.out.println("    // String wikiText = Files.readString(Path.of(\"wikitext2.txt\"));");
            System.out.println("    // double wikiPPL  = ppl.evaluate(wikiText);");
            System.out.println("    // System.out.println(\"WikiText-2 PPL: \" + wikiPPL);");

            System.out.println("\n  Step 3: Generation metrics (e.g., CNN/DailyMail summarization)");
            RougeMetric rouge2 = RougeMetric.builder()
                    .variant(RougeMetric.Variant.ROUGE_2)
                    .build();
            System.out.println("    // double rougeScore = rouge.compute(summaries, references);");
            System.out.println("    // System.out.println(\"ROUGE-2: \" + rougeScore);");

            System.out.println();
            System.out.println("  A complete evaluation report covers:");
            System.out.println("    Knowledge/Reasoning: MMLU, ARC, GSM8K, HellaSwag, TruthfulQA, WinoGrande");
            System.out.println("    Language Quality:    Perplexity on WikiText-2, PTB, or C4");
            System.out.println("    Generation:         BLEU/ROUGE/EM for translation/summarization/QA");
        }

        // ============================================================
        // BENCHMARK REFERENCE TABLE
        // ============================================================
        System.out.println("\n=== Benchmark Reference Table ===");
        System.out.println("  Benchmark    | Task type        | Shots | Primary metric | Notes");
        System.out.println("  -------------|------------------|-------|----------------|---------------------------");
        System.out.println("  MMLU         | MCQ (57 subj)    |   5   | Accuracy       | Broad knowledge coverage");
        System.out.println("  ARC-C        | MCQ science      |  25   | Accuracy       | Adversarially filtered");
        System.out.println("  GSM8K        | Math generation  |   5   | Exact Match    | Chain-of-thought helps");
        System.out.println("  HellaSwag    | MCQ sentence     |  10   | Accuracy       | Physical commonsense");
        System.out.println("  TruthfulQA   | Truthfulness     |   0   | Accuracy       | Hallucination probe");
        System.out.println("  WinoGrande   | Pronoun 2-way    |   5   | Accuracy       | Social commonsense");
        System.out.println("  Perplexity   | LM quality       |   -   | PPL (lower)    | WikiText-2 standard");
        System.out.println("  BLEU-4       | Translation/gen  |   -   | Score [0,1]    | n-gram precision");
        System.out.println("  ROUGE-L      | Summarization    |   -   | Score [0,1]    | LCS recall");
        System.out.println("  Exact Match  | Extractive QA    |   -   | Fraction [0,1] | After normalization");

        System.out.println("\nLLM evaluation benchmark example completed successfully.");
    }
}

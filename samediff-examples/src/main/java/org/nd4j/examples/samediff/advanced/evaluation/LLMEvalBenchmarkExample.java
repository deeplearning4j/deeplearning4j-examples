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
import org.eclipse.deeplearning4j.llm.eval.EvalResult;
import org.eclipse.deeplearning4j.llm.eval.EvalRunner;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.eval.benchmark.BenchmarkTask;
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
 *   3. Running the evaluation harness: EvalRunner.evaluateAll()
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
 *   - EvalConfig:           shared settings (numFewShot, maxSamples, maxNewTokens, batchSize)
 *   - EvalRunner:           executes the evaluation pipeline (instance, with TextGenerator)
 *   - EvalResult:           per-benchmark result (accuracy, summary, metric scores)
 *   - PerplexityEvaluator:  static methods for computing perplexity on a text corpus
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

            // EvalConfig accepts numFewShot, maxSamples, maxNewTokens, batchSize.
            // Model and tokenizer are passed directly to EvalRunner, not EvalConfig.
            // Benchmark instances are passed directly to EvalRunner.evaluate/evaluateAll.
            List<BenchmarkTask> benchmarkList = Arrays.asList(
                    new MMLUBenchmark(),                // 57 subjects, 4-choice
                    new ArcBenchmark(),                 // science reasoning
                    new Gsm8kBenchmark(),               // math word problems
                    new HellaSwagBenchmark(),           // commonsense NLI
                    new TruthfulQABenchmark(),          // hallucination test
                    new WinograndeBenchmark());         // pronoun resolution

            EvalConfig evalConfig = EvalConfig.builder()
                    .numFewShot(5)                              // global default: 5-shot
                    .build();

            System.out.println("  Benchmarks configured: " + benchmarkList.size());
            System.out.println("  Default few-shot:      " + evalConfig.getNumFewShot());
            System.out.println("  Model:                 " + (sd != null ? "set" : "null"));
            System.out.println("  Tokenizer:             " + (tokenizer != null ? "set" : "null (placeholder)"));
        }

        // ============================================================
        // 2. INDIVIDUAL BENCHMARKS - Configuration and properties
        // ============================================================
        System.out.println("\n=== 2. Individual Benchmark Configurations ===");
        {
            // Each benchmark is constructed with its no-arg constructor.
            // Shot count and other properties are configured via EvalConfig or
            // accessed via the BenchmarkTask interface (defaultFewShot(), name(), etc.).

            MMLUBenchmark mmlu = new MMLUBenchmark();
            System.out.println("  MMLU:");
            System.out.println("    Default few-shot: " + mmlu.defaultFewShot());
            System.out.println("    Task type:        multiple-choice (A/B/C/D), 57 subjects");
            System.out.println("    Scoring:          accuracy (fraction of correct answers)");

            ArcBenchmark arc = new ArcBenchmark();
            System.out.println("\n  ARC:");
            System.out.println("    Default few-shot: " + arc.defaultFewShot());
            System.out.println("    Task type:        multiple-choice science questions");
            System.out.println("    Subsets:          ARC-Easy and ARC-Challenge (adversarially filtered)");

            Gsm8kBenchmark gsm8k = new Gsm8kBenchmark();
            System.out.println("\n  GSM8K:");
            System.out.println("    Default few-shot: " + gsm8k.defaultFewShot());
            System.out.println("    Task type:        free-form math answer generation");
            System.out.println("    Scoring:          exact match on final numeric answer");
            System.out.println("    Note:             chain-of-thought prompting improves results significantly");

            HellaSwagBenchmark hellaswag = new HellaSwagBenchmark();
            System.out.println("\n  HellaSwag:");
            System.out.println("    Default few-shot: " + hellaswag.defaultFewShot());
            System.out.println("    Task type:        4-way sentence completion (commonsense NLI)");

            TruthfulQABenchmark truthfulqa = new TruthfulQABenchmark();
            System.out.println("\n  TruthfulQA:");
            System.out.println("    Default few-shot: " + truthfulqa.defaultFewShot());
            System.out.println("    Task type:        factual accuracy vs plausible hallucination");

            WinograndeBenchmark winogrande = new WinograndeBenchmark();
            System.out.println("\n  WinoGrande:");
            System.out.println("    Default few-shot: " + winogrande.defaultFewShot());
            System.out.println("    Task type:        2-way pronoun resolution");
        }

        // ============================================================
        // 3. RUNNING THE EVALUATION HARNESS
        // ============================================================
        System.out.println("\n=== 3. EvalRunner ===");
        {
            // EvalConfig controls shared settings: numFewShot, maxSamples, maxNewTokens, batchSize.
            // Model and tokenizer are passed to EvalRunner directly, not through EvalConfig.
            // Benchmark instances are also passed directly to EvalRunner.
            EvalConfig config = EvalConfig.builder()
                    .numFewShot(5)
                    .build();

            // EvalRunner is instantiated and then its instance methods are called.
            // evaluate(TextGenerator, BenchmarkTask)        - single benchmark
            // evaluateAll(TextGenerator, List<BenchmarkTask>) - multiple benchmarks
            //
            // For each benchmark it:
            //   1. Loads the benchmark dataset
            //   2. Constructs N-shot prompts for each question
            //   3. Runs model inference (tokenize -> forward pass -> decode)
            //   4. Scores the outputs (accuracy for MCQ, exact match for math)
            //   5. Returns EvalResult (singular) with scores and sample details
            //
            // NOTE: This call requires a real model and tokenizer to produce
            // meaningful results. With a stub model, results will be random/empty.
            System.out.println("  EvalRunner evaluates benchmarks via instance methods.");
            System.out.println("  (with a real model and tokenizer, this would run inference on each question)");
            System.out.println();
            System.out.println("  Usage:");
            System.out.println("    EvalRunner runner = new EvalRunner();");
            System.out.println("    List<BenchmarkTask> tasks = Arrays.asList(new MMLUBenchmark(), ...);");
            System.out.println("    Map<String, EvalResult> results = runner.evaluateAll(textGenerator, tasks, config);");
            System.out.println("    EvalResult mmluResult = results.get(\"mmlu\");");
            System.out.println("    double accuracy = mmluResult.accuracy();");
            System.out.println("    System.out.println(mmluResult.summary());");

            // Stub EvalResult using builder for illustration (no real inference performed)
            EvalResult stubResult = EvalResult.builder()
                    .benchmarkName("stub")
                    .primaryScore(0.0)
                    .totalSamples(0)
                    .correctSamples(0)
                    .build();
            System.out.println("\n  EvalResult stub created (benchmarkName=" + stubResult.getBenchmarkName()
                    + ", primaryScore=" + stubResult.getPrimaryScore() + ")");
        }

        // ============================================================
        // 4. INSPECTING EVAL RESULTS
        // ============================================================
        System.out.println("\n=== 4. EvalResult API ===");
        {
            // EvalResult (singular) is returned per-benchmark by EvalRunner.
            // Use EvalRunner.evaluateAll() to get a Map<String, EvalResult>.
            EvalResult stubResult = EvalResult.builder()
                    .benchmarkName("stub")
                    .primaryScore(0.0)
                    .totalSamples(0)
                    .correctSamples(0)
                    .build();

            System.out.println("  EvalResult methods (per benchmark):");
            System.out.println("    result.accuracy()            - accuracy [0,1]");
            System.out.println("    result.getBenchmarkName()    - name of this benchmark");
            System.out.println("    result.getPrimaryScore()     - primary metric score");
            System.out.println("    result.getMetricScores()     - map of metric name -> score");
            System.out.println("    result.getCategoryScores()   - map of category -> score");
            System.out.println("    result.getTotalSamples()     - number of samples evaluated");
            System.out.println("    result.getCorrectSamples()   - number of correct answers");
            System.out.println("    result.getEvaluationTimeMs() - wall clock time for evaluation");
            System.out.println("    result.getSampleResults()    - list of per-sample results");
            System.out.println("    result.summary()             - formatted summary string");
            System.out.println("    result.writeJson(file)       - persist results to JSON file");
            System.out.println();
            System.out.println("  Usage with evaluateAll:");
            System.out.println("    Map<String, EvalResult> results = runner.evaluateAll(textGen, tasks);");
            System.out.println("    double mmlAccuracy   = results.get(\"mmlu\").accuracy();");
            System.out.println("    double arcAccuracy   = results.get(\"arc\").accuracy();");
            System.out.println("    double gsm8kAccuracy = results.get(\"gsm8k\").accuracy():");
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

            // PerplexityEvaluator has only static methods — no builder, no instance creation.
            // Signatures:
            //   PerplexityEvaluator.evaluate(SameDiff model, Tokenizer tokenizer,
            //                                String text, int stride, int maxSeq)
            //   PerplexityEvaluator.evaluateWikiText2(SameDiff model, Tokenizer tokenizer,
            //                                         int stride, int maxSeq)
            //
            // stride: sliding window stride (should be < maxSeq for full token coverage)
            // maxSeq: max sequence length of the model (e.g. 2048 for LLaMA)

            System.out.println("  PerplexityEvaluator uses static methods only:");
            System.out.println("    int stride = 512;   // overlapping windows for full coverage");
            System.out.println("    int maxSeq = 2048;  // model's max context length");
            System.out.println("    (stride < maxSeqLen ensures all tokens are evaluated)");
            System.out.println();
            System.out.println("  Usage:");
            System.out.println("    String text = Files.readString(Path.of(\"wikitext2.txt\"));");
            System.out.println("    double ppl  = PerplexityEvaluator.evaluate(model, tokenizer, text, 512, 2048);");
            System.out.println("    System.out.println(\"Perplexity: \" + ppl);");
            System.out.println();
            System.out.println("  WikiText-2 convenience method:");
            System.out.println("    double ppl = PerplexityEvaluator.evaluateWikiText2(model, tokenizer, 512, 2048);");
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

            // BleuMetric: no-arg constructor (BLEU-1), or BleuMetric(int maxNgram) for BLEU-N.
            // score(String hypothesis, List<String> references) computes BLEU for one hypothesis.
            BleuMetric bleu = new BleuMetric(4);  // BLEU-4 (standard)

            String hypothesis0 = "The cat sat on the mat";
            List<String> refs0 = Arrays.asList("The cat is sitting on the mat");

            double bleuScore = bleu.score(hypothesis0, refs0);
            System.out.println("  BLEU-4 example:");
            System.out.println("    hypothesis: \"" + hypothesis0 + "\"");
            System.out.println("    reference:  \"" + refs0.get(0) + "\"");
            System.out.println("    BLEU-4 score:  " + bleuScore);
            System.out.println("    BLEU measures n-gram precision (standard BLEU-4: 1- to 4-gram)");
            System.out.println("    Note: use BleuMetric(maxNgram) to set n-gram order");

            // --- ROUGE ---
            // ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
            //   Measures recall of n-grams and longest common subsequence.
            //   ROUGE-1: unigram overlap
            //   ROUGE-2: bigram overlap
            //   ROUGE-L: longest common subsequence (order-aware, more flexible)
            //   Standard for: summarization evaluation.

            // RougeMetric: RougeMetric(), RougeMetric(RougeType), or RougeMetric(RougeType, ScoreType).
            // RougeMetric.RougeType values: ROUGE_1, ROUGE_2, ROUGE_L
            // score(String hypothesis, List<String> references) computes ROUGE for one hypothesis.
            RougeMetric rouge = new RougeMetric(RougeMetric.RougeType.ROUGE_L);  // LCS-based (most common)

            double rougeScore = rouge.score(hypothesis0, refs0);
            System.out.println("\n  ROUGE-L example:");
            System.out.println("    hypothesis: \"" + hypothesis0 + "\"");
            System.out.println("    reference:  \"" + refs0.get(0) + "\"");
            System.out.println("    ROUGE-L score: " + rougeScore);
            System.out.println("    Variant:       ROUGE_L (longest common subsequence)");
            System.out.println("    Available:     ROUGE_1 (unigram), ROUGE_2 (bigram), ROUGE_L (LCS)");
            System.out.println("    Use ROUGE for: summarization, abstractive generation");

            // --- Exact Match ---
            // ExactMatchMetric:
            //   Binary: 1 if generated answer exactly matches reference (after normalization).
            //   Normalization: lowercase, strip punctuation, collapse whitespace.
            //   Standard for: extractive QA (SQuAD), GSM8K math answers.

            // ExactMatchMetric: score(String hypothesis, List<String> references).
            // Returns 1.0 if hypothesis exactly matches any reference (after normalization), else 0.0.
            // normalize() lowercases, strips punctuation, collapses whitespace.
            ExactMatchMetric em = new ExactMatchMetric();
            double emParis   = em.score("Paris",           Arrays.asList("Paris"));    // match
            double em42      = em.score("42",              Arrays.asList("43"));       // no match
            double emLincoln = em.score("Abraham Lincoln", Arrays.asList("Abraham Lincoln")); // match
            double emScore   = (emParis + em42 + emLincoln) / 3.0;
            System.out.println("\n  Exact Match example:");
            System.out.println("    score(\"Paris\",           [\"Paris\"]):           " + emParis);
            System.out.println("    score(\"42\",              [\"43\"]):              " + em42);
            System.out.println("    score(\"Abraham Lincoln\", [\"Abraham Lincoln\"]): " + emLincoln);
            System.out.println("    Average EM over 3 samples: " + emScore + "  (2/3 correct)");
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

            // Construct benchmark tasks — each knows its dataset, prompt format, and metric
            BenchmarkTask[] benchmarks = {
                    new MMLUBenchmark(),
                    new ArcBenchmark(),
                    new Gsm8kBenchmark(),
                    new HellaSwagBenchmark(),
                    new TruthfulQABenchmark(),
                    new WinograndeBenchmark()
            };

            // EvalConfig controls shared settings across all benchmarks
            EvalConfig fullConfig = EvalConfig.builder()
                    .numFewShot(5)
                    .maxSamples(100)
                    .maxNewTokens(256)
                    .batchSize(4)
                    .build();

            System.out.println("    Configured " + benchmarks.length + " benchmarks");
            for (BenchmarkTask b : benchmarks) {
                System.out.println("      " + b.name() + " — metric: " + b.primaryMetric().name()
                        + ", defaultFewShot=" + b.defaultFewShot());
            }
            System.out.println("    numFewShot=" + fullConfig.getNumFewShot()
                    + " maxSamples=" + fullConfig.getMaxSamples());
            System.out.println("    // EvalRunner.run(benchmarks, pipeline, fullConfig);");

            System.out.println("\n  Step 2: Perplexity on WikiText-2");
            // PerplexityEvaluator provides static methods for computing perplexity
            // on any text corpus given a SameDiff model and tokenizer.
            // PerplexityEvaluator.evaluate(model, tokenizer, text, strideLen, maxSeqLen)
            // PerplexityEvaluator.evaluateWikiText2(model, tokenizer, stride, maxSeq)
            System.out.println("    PerplexityEvaluator.evaluate(model, tokenizer, text, 512, 2048)");
            System.out.println("    PerplexityEvaluator.evaluateWikiText2(model, tokenizer, 512, 2048)");

            System.out.println("\n  Step 3: Generation metrics (e.g., CNN/DailyMail summarization)");
            RougeMetric rouge2 = new RougeMetric(RougeMetric.RougeType.ROUGE_2);
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

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

package org.nd4j.examples.samediff.quickstart.pipeline;

import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.TokenizerFactory;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Standalone Tokenizer Usage — API Reference
 *
 * This example demonstrates every major operation on HuggingFaceTokenizer,
 * the native JVM tokenizer backed by the Rust `tokenizers` crate via JNI.
 * It supports all HuggingFace tokenizer variants:
 *   - BPE (Byte-Pair Encoding) — GPT-2, Qwen, LLaMA, Mistral
 *   - WordPiece — BERT, DistilBERT
 *   - SentencePiece (BPE/Unigram) — LLaMA, Gemma, T5
 *
 * Key classes:
 *   - {@link HuggingFaceTokenizer} — Concrete tokenizer implementation
 *   - {@link Tokenizer}            — Interface with encode/decode/vocab operations
 *   - {@link Encoding}             — Result of encode(): token IDs, tokens, attention mask
 *   - {@link TokenizerFactory}     — Registry for named tokenizer instances
 *   - {@link ChatTemplate}         — Applies Jinja2-style chat template to messages
 *
 * Loading sources:
 *   - tokenizer.json (HuggingFace format, full vocabulary + merges + special tokens)
 *   - tokenizer directory (searches for tokenizer.json + tokenizer_config.json)
 *   - GGUF file (extracts tokenizer metadata embedded in the GGUF)
 *   - Raw JSON string (for programmatic construction)
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.pipeline.TokenizerExample"
 */
public class TokenizerExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. LOADING A TOKENIZER
        // ============================================================
        System.out.println("=== 1. Loading a Tokenizer ===");
        System.out.println();

        // Three loading methods — use whichever fits your setup:

        // (a) From a tokenizer.json file (most common when you have a HuggingFace checkout)
        //     The file must contain the full vocabulary, merge rules, and special token map.
        System.out.println("  (a) From tokenizer.json:");
        System.out.println("    Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(\"path/to/tokenizer.json\");");
        System.out.println();

        // (b) From a directory that contains tokenizer.json (+ optional tokenizer_config.json).
        //     Equivalent to (a) but searches the directory automatically.
        System.out.println("  (b) From a directory:");
        System.out.println("    Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(new File(\"path/to/model-dir\"));");
        System.out.println();

        // (c) From a GGUF model file.
        //     GGUF files embed tokenizer vocabularies in their metadata section.
        //     This extracts them without needing a separate tokenizer.json.
        System.out.println("  (c) From a GGUF file (extracts embedded tokenizer metadata):");
        System.out.println("    Tokenizer tokenizer = HuggingFaceTokenizer.fromGGUF(\"path/to/model.gguf\");");
        System.out.println();

        // (d) TokenizerFactory: name-based registry useful when sharing tokenizers
        //     across multiple components in a larger pipeline.
        System.out.println("  (d) Via TokenizerFactory (named registry):");
        System.out.println("    TokenizerFactory.register(\"my-tokenizer\", tokenizer);");
        System.out.println("    Tokenizer t = TokenizerFactory.get(\"my-tokenizer\");");
        System.out.println();

        // For this educational example we use a conceptual tokenizer reference.
        // In a real run, replace with an actual tokenizer.json path.
        System.out.println("  NOTE: Replace \"path/to/tokenizer.json\" with a real file to run.");
        System.out.println("  The QwenTextGenerationExample shows a complete download + load flow.");
        System.out.println();

        // ============================================================
        // 2. ENCODING — TEXT TO TOKEN IDs
        // ============================================================
        System.out.println("=== 2. Encoding text to token IDs ===");
        System.out.println();

        // tokenizer.encode(text) — basic encode, returns Encoding
        // tokenizer.encode(text, addSpecialTokens) — control BOS/EOS injection
        //
        // Encoding fields:
        //   getIds()            — int[] of token IDs (input to the model)
        //   getTokens()         — String[] of human-readable token strings
        //   getAttentionMask()  — int[] (1=real token, 0=padding)
        //   getTypeIds()        — int[] (segment IDs for BERT-style models)
        //   getOffsets()        — int[][] character offsets into the original string

        System.out.println("  // Encode with special tokens (BOS/EOS added by the tokenizer config):");
        System.out.println("  Encoding enc = tokenizer.encode(\"Hello, world!\", true);");
        System.out.println("  int[]    ids  = enc.getIds();            // e.g. [1, 15043, 29892, 3186, 29991, 2]");
        System.out.println("  String[] toks = enc.getTokens();         // e.g. [\"<s>\", \"Hello\", \",\", \"world\", \"!\", \"</s>\"]");
        System.out.println("  int[]    mask = enc.getAttentionMask();  // e.g. [1, 1, 1, 1, 1, 1]");
        System.out.println();
        System.out.println("  // Encode without special tokens (useful for sub-segment encoding):");
        System.out.println("  Encoding enc2 = tokenizer.encode(\"Hello, world!\", false);");
        System.out.println();
        System.out.println("  // Encode and pad/truncate to fixed length:");
        System.out.println("  Encoding enc3 = tokenizer.encode(\"Hello, world!\", true, 512);");
        System.out.println("  // Padding token fills positions beyond the real sequence;");
        System.out.println("  // attention mask marks them as 0.");
        System.out.println();

        // Demonstrate what the results look like conceptually:
        System.out.println("  Conceptual output for 'Hello, world!':");
        System.out.println("    ids:   [1, 15043, 29892, 3186, 29991, 2]");
        System.out.println("    tokens:[\"<s>\", \"Hello\", \",\", \"world\", \"!\", \"</s>\"]");
        System.out.println("    mask:  [1, 1, 1, 1, 1, 1]");
        System.out.println();

        // ============================================================
        // 3. DECODING — TOKEN IDs BACK TO TEXT
        // ============================================================
        System.out.println("=== 3. Decoding token IDs back to text ===");
        System.out.println();

        // tokenizer.decode(int[] ids) — basic decode
        // tokenizer.decode(int[] ids, boolean skipSpecialTokens) — filter out <s>, </s>, etc.

        System.out.println("  int[] ids = {1, 15043, 29892, 3186, 29991, 2};");
        System.out.println();
        System.out.println("  // Decode including special tokens:");
        System.out.println("  String text1 = tokenizer.decode(ids, false);");
        System.out.println("  // -> \"<s> Hello, world!</s>\"");
        System.out.println();
        System.out.println("  // Decode skipping special tokens (typical for generation output):");
        System.out.println("  String text2 = tokenizer.decode(ids, true);");
        System.out.println("  // -> \"Hello, world!\"");
        System.out.println();

        // ============================================================
        // 4. VOCABULARY OPERATIONS
        // ============================================================
        System.out.println("=== 4. Vocabulary operations ===");
        System.out.println();

        // getVocabSize()  — total number of tokens in the vocabulary
        // idToToken(id)   — convert a single ID to its string representation
        // tokenToId(tok)  — convert a token string to its ID (-1 if unknown)

        System.out.println("  int   vocabSize = tokenizer.getVocabSize();");
        System.out.println("  // Qwen3.5 vocab: 151936, LLaMA-3: 128256, GPT-2: 50257");
        System.out.println();
        System.out.println("  String tok = tokenizer.idToToken(15043);   // -> \"Hello\"");
        System.out.println("  int    id  = tokenizer.tokenToId(\"Hello\"); // -> 15043");
        System.out.println("  int    unk = tokenizer.tokenToId(\"<XYZZY_NOT_A_TOKEN>\"); // -> -1");
        System.out.println();

        // Special token IDs:
        System.out.println("  // Special token accessors:");
        System.out.println("  int bosId = tokenizer.getBosTokenId();  // beginning-of-sequence");
        System.out.println("  int eosId = tokenizer.getEosTokenId();  // end-of-sequence");
        System.out.println("  int padId = tokenizer.getPadTokenId();  // padding token");
        System.out.println("  int unkId = tokenizer.getUnkTokenId();  // unknown token");
        System.out.println();
        System.out.println("  String bosStr = tokenizer.getBosToken();  // e.g. \"<s>\" or \"<|begin_of_text|>\"");
        System.out.println("  String eosStr = tokenizer.getEosToken();  // e.g. \"</s>\" or \"<|end_of_text|>\"");
        System.out.println();

        // ============================================================
        // 5. BATCH ENCODING
        // ============================================================
        System.out.println("=== 5. Batch encoding ===");
        System.out.println();

        // Batch encoding is more efficient than encoding strings one at a time
        // because the native Rust tokenizer parallelizes across tokens.
        // All outputs are padded/truncated to the same length for easy batching.

        System.out.println("  List<String> texts = Arrays.asList(");
        System.out.println("      \"The capital of France is Paris.\",");
        System.out.println("      \"Deep learning is a subfield of machine learning.\",");
        System.out.println("      \"SameDiff provides automatic differentiation.\"");
        System.out.println("  );");
        System.out.println();
        System.out.println("  // Returns List<Encoding>, one per input string.");
        System.out.println("  // All Encodings are padded to the same length (longest sequence).");
        System.out.println("  List<Encoding> batch = tokenizer.encodeBatch(texts);");
        System.out.println();
        System.out.println("  for (int i = 0; i < batch.size(); i++) {");
        System.out.println("      Encoding e = batch.get(i);");
        System.out.println("      System.out.println(\"Text \" + i + \": \" + e.getIds().length + \" tokens\");");
        System.out.println("  }");
        System.out.println();
        System.out.println("  // Batch encode with explicit max length (pads shorter, truncates longer):");
        System.out.println("  List<Encoding> padded = tokenizer.encodeBatch(texts, true, 128);");
        System.out.println();

        // ============================================================
        // 6. CHAT TEMPLATE
        // ============================================================
        System.out.println("=== 6. Chat template formatting ===");
        System.out.println();

        // Chat templates convert a list of role-tagged messages into a
        // single prompt string formatted for the specific model's instruction
        // format (ChatML, LLaMA-3 instruct, Gemma-it, etc.).
        // The template is stored in tokenizer_config.json as a Jinja2 string.

        System.out.println("  // Build a conversation:");
        System.out.println("  List<ChatTemplate.Message> messages = Arrays.asList(");
        System.out.println("      ChatTemplate.Message.system(\"You are a helpful AI assistant.\"),");
        System.out.println("      ChatTemplate.Message.user(\"What is 2 + 2?\"),");
        System.out.println("      ChatTemplate.Message.assistant(\"2 + 2 equals 4.\"),");
        System.out.println("      ChatTemplate.Message.user(\"And what is 3 + 3?\")");
        System.out.println("  );");
        System.out.println();
        System.out.println("  // Apply the chat template (addGenerationPrompt=true appends the");
        System.out.println("  // assistant turn opener so the model starts generating immediately):");
        System.out.println("  String prompt = tokenizer.applyChatTemplate(messages, true);");
        System.out.println();
        System.out.println("  // For ChatML format (used by Qwen, Mistral, many others):");
        System.out.println("  // <|im_start|>system");
        System.out.println("  // You are a helpful AI assistant.<|im_end|>");
        System.out.println("  // <|im_start|>user");
        System.out.println("  // What is 2 + 2?<|im_end|>");
        System.out.println("  // <|im_start|>assistant");
        System.out.println("  // 2 + 2 equals 4.<|im_end|>");
        System.out.println("  // <|im_start|>user");
        System.out.println("  // And what is 3 + 3?<|im_end|>");
        System.out.println("  // <|im_start|>assistant    <-- generation starts here");
        System.out.println();
        System.out.println("  // For LLaMA-3 instruct format:");
        System.out.println("  // <|begin_of_text|><|start_header_id|>system<|end_header_id|>");
        System.out.println("  // ...");
        System.out.println("  // <|start_header_id|>assistant<|end_header_id|>");
        System.out.println();
        System.out.println("  // The format is automatically selected from tokenizer_config.json.");
        System.out.println("  System.out.println(\"Formatted prompt: \" + prompt);");
        System.out.println();

        // ============================================================
        // 7. SPECIAL TOKENS AND CONFIGURATION
        // ============================================================
        System.out.println("=== 7. Special tokens and tokenizer configuration ===");
        System.out.println();

        System.out.println("  // Inspect added/special tokens:");
        System.out.println("  List<String> addedTokens = tokenizer.getAddedTokens();");
        System.out.println("  // e.g. [\"<|endoftext|>\", \"<|im_start|>\", \"<|im_end|>\", ...]");
        System.out.println();
        System.out.println("  // Check if an ID corresponds to a special token:");
        System.out.println("  boolean isSpecial = tokenizer.isSpecialTokenId(bosId);  // true");
        System.out.println();
        System.out.println("  // Tokenizer type:");
        System.out.println("  String tokType = tokenizer.getTokenizerType();");
        System.out.println("  // -> \"BPE\", \"WordPiece\", or \"Unigram\"");
        System.out.println();
        System.out.println("  // Chat template string (raw Jinja2 template from tokenizer_config.json):");
        System.out.println("  String template = tokenizer.getChatTemplate();");
        System.out.println("  System.out.println(\"Has chat template: \" + (template != null));");
        System.out.println();

        // ============================================================
        // 8. RESOURCE CLEANUP
        // ============================================================
        System.out.println("=== 8. Resource cleanup ===");
        System.out.println();

        // Tokenizer holds a native (Rust/JNI) resource that must be closed.
        // Use try-with-resources or call close() explicitly.
        System.out.println("  // HuggingFaceTokenizer implements AutoCloseable:");
        System.out.println("  try (Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(\"tokenizer.json\")) {");
        System.out.println("      Encoding enc = tokenizer.encode(\"Hello\", true);");
        System.out.println("      // ... use tokenizer ...");
        System.out.println("  }  // native resources freed here");
        System.out.println();

        System.out.println("Tokenizer API reference example completed.");
        System.out.println("Use QwenTextGenerationExample for a runnable end-to-end example");
        System.out.println("that downloads a real model and runs a complete tokenize→generate→decode pipeline.");
    }
}

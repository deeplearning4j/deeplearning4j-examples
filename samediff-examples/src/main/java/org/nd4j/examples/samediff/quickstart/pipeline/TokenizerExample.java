/*
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.nd4j.examples.samediff.quickstart.pipeline;

import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

/**
 * Demonstrates HuggingFaceTokenizer by creating a minimal BPE tokenizer from
 * a JSON string, then exercising encode, decode, vocab lookup, batch encoding,
 * and chat template formatting with real method calls and real output.
 */
public class TokenizerExample {

    // A minimal valid BPE tokenizer.json with a small vocabulary.
    // This lets the example run without downloading any model files.
    private static final String TOKENIZER_JSON =
            "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null," +
            "\"added_tokens\":[" +
            "{\"id\":0,\"content\":\"<s>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}," +
            "{\"id\":1,\"content\":\"</s>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}" +
            "]," +
            "\"normalizer\":null," +
            "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true}," +
            "\"post_processor\":null," +
            "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true}," +
            "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null," +
            "\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null," +
            "\"fuse_unk\":false,\"byte_fallback\":false," +
            "\"vocab\":{\"<s>\":0,\"</s>\":1," +
            "\"H\":2,\"e\":3,\"l\":4,\"o\":5,\" \":6,\"w\":7,\"r\":8,\"d\":9,\"!\":10," +
            "\"he\":11,\"ll\":12,\"wo\":13,\"rld\":14}," +
            "\"merges\":[\"h e\",\"l l\",\"w o\",\"r l d\"]}}";

    public static void main(String[] args) throws Exception {

        // ================================================================
        // 1. Load tokenizer from JSON string via temp file
        // ================================================================
        System.out.println("=== 1. Load tokenizer ===");

        File tmpFile = File.createTempFile("tokenizer", ".json");
        tmpFile.deleteOnExit();
        Files.write(tmpFile.toPath(), TOKENIZER_JSON.getBytes(StandardCharsets.UTF_8));

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tmpFile.getAbsolutePath());
        System.out.println("  Tokenizer loaded from: " + tmpFile.getName());
        System.out.println("  Vocab size: " + tokenizer.getVocabSize());

        // ================================================================
        // 2. Encode text to token IDs
        // ================================================================
        System.out.println("\n=== 2. Encode text ===");

        String text = "Hello world!";
        Encoding enc = tokenizer.encode(text, false);

        System.out.println("  Text:           \"" + text + "\"");
        System.out.println("  Token IDs:      " + Arrays.toString(enc.getIds()));
        System.out.println("  Tokens:         " + Arrays.toString(enc.getTokens()));
        System.out.println("  Attention mask: " + Arrays.toString(enc.getAttentionMask()));
        System.out.println("  Num tokens:     " + enc.getIds().length);

        // Encode with special tokens
        Encoding encSpecial = tokenizer.encode(text, true);
        System.out.println("  With special tokens: " + Arrays.toString(encSpecial.getIds()));

        // ================================================================
        // 3. Decode token IDs back to text
        // ================================================================
        System.out.println("\n=== 3. Decode ===");

        int[] ids = enc.getIds();
        String decoded = tokenizer.decode(ids, false);
        String decodedSkip = tokenizer.decode(ids, true);

        System.out.println("  IDs:                    " + Arrays.toString(ids));
        System.out.println("  Decoded (with special): \"" + decoded + "\"");
        System.out.println("  Decoded (skip special): \"" + decodedSkip + "\"");

        // ================================================================
        // 4. Vocabulary operations
        // ================================================================
        System.out.println("\n=== 4. Vocab operations ===");

        System.out.println("  Total vocab size: " + tokenizer.getVocabSize());

        // Look up individual tokens
        for (int id = 0; id < Math.min(15, tokenizer.getVocabSize()); id++) {
            String token = tokenizer.idToToken(id);
            int backId = tokenizer.tokenToId(token);
            System.out.println("  ID " + id + " -> \"" + token + "\" -> ID " + backId);
        }

        // Unknown token
        int unkId = tokenizer.tokenToId("DOES_NOT_EXIST");
        System.out.println("  Unknown token ID: " + unkId);

        // Special tokens
        System.out.println("  BOS token ID: " + tokenizer.getBosTokenId());
        System.out.println("  EOS token ID: " + tokenizer.getEosTokenId());

        // ================================================================
        // 5. Batch encoding
        // ================================================================
        System.out.println("\n=== 5. Batch encoding ===");

        List<String> texts = Arrays.asList(
                "Hello",
                "Hello world",
                "Hello world!"
        );

        List<Encoding> batch = tokenizer.encodeBatch(texts, false);
        for (int i = 0; i < batch.size(); i++) {
            Encoding e = batch.get(i);
            System.out.println("  \"" + texts.get(i) + "\" -> " +
                    e.getIds().length + " tokens: " + Arrays.toString(e.getIds()));
        }

        // ================================================================
        // 6. Chat template formatting
        // ================================================================
        System.out.println("\n=== 6. Chat templates ===");

        // ChatML format (used by Qwen, Mistral, many others)
        ChatTemplate chatML = ChatTemplate.chatML();
        List<ChatTemplate.Message> messages = Arrays.asList(
                ChatTemplate.Message.system("You are a helpful assistant."),
                ChatTemplate.Message.user("What is 2 + 2?"),
                ChatTemplate.Message.assistant("4."),
                ChatTemplate.Message.user("And 3 + 3?")
        );

        String formatted = chatML.apply(messages, true);
        System.out.println("  ChatML format:");
        System.out.println(formatted);

        // LLaMA-2 format
        ChatTemplate llama2 = ChatTemplate.llama2();
        String llama2Formatted = llama2.apply(messages, true);
        System.out.println("  LLaMA-2 format:");
        System.out.println(llama2Formatted);

        // ================================================================
        // 7. Multiple encodings comparison
        // ================================================================
        System.out.println("=== 7. Encoding comparison ===");

        String[] testTexts = {"Hello", "Hello world", "Hello world!", "Hello Hello Hello"};
        for (String t : testTexts) {
            Encoding e = tokenizer.encode(t, false);
            System.out.printf("  %-20s -> %d tokens%n", "\"" + t + "\"", e.getIds().length);
        }

        // ================================================================
        // 8. Cleanup
        // ================================================================
        System.out.println("\n=== 8. Cleanup ===");
        tokenizer.close();
        System.out.println("  Tokenizer closed (native resources freed).");

        System.out.println("\nTokenizerExample complete.");
    }
}

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

package org.nd4j.examples.samediff.quickstart.modeling.audio;

import org.eclipse.deeplearning4j.audio.whisper.WhisperModel;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader.WhisperModelSize;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader.WhisperModelFormat;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.audio.whisper.WhisperConfig;
import org.eclipse.deeplearning4j.audio.whisper.WhisperDecoderResult;
import org.eclipse.deeplearning4j.audio.whisper.WhisperTokenizer;
import org.eclipse.deeplearning4j.audio.feature.WhisperMelSpectrogram;
import org.eclipse.deeplearning4j.audio.feature.AudioFeatureExtractor;
import org.eclipse.deeplearning4j.audio.io.AudioLoader;
import org.eclipse.deeplearning4j.audio.transform.AudioPreprocessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Whisper Speech-to-Text — Complete Example
 *
 * Demonstrates the full pipeline for running OpenAI's Whisper automatic speech
 * recognition (ASR) model via the DL4J audio stack:
 *
 *   1.  Download a Whisper ONNX model (tiny) via WhisperModelDownloader
 *   2.  Survey all WhisperConfig presets and their architecture parameters
 *   3.  Load the WhisperModel from the downloaded artifacts
 *   4.  Transcribe audio (file-based and raw INDArray-based APIs)
 *   5.  Use WhisperTokenizer for encoding, decoding, and special-token inspection
 *   6.  Extract Mel-spectrogram features with WhisperMelSpectrogram
 *   7.  Preprocess audio with AudioPreprocessor (resampling, normalization)
 *   8.  Extract general audio features with AudioFeatureExtractor (MFCC, Mel, log-Mel)
 *   9.  Load raw audio files with AudioLoader
 *   10. Inspect WhisperDecoderResult (text, language, timestamped segments)
 *
 * Key classes:
 *   - {@link WhisperModelDownloader} — Downloads Whisper ONNX weights from the DL4J model hub
 *   - {@link WhisperConfig}         — Architecture configuration for each model size variant
 *   - {@link WhisperModel}          — End-to-end ASR inference (encoder + decoder + tokenizer)
 *   - {@link WhisperTokenizer}      — BPE tokenizer with Whisper-specific special tokens
 *   - {@link WhisperMelSpectrogram} — Log-Mel spectrogram extraction (80 bins, 3000 frames)
 *   - {@link AudioPreprocessor}     — Resampling, normalization, pre-emphasis filtering
 *   - {@link AudioFeatureExtractor} — MFCC, Mel-spectrogram, log-Mel-spectrogram extraction
 *   - {@link AudioLoader}           — WAV file loading (returns raw float samples as INDArray)
 *   - {@link WhisperDecoderResult}  — Transcription output with optional timestamps
 *
 * Model requirements:
 *   - Tiny ONNX model is ~150MB total (encoder + decoder)
 *   - Models are cached in ~/.cache/dl4j-whisper-models/ by default
 *   - A real audio WAV file at 16 kHz is needed for live transcription
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.WhisperSpeechToTextExample"
 *
 * For GPU acceleration, use the nd4j-cuda backend:
 *   mvn exec:java -Dexec.mainClass="..." -Dnd4j.backend=nd4j-cuda-12.9-platform
 */
public class WhisperSpeechToTextExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. MODEL DOWNLOAD
        // ============================================================
        System.out.println("=== 1. Downloading Whisper Tiny ONNX Model ===");

        // WhisperModelDownloader fetches the encoder and decoder ONNX files from the
        // DL4J model hub and caches them locally. Subsequent runs skip the download.
        //
        // Available model sizes (in order of increasing accuracy and resource use):
        //   WhisperModelSize.TINY    — ~150MB total, fastest, least accurate
        //   WhisperModelSize.BASE    — ~290MB total
        //   WhisperModelSize.SMALL   — ~970MB total
        //   WhisperModelSize.MEDIUM  — ~3.0GB total
        //   WhisperModelSize.LARGE_V2 / LARGE_V3 — ~6.2GB total, most accurate
        //   WhisperModelSize.TURBO   — distilled large-v3, ~1.6GB, fast + accurate
        //
        // Available formats:
        //   WhisperModelFormat.ONNX  — ONNX runtime (CPU/CUDA)
        //   WhisperModelFormat.GGUF  — quantized GGUF (smaller, for CPU inference)
        WhisperModelDownloader downloader = new WhisperModelDownloader();
        DownloadResult downloadResult = downloader.downloadOnnx(WhisperModelSize.TINY);

        System.out.println("  Model directory:    " + downloadResult.getModelDir().getAbsolutePath());
        System.out.println("  Model size variant: " + downloadResult.getModelSize());
        System.out.println("  Model format:       " + downloadResult.getFormat());
        System.out.println("  Config available:   " + (downloadResult.getConfig() != null));

        // The DownloadResult contains the WhisperConfig that matches the downloaded variant
        WhisperConfig downloadedConfig = downloadResult.getConfig();
        System.out.println("  Config model name:  " + downloadedConfig.getModelName());

        // ============================================================
        // 2. WHISPER CONFIG REFERENCE
        // ============================================================
        System.out.println("\n=== 2. WhisperConfig Presets ===");

        // WhisperConfig encodes the architecture hyperparameters for every supported
        // Whisper model variant. Each preset matches the corresponding OpenAI checkpoint.
        WhisperConfig[] configs = {
                WhisperConfig.tiny(),
                WhisperConfig.base(),
                WhisperConfig.small(),
                WhisperConfig.medium(),
                WhisperConfig.largeV2(),
                WhisperConfig.largeV3(),
                WhisperConfig.turbo()
        };

        System.out.printf("  %-12s %8s %8s %10s %8s %8s%n",
                "Name", "EncLayers", "DecLayers", "HiddenSize", "Heads", "MelBins");
        System.out.println("  " + "-".repeat(62));
        for (WhisperConfig cfg : configs) {
            System.out.printf("  %-12s %8d %8d %10d %8d %8d%n",
                    cfg.getModelName(),
                    cfg.getNumEncoderLayers(),
                    cfg.getNumDecoderLayers(),
                    cfg.getHiddenSize(),
                    cfg.getNumAttentionHeads(),
                    cfg.getNumMelBins());
        }

        // Audio framing constants — the same for every model size
        WhisperConfig tiny = WhisperConfig.tiny();
        System.out.println();
        System.out.println("  Audio constants (shared across all model sizes):");
        System.out.println("    sampleRate          = " + tiny.getSampleRate()
                + " Hz  (Whisper always expects 16 kHz input)");
        System.out.println("    nFft                = " + tiny.getNFft()
                + "     (STFT window size in samples)");
        System.out.println("    hopLength           = " + tiny.getHopLength()
                + "     (STFT hop size in samples → 10 ms per frame)");
        System.out.println("    chunkLengthSeconds  = " + tiny.getChunkLengthSeconds()
                + "    (maximum audio chunk length)");
        System.out.println("    getNumFrames()      = " + tiny.getNumFrames()
                + "   (time frames in the Mel spectrogram: 30s / 10ms)");
        System.out.println("    getChunkLengthSamples() = " + tiny.getChunkLengthSamples()
                + " (30 s × 16000 Hz)");

        // ============================================================
        // 3. LOAD WHISPER MODEL
        // ============================================================
        System.out.println("\n=== 3. Loading Whisper Model ===");

        // WhisperModel.fromDownload() uses the already-cached artifacts produced by
        // WhisperModelDownloader.  It constructs and wires the ONNX encoder, ONNX
        // decoder, and WhisperTokenizer internally.
        WhisperModel model = WhisperModel.fromDownload(WhisperModelSize.TINY, WhisperModelFormat.ONNX);

        System.out.println("  Model loaded successfully.");
        System.out.println("  Encoder:   Whisper ONNX audio encoder (Mel → hidden states)");
        System.out.println("  Decoder:   Whisper ONNX autoregressive decoder (tokens → tokens)");
        System.out.println("  Tokenizer: WhisperTokenizer (multilingual BPE, 50,257 text tokens)");

        // ============================================================
        // 4. TRANSCRIBE AUDIO
        // ============================================================
        System.out.println("\n=== 4. Transcription API ===");

        // -- File-based transcription API --
        //
        // The simplest entry point: pass a WAV file and get back a WhisperDecoderResult.
        // Whisper handles Mel extraction, chunking, and language detection internally.
        //
        //   WhisperDecoderResult result = model.transcribe(audioFile);
        //
        // With explicit language, task, and timestamp control:
        //
        //   // Language code "en", task "transcribe", no word timestamps
        //   WhisperDecoderResult result = model.transcribe(audioFile, "en", "transcribe", false);
        //
        //   // Language code "fr", task "translate" (translate to English), with timestamps
        //   WhisperDecoderResult result = model.transcribe(audioFile, "fr", "translate", true);
        //
        // Supported tasks:
        //   "transcribe" — output in the source language
        //   "translate"  — transcribe and translate to English

        System.out.println("  File-based transcription signatures:");
        System.out.println("    model.transcribe(File audioFile)");
        System.out.println("      -> auto-detect language, transcribe, no timestamps");
        System.out.println("    model.transcribe(File audioFile, String language, String task, boolean timestamps)");
        System.out.println("      -> e.g. model.transcribe(file, \"en\", \"transcribe\", false)");
        System.out.println("      -> e.g. model.transcribe(file, \"fr\", \"translate\",  true)");
        System.out.println("    Supported tasks: \"transcribe\", \"translate\"");

        // -- INDArray-based transcription API (synthetic audio demo) --
        //
        // When you already have audio samples as an INDArray (e.g., from AudioLoader or
        // a custom pipeline), use the array-based overloads.  The array must be a 1-D
        // float array of raw PCM samples; the sampleRate tells Whisper whether resampling
        // is needed before feature extraction.
        System.out.println();
        System.out.println("  INDArray-based transcription (synthetic audio demo):");

        // Create 5 seconds of synthetic white-noise audio at 16 kHz
        INDArray syntheticAudio = Nd4j.rand(DataType.FLOAT, 16000 * 5);   // shape [80000]
        System.out.println("    Synthetic audio shape: " + Arrays.toString(syntheticAudio.shape())
                + "  (" + (16000 * 5 / 16000) + " seconds @ 16 kHz)");

        // Simple INDArray transcription (auto-detect language, no timestamps)
        System.out.println("    Signature: model.transcribe(INDArray audio, int sampleRate)");
        System.out.println("    Example:   model.transcribe(syntheticAudio, 16000)");

        // INDArray transcription with full control
        System.out.println("    Signature: model.transcribe(INDArray audio, int sampleRate,");
        System.out.println("                                String language, String task, boolean timestamps)");
        System.out.println("    Example:   model.transcribe(syntheticAudio, 16000, \"en\", \"transcribe\", true)");

        // NOTE: We do not call transcribe() on the synthetic noise here because the
        // result would be meaningless gibberish.  In a real application you would do:
        //
        //   File wavFile = new File("/path/to/audio.wav");
        //   WhisperDecoderResult result = model.transcribe(wavFile, "en", "transcribe", false);
        //   System.out.println(result.getText());

        // ============================================================
        // 5. WHISPER TOKENIZER
        // ============================================================
        System.out.println("\n=== 5. WhisperTokenizer ===");

        // The tokenizer is obtained from the loaded model — it has no public constructor.
        // (WhisperTokenizer.fromFile(File) and WhisperTokenizer.fromDirectory(File) also
        //  exist for loading from disk, but the model-attached instance is preferred.)
        WhisperTokenizer tokenizer = model.getTokenizer();

        // Whisper uses a multilingual BPE vocabulary with special control tokens.
        // The most important special token IDs are defined as public constants:
        System.out.println("  Special token constants:");
        System.out.println("    WhisperTokenizer.SOT              = " + WhisperTokenizer.SOT
                + "  (start-of-transcript)");
        System.out.println("    WhisperTokenizer.EOT              = " + WhisperTokenizer.EOT
                + "  (end-of-transcript / EOS)");
        System.out.println("    WhisperTokenizer.TRANSCRIBE       = " + WhisperTokenizer.TRANSCRIBE
                + "  (task: transcribe in source language)");
        System.out.println("    WhisperTokenizer.TRANSLATE        = " + WhisperTokenizer.TRANSLATE
                + "  (task: translate to English)");
        System.out.println("    WhisperTokenizer.NO_TIMESTAMPS    = " + WhisperTokenizer.NO_TIMESTAMPS
                + "  (suppress timestamp tokens)");
        System.out.println("    WhisperTokenizer.TIMESTAMP_BEGIN  = " + WhisperTokenizer.TIMESTAMP_BEGIN
                + "  (first timestamp token; each +1 = +20 ms)");

        // createPromptTokens() builds the decoder prompt token sequence that encodes
        // the desired language and task before the model starts generating text.
        int[] promptNoTs = tokenizer.createPromptTokens("en", "transcribe", false);
        int[] promptWithTs = tokenizer.createPromptTokens("en", "transcribe", true);
        System.out.println();
        System.out.println("  createPromptTokens(\"en\", \"transcribe\", false): "
                + Arrays.toString(promptNoTs));
        System.out.println("  createPromptTokens(\"en\", \"transcribe\", true):  "
                + Arrays.toString(promptWithTs));

        // Encode / decode plain text
        int[] ids = tokenizer.encodeToIds("Hello world");
        System.out.println();
        System.out.println("  encodeToIds(\"Hello world\"): " + Arrays.toString(ids));

        String decoded = tokenizer.decodeSkippingSpecial(ids);
        System.out.println("  decodeSkippingSpecial(ids): \"" + decoded + "\"");

        // Timestamp helpers
        System.out.println();
        System.out.println("  Timestamp token utilities:");
        System.out.println("    tokenizer.isTimestampToken(50364): "
                + tokenizer.isTimestampToken(WhisperTokenizer.TIMESTAMP_BEGIN));
        System.out.println("    tokenizer.isTimestampToken(100):   "
                + tokenizer.isTimestampToken(100));
        // Each timestamp token represents a 20 ms interval starting from TIMESTAMP_BEGIN.
        // Token 50364 -> 0.00 s, 50365 -> 0.02 s, 50366 -> 0.04 s, …
        System.out.println("    tokenizer.timestampToSeconds(" + WhisperTokenizer.TIMESTAMP_BEGIN + "): "
                + tokenizer.timestampToSeconds(WhisperTokenizer.TIMESTAMP_BEGIN) + " s");
        System.out.println("    tokenizer.timestampToSeconds(" + (WhisperTokenizer.TIMESTAMP_BEGIN + 50) + "): "
                + tokenizer.timestampToSeconds(WhisperTokenizer.TIMESTAMP_BEGIN + 50) + " s");

        // Language support
        java.util.Set<String> languages = WhisperTokenizer.getSupportedLanguages();
        System.out.println();
        System.out.println("  getSupportedLanguages() -> " + languages.size() + " languages");
        System.out.println("  First 10: " + languages.stream().limit(10).collect(java.util.stream.Collectors.toList()));

        // ============================================================
        // 6. MEL SPECTROGRAM FEATURES
        // ============================================================
        System.out.println("\n=== 6. WhisperMelSpectrogram ===");

        // WhisperMelSpectrogram implements the exact log-Mel feature extraction that
        // Whisper expects.  It is initialized from a WhisperConfig so that the number
        // of Mel bins, FFT size, and hop length all match the chosen model variant.
        WhisperMelSpectrogram melSpec = new WhisperMelSpectrogram(WhisperConfig.tiny());

        // The standard Whisper input is exactly 30 seconds of 16 kHz audio.
        // WhisperConfig.getChunkLengthSamples() = 480,000 samples.
        INDArray audioChunk = Nd4j.rand(DataType.FLOAT, (int) WhisperConfig.tiny().getChunkLengthSamples());
        System.out.println("  Input audio chunk shape: " + Arrays.toString(audioChunk.shape())
                + "  (" + WhisperConfig.tiny().getChunkLengthSamples() + " samples = 30 s)");

        // extractFeatures() returns the log-Mel spectrogram as [1, numMelBins, numFrames]
        // = [1, 80, 3000] for the tiny model.
        INDArray melFeatures = melSpec.extractFeatures(audioChunk);
        System.out.println("  extractFeatures() output shape: " + Arrays.toString(melFeatures.shape())
                + "  (batch=1, melBins=80, frames=3000)");

        // Once you have the Mel features you can bypass Whisper's internal feature
        // extraction and feed them directly to the encoder via:
        System.out.println();
        System.out.println("  Direct Mel-input transcription signature:");
        System.out.println("    model.transcribeMel(INDArray melFeatures)");
        System.out.println("    -> melFeatures shape must be [1, numMelBins, numFrames]");
        System.out.println("    -> e.g. model.transcribeMel(melFeatures)  // from above extraction");

        // ============================================================
        // 7. AUDIO PREPROCESSING
        // ============================================================
        System.out.println("\n=== 7. AudioPreprocessor ===");

        // AudioPreprocessor handles the common signal-processing steps that should be
        // applied before feeding audio to Whisper (or any other ASR/audio model):
        //   - Resampling: convert from any sample rate to 16 kHz
        //   - Normalization: scale peak amplitude to [-1, 1]
        //   - Pre-emphasis: apply a high-pass filter to boost high-frequency content
        AudioPreprocessor preprocessor = AudioPreprocessor.builder()
                .targetSampleRate(16000)         // resample to 16 kHz (Whisper requirement)
                .normalize(true)                  // peak-normalize to [-1, 1]
                .applyPreEmphasis(true)           // enable first-order high-pass filter
                .preEmphasisCoeff(0.97)           // standard pre-emphasis coefficient
                .build();

        // Simulate audio captured at 44.1 kHz (CD quality), e.g., from a microphone
        INDArray audio44kHz = Nd4j.rand(DataType.FLOAT, 44100 * 5);   // 5 s at 44.1 kHz
        System.out.println("  Input audio (44.1 kHz) shape: " + Arrays.toString(audio44kHz.shape()));

        // process() resamples from 44100 Hz → 16000 Hz and applies normalization +
        // pre-emphasis.  The output length is approximately 5 × 16000 = 80,000 samples.
        INDArray processed = preprocessor.process(audio44kHz, 44100);
        System.out.println("  Processed audio (16 kHz) shape: " + Arrays.toString(processed.shape()));
        System.out.println("  (resampled from 44100 Hz to 16000 Hz, normalized, pre-emphasis applied)");

        // padOrTrim() is a static helper that zero-pads or truncates an audio array so
        // its length equals exactly the given target (here: one 30-second Whisper chunk).
        System.out.println();
        System.out.println("  Static utility:");
        System.out.println("    AudioPreprocessor.padOrTrim(audio, 480000)");
        System.out.println("      -> pads with silence if audio.length() < 480000");
        System.out.println("      -> trims from the end if audio.length() > 480000");
        System.out.println("      -> returns exactly [480000] float array");

        INDArray paddedAudio = AudioPreprocessor.padOrTrim(processed, 480000);
        System.out.println("  padOrTrim result shape: " + Arrays.toString(paddedAudio.shape()));

        // ============================================================
        // 8. AUDIO FEATURE EXTRACTOR
        // ============================================================
        System.out.println("\n=== 8. AudioFeatureExtractor ===");

        // AudioFeatureExtractor is a general-purpose audio feature library that goes
        // beyond Whisper's built-in Mel extraction.  Use it when you need MFCC features
        // or custom Mel filterbank parameters (e.g., for speaker verification, emotion
        // recognition, or music information retrieval).
        AudioFeatureExtractor extractor = AudioFeatureExtractor.builder()
                .sampleRate(16000)          // input sample rate in Hz
                .fftSize(2048)              // STFT window size (frequency resolution)
                .hopLength(512)             // STFT hop size (time resolution)
                .numMelBins(128)            // number of Mel filterbank bands
                .numMfcc(13)               // number of MFCC coefficients to keep
                .lowerEdgeHz(0.0)           // lowest Mel filterbank center frequency
                .upperEdgeHz(8000.0)        // highest Mel filterbank center frequency
                .applyPreEmphasis(true)     // pre-emphasis before feature extraction
                .preEmphasisCoeff(0.97)
                .build();

        INDArray audioForFeatures = Nd4j.rand(DataType.FLOAT, 16000 * 3);  // 3 s of audio
        System.out.println("  Input audio shape: " + Arrays.toString(audioForFeatures.shape())
                + "  (3 seconds @ 16 kHz)");

        // MFCC: [numFrames, numMfcc] — compact representation of spectral shape
        INDArray mfcc = extractor.extractMfcc(audioForFeatures);
        System.out.println();
        System.out.println("  extractMfcc() shape:              " + Arrays.toString(mfcc.shape())
                + "  [frames, 13 coefficients]");

        // Mel spectrogram: [numFrames, numMelBins] — energy in each Mel band over time
        INDArray melSpectrogram = extractor.extractMelSpectrogram(audioForFeatures);
        System.out.println("  extractMelSpectrogram() shape:    " + Arrays.toString(melSpectrogram.shape())
                + "  [frames, 128 Mel bins]");

        // Log-Mel spectrogram: same shape, values in log scale (closer to human perception)
        INDArray logMelSpectrogram = extractor.extractLogMelSpectrogram(audioForFeatures);
        System.out.println("  extractLogMelSpectrogram() shape: " + Arrays.toString(logMelSpectrogram.shape())
                + "  [frames, 128 Mel bins, log scale]");

        System.out.println();
        System.out.println("  AudioFeatureExtractor builder parameters:");
        System.out.println("    sampleRate(int)         — input sample rate in Hz (default 16000)");
        System.out.println("    fftSize(int)            — STFT window size (default 2048)");
        System.out.println("    hopLength(int)          — STFT hop size (default 512)");
        System.out.println("    numMelBins(int)         — Mel filterbank bands (default 128)");
        System.out.println("    numMfcc(int)            — MFCC coefficients to retain (default 13)");
        System.out.println("    lowerEdgeHz(double)     — lowest Mel center frequency (default 0.0)");
        System.out.println("    upperEdgeHz(double)     — highest Mel center frequency (default 8000.0)");
        System.out.println("    applyPreEmphasis(bool)  — enable pre-emphasis filter (default false)");
        System.out.println("    preEmphasisCoeff(double)— pre-emphasis filter coefficient (default 0.97)");

        // ============================================================
        // 9. AUDIO I/O REFERENCE
        // ============================================================
        System.out.println("\n=== 9. AudioLoader API ===");

        // AudioLoader provides simple utilities for reading PCM audio from disk.
        // All methods return raw float samples normalized to [-1.0, 1.0].
        System.out.println("  Loading WAV files:");
        System.out.println("    INDArray samples = AudioLoader.loadWav(new File(\"audio.wav\"))");
        System.out.println("      -> returns shape [numSamples]");
        System.out.println("      -> samples normalized to [-1.0, 1.0]");
        System.out.println();
        System.out.println("    INDArray samples = AudioLoader.loadWav(\"/path/to/audio.wav\")");
        System.out.println("      -> same as above but accepts a String path");
        System.out.println();
        System.out.println("    int rate = AudioLoader.getSampleRate(new File(\"audio.wav\"))");
        System.out.println("      -> reads only the WAV header; does not load the full file");
        System.out.println("      -> use this to check whether resampling is needed before transcription");
        System.out.println();
        System.out.println("  Typical loading + preprocessing pipeline:");
        System.out.println("    File wavFile = new File(\"/path/to/recording.wav\");");
        System.out.println("    int nativeRate = AudioLoader.getSampleRate(wavFile);");
        System.out.println("    INDArray raw     = AudioLoader.loadWav(wavFile);");
        System.out.println("    INDArray ready   = preprocessor.process(raw, nativeRate);  // -> 16 kHz");
        System.out.println("    INDArray trimmed = AudioPreprocessor.padOrTrim(ready, 480000);");
        System.out.println("    WhisperDecoderResult r = model.transcribeMel(");
        System.out.println("        new WhisperMelSpectrogram(WhisperConfig.tiny()).extractFeatures(trimmed));");

        // ============================================================
        // 10. TRANSCRIPTION RESULT
        // ============================================================
        System.out.println("\n=== 10. WhisperDecoderResult ===");

        // WhisperDecoderResult wraps the decoder's output and provides structured
        // access to the transcription text, detected language, and (optionally) the
        // timestamped word/segment list.
        System.out.println("  WhisperDecoderResult fields:");
        System.out.println();
        System.out.println("    result.getText()");
        System.out.println("      -> the full transcription as a plain String");
        System.out.println();
        System.out.println("    result.getLanguage()");
        System.out.println("      -> ISO 639-1 language code detected by Whisper (e.g. \"en\", \"fr\")");
        System.out.println("      -> always populated; set to the forced language if one was specified");
        System.out.println();
        System.out.println("    result.getSegments()");
        System.out.println("      -> List<WhisperDecoderResult.Segment>");
        System.out.println("      -> empty if transcription was called with timestamps=false");
        System.out.println("      -> each Segment has:");
        System.out.println("           segment.getStart()  -> double, segment start time in seconds");
        System.out.println("           segment.getEnd()    -> double, segment end time in seconds");
        System.out.println("           segment.getText()   -> String, transcribed text for this segment");
        System.out.println("           segment.getTokens() -> int[], raw token IDs for this segment");
        System.out.println();
        System.out.println("  Example usage with a real audio file:");
        System.out.println("    File wavFile = new File(\"/path/to/speech.wav\");");
        System.out.println("    WhisperDecoderResult result =");
        System.out.println("        model.transcribe(wavFile, \"en\", \"transcribe\", true);");
        System.out.println();
        System.out.println("    System.out.println(\"Text:     \" + result.getText());");
        System.out.println("    System.out.println(\"Language: \" + result.getLanguage());");
        System.out.println("    for (WhisperDecoderResult.Segment seg : result.getSegments()) {");
        System.out.println("        System.out.printf(\"  [%.2f -> %.2f] %s%n\",");
        System.out.println("            seg.getStart(), seg.getEnd(), seg.getText());");
        System.out.println("    }");

        // ============================================================
        // Cleanup
        // ============================================================
        // WhisperModel implements Closeable; always close it to release ONNX sessions
        // and any off-heap native memory held by the encoder/decoder graphs.
        model.close();

        System.out.println("\nWhisper speech-to-text example completed successfully.");
    }
}

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

package org.nd4j.examples.samediff.quickstart.modeling;

import org.eclipse.deeplearning4j.audio.training.TtsTrainingPipeline;
import org.eclipse.deeplearning4j.audio.training.TtsTrainingExample;
import org.eclipse.deeplearning4j.audio.feature.AudioFeatureExtractor;
import org.eclipse.deeplearning4j.audio.transform.AudioPreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;
import org.nd4j.autodiff.samediff.config.TtsTrainingConfig;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.PeftConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.curation.audio.AudioDataProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TTS Training Pipeline — Complete API Reference
 *
 * This example demonstrates the full Text-to-Speech (TTS) fine-tuning pipeline
 * using SameDiff and DL4J's audio processing stack.
 *
 * Sections covered:
 *
 *   1. TtsFineTuneConfig presets — voiceCloning, fullFinetune, withLora
 *   2. Custom TtsFineTuneConfig — builder pattern with all audio parameters
 *   3. TtsTrainingConfig — learning rate, LoRA/PEFT, mixed precision
 *   4. AudioDataProcessor — waveform normalization, mel-spectrogram, truncation
 *   5. TtsTrainingExample — assembling labeled audio examples for training
 *   6. Synthetic SameDiff model — minimal model with 2D weight matrices
 *   7. TtsTrainingPipeline — create, configure, and inspect the pipeline
 *   8. AudioFeatureExtractor — MFCC, mel-spectrogram, log-mel extraction
 *
 * Key classes:
 *   - {@link TtsFineTuneConfig} — Audio pre-processing and encoder freeze settings
 *   - {@link TtsTrainingConfig} — Optimizer, LoRA/PEFT, precision, batch settings
 *   - {@link AudioDataProcessor} — Normalize, mel-spectrogram, truncation utilities
 *   - {@link TtsTrainingExample} — Labeled (waveform, text) pairs for fine-tuning
 *   - {@link TtsTrainingPipeline} — Orchestrates model training end-to-end
 *   - {@link AudioFeatureExtractor} — General-purpose MFCC / mel-spectrogram extraction
 *   - {@link LoraConfig} — Low-Rank Adaptation configuration (extends PeftConfig)
 *   - {@link AudioPreprocessor} — Raw audio transforms (resampling, channel mixing)
 *
 * Important: Calling pipeline.train() requires a real model with proper SameDiff ops.
 * This example demonstrates API setup, configuration, and method signatures only.
 * Calling train() on a synthetic placeholder model will throw an exception at runtime.
 *
 * Run with:
 *   cd samediff-examples
 *   mvn exec:java -Dexec.mainClass="org.nd4j.examples.samediff.quickstart.modeling.TtsTrainingPipelineExample"
 */
public class TtsTrainingPipelineExample {

    public static void main(String[] args) throws Exception {

        // ============================================================
        // 1. TTS FINE-TUNE CONFIG PRESETS
        // ============================================================
        System.out.println("=== 1. TtsFineTuneConfig Presets ===");

        // voiceCloning() — Freeze the text encoder, train only speaker embedding.
        // Ideal for cloning a new voice with a small reference audio dataset.
        TtsFineTuneConfig voiceCloningConfig = TtsFineTuneConfig.voiceCloning();
        System.out.println("  voiceCloning():");
        System.out.println("    freezeTextEncoder:      " + voiceCloningConfig.isFreezeTextEncoder());
        System.out.println("    trainSpeakerEmbedding:  " + voiceCloningConfig.isTrainSpeakerEmbedding());

        // fullFinetune() — All parameters trainable, including the text encoder.
        // Suitable for adapting a TTS model to a new language or accent at full capacity.
        TtsFineTuneConfig fullFinetuneConfig = TtsFineTuneConfig.fullFinetune();
        System.out.println("  fullFinetune():");
        System.out.println("    freezeTextEncoder:      " + fullFinetuneConfig.isFreezeTextEncoder());

        // withLora(rank) — Freeze the text encoder, attach LoRA adapters to the decoder.
        // Efficient fine-tuning: only rank-16 low-rank matrices are trained.
        TtsFineTuneConfig loraConfig16 = TtsFineTuneConfig.withLora(16);
        System.out.println("  withLora(16):");
        System.out.println("    freezeTextEncoder:      " + loraConfig16.isFreezeTextEncoder());
        System.out.println("    decoderLoraConfig set:  " + (loraConfig16.getDecoderLoraConfig() != null));

        // Print shared audio processing fields (same across all presets)
        System.out.println("  Shared audio parameters (defaults):");
        System.out.println("    sampleRate:    " + voiceCloningConfig.getSampleRate() + " Hz");
        System.out.println("    numMelBins:    " + voiceCloningConfig.getNumMelBins());
        System.out.println("    fftSize:       " + voiceCloningConfig.getFftSize());
        System.out.println("    hopLength:     " + voiceCloningConfig.getHopLength());
        System.out.println("    fMin:          " + voiceCloningConfig.getFMin() + " Hz");
        System.out.println("    fMax:          " + voiceCloningConfig.getFMax() + " Hz");
        System.out.println("    normalization: " + voiceCloningConfig.getNormalization());

        // AudioNormalization enum values
        System.out.println("  AudioNormalization enum values:");
        for (TtsFineTuneConfig.AudioNormalization norm : TtsFineTuneConfig.AudioNormalization.values()) {
            System.out.println("    " + norm);
        }
        // PEAK  — Divide by absolute maximum sample value; scales to [-1, 1]
        // RMS   — Divide by RMS level; preserves perceived loudness relationships
        // NONE  — No normalization applied; waveform passed through as-is

        // ============================================================
        // 2. CUSTOM TTS FINE-TUNE CONFIG
        // ============================================================
        System.out.println("\n=== 2. Custom TtsFineTuneConfig ===");

        // Build a fully customized config for a 22kHz model
        TtsFineTuneConfig ttsConfig = TtsFineTuneConfig.builder()
                .sampleRate(22050)                                    // 22.05 kHz audio
                .numMelBins(80)                                       // mel filterbank bins
                .fftSize(1024)                                        // STFT window size
                .hopLength(256)                                       // STFT hop (frame stride)
                .winLength(1024)                                      // window length (= fftSize)
                .fMin(0.0)                                            // lowest mel frequency (Hz)
                .fMax(8000.0)                                         // highest mel frequency (Hz)
                .logOffset(1e-6)                                      // log(mel + logOffset) for stability
                .freezeTextEncoder(true)                              // keep text encoder frozen
                .trainSpeakerEmbedding(true)                          // train speaker embedding
                .speakerEmbeddingDim(256)                             // speaker embedding dimension
                .maxAudioDuration(30.0)                               // max clip length in seconds
                .normalization(TtsFineTuneConfig.AudioNormalization.RMS)  // RMS normalization
                .build();

        System.out.println("  Custom config built:");
        System.out.println("    sampleRate:            " + ttsConfig.getSampleRate() + " Hz");
        System.out.println("    numMelBins:            " + ttsConfig.getNumMelBins());
        System.out.println("    fftSize:               " + ttsConfig.getFftSize());
        System.out.println("    hopLength:             " + ttsConfig.getHopLength());
        System.out.println("    winLength:             " + ttsConfig.getWinLength());
        System.out.println("    fMin:                  " + ttsConfig.getFMin());
        System.out.println("    fMax:                  " + ttsConfig.getFMax());
        System.out.println("    logOffset:             " + ttsConfig.getLogOffset());
        System.out.println("    freezeTextEncoder:     " + ttsConfig.isFreezeTextEncoder());
        System.out.println("    trainSpeakerEmbedding: " + ttsConfig.isTrainSpeakerEmbedding());
        System.out.println("    speakerEmbeddingDim:   " + ttsConfig.getSpeakerEmbeddingDim());
        System.out.println("    maxAudioDuration:      " + ttsConfig.getMaxAudioDuration() + " sec");
        System.out.println("    normalization:         " + ttsConfig.getNormalization());

        // Validation and derived properties
        ttsConfig.validate();  // throws IllegalStateException if config is inconsistent
        int maxSamples = ttsConfig.getMaxSamples();
        System.out.println("  Derived:");
        System.out.println("    getMaxSamples():           " + maxSamples
                + "  (" + ttsConfig.getMaxAudioDuration() + " sec * " + ttsConfig.getSampleRate() + " Hz)");
        // computeNumFrames: number of mel spectrogram frames for a given waveform length
        int numSamples = ttsConfig.getSampleRate() * 3;  // 3-second clip
        int numFrames = ttsConfig.computeNumFrames(numSamples);
        System.out.println("    computeNumFrames(" + numSamples + "): " + numFrames
                + "  (numSamples / hopLength)");

        // ============================================================
        // 3. TTS TRAINING CONFIG
        // ============================================================
        System.out.println("\n=== 3. TtsTrainingConfig ===");

        // Basic training config without PEFT
        TtsTrainingConfig basicTrainingConfig = TtsTrainingConfig.builder()
                .learningRate(1e-4)                    // peak learning rate
                .minLearningRate(1e-6)                 // cosine schedule minimum
                .warmupSteps(500)                      // linear warmup steps
                .numEpochs(10)                         // training epochs
                .batchSize(8)                          // examples per batch
                .maxAudioLengthSec(30.0)               // maximum audio clip length
                .gradientAccumulationSteps(2)          // accumulate gradients over N steps
                .computeDataType(DataType.BFLOAT16)    // mixed-precision compute type
                .weightDecay(0.01)                     // AdamW weight decay
                .build();

        System.out.println("  Basic training config:");
        System.out.println("    learningRate:               " + basicTrainingConfig.getLearningRate());
        System.out.println("    minLearningRate:            " + basicTrainingConfig.getMinLearningRate());
        System.out.println("    warmupSteps:                " + basicTrainingConfig.getWarmupSteps());
        System.out.println("    numEpochs:                  " + basicTrainingConfig.getNumEpochs());
        System.out.println("    batchSize:                  " + basicTrainingConfig.getBatchSize());
        System.out.println("    maxAudioLengthSec:          " + basicTrainingConfig.getMaxAudioLengthSec());
        System.out.println("    gradientAccumulationSteps:  " + basicTrainingConfig.getGradientAccumulationSteps());
        System.out.println("    computeDataType:            " + basicTrainingConfig.getComputeDataType());
        System.out.println("    weightDecay:                " + basicTrainingConfig.getWeightDecay());
        System.out.println("    peftConfig:                 " + basicTrainingConfig.getPeftConfig());

        // Training config with LoRA via PeftConfig interface.
        // LoraConfig implements PeftConfig, so the field type is PeftConfig.
        // r=16 means the low-rank decomposition rank (smaller = fewer params, less expressivity).
        // loraAlpha controls the scaling factor: effective_lr_scale = loraAlpha / r.
        // loraDropout applies dropout to LoRA activations during training.
        LoraConfig loraAdapterConfig = LoraConfig.builder()
                .r(16)                  // rank of the low-rank matrices
                .loraAlpha(32)          // LoRA scaling factor (alpha/r = 2.0 scaling)
                .loraDropout(0.05)      // dropout applied inside LoRA layers
                .build();

        TtsTrainingConfig loraTrainingConfig = TtsTrainingConfig.builder()
                .learningRate(1e-4)
                .minLearningRate(1e-6)
                .warmupSteps(500)
                .numEpochs(10)
                .batchSize(8)
                .maxAudioLengthSec(30.0)
                .gradientAccumulationSteps(2)
                .computeDataType(DataType.BFLOAT16)
                .weightDecay(0.01)
                .peftConfig(loraAdapterConfig)         // attach LoRA adapter (PeftConfig field)
                .build();

        System.out.println("  Training config with LoRA:");
        PeftConfig peft = loraTrainingConfig.getPeftConfig();
        System.out.println("    peftConfig type:   " + (peft != null ? peft.getClass().getSimpleName() : "null"));
        if (peft instanceof LoraConfig) {
            LoraConfig lc = (LoraConfig) peft;
            System.out.println("    r (rank):          " + lc.getR());
            System.out.println("    loraAlpha:         " + lc.getLoraAlpha());
            System.out.println("    loraDropout:       " + lc.getLoraDropout());
        }

        // ============================================================
        // 4. AUDIO DATA PROCESSOR
        // ============================================================
        System.out.println("\n=== 4. AudioDataProcessor ===");

        // AudioDataProcessor takes a TtsFineTuneConfig and applies all configured
        // audio transforms: normalization, mel-spectrogram computation, truncation.
        AudioDataProcessor processor = new AudioDataProcessor(ttsConfig);

        // Create a synthetic 1-second waveform at 24 kHz (shape [1, 24000])
        // Shape: [batch=1, numSamples=24000] (single channel)
        INDArray waveform1sec = Nd4j.rand(DataType.FLOAT, 1, 24000);
        System.out.println("  Synthetic waveform shape:  " + Arrays.toString(waveform1sec.shape()));

        // normalizeAudio: applies the normalization strategy configured in TtsFineTuneConfig
        INDArray normalized = processor.normalizeAudio(waveform1sec);
        System.out.println("  normalizeAudio() output:   " + Arrays.toString(normalized.shape()));

        // computeMelSpectrogram: STFT -> mel filterbank -> log compression
        // Output shape: [batch, numMelBins, numFrames]
        INDArray melSpec = processor.computeMelSpectrogram(waveform1sec);
        System.out.println("  computeMelSpectrogram():   " + Arrays.toString(melSpec.shape())
                + "  [batch, numMelBins=" + ttsConfig.getNumMelBins() + ", numFrames]");

        // process: convenience method that runs normalizeAudio then computeMelSpectrogram
        INDArray processed = processor.process(waveform1sec);
        System.out.println("  process() output:          " + Arrays.toString(processed.shape()));

        // truncate: clips waveform to config.getMaxSamples() if longer
        // Create a 35-second waveform to demonstrate truncation
        INDArray longWaveform = Nd4j.rand(DataType.FLOAT, 1, 22050 * 35);
        System.out.println("  Long waveform shape:       " + Arrays.toString(longWaveform.shape())
                + "  (35 sec)");
        INDArray truncated = processor.truncate(longWaveform);
        System.out.println("  truncate() output:         " + Arrays.toString(truncated.shape())
                + "  (capped at " + ttsConfig.getMaxSamples() + " samples = "
                + ttsConfig.getMaxAudioDuration() + " sec)");

        // computeNumFrames: same formula as TtsFineTuneConfig.computeNumFrames()
        int processorFrames = processor.computeNumFrames(22050 * 3);  // 3-second clip
        System.out.println("  computeNumFrames(66150):   " + processorFrames);

        // ============================================================
        // 5. TRAINING EXAMPLES
        // ============================================================
        System.out.println("\n=== 5. TtsTrainingExample ===");

        // Build a 22 kHz waveform (shape [1, numSamples])
        INDArray waveform22k = Nd4j.rand(DataType.FLOAT, 1, 22050 * 5);  // 5-second clip

        // Minimal example: waveform and text are @NonNull fields
        TtsTrainingExample basicExample = TtsTrainingExample.builder()
                .audioWaveform(waveform22k)            // @NonNull — raw audio samples
                .text("Hello, this is a test.")        // @NonNull — transcript / conditioning text
                .sampleRate(22050)                     // sample rate of the provided waveform
                .build();

        System.out.println("  Basic TtsTrainingExample:");
        System.out.println("    text:          \"" + basicExample.getText() + "\"");
        System.out.println("    sampleRate:    " + basicExample.getSampleRate());
        System.out.println("    audioWaveform: " + Arrays.toString(basicExample.getAudioWaveform().shape()));
        System.out.println("    speakerEmb:    " + basicExample.getSpeakerEmbedding());

        // Example with an explicit speaker embedding vector.
        // The speaker embedding dimension must match speakerEmbeddingDim in TtsFineTuneConfig.
        INDArray speakerEmb = Nd4j.rand(DataType.FLOAT, 256);  // 256-dim speaker vector
        TtsTrainingExample speakerExample = TtsTrainingExample.builder()
                .audioWaveform(waveform22k)
                .text("Welcome to the text-to-speech demo.")
                .sampleRate(22050)
                .speakerEmbedding(speakerEmb)          // optional — pre-computed speaker vector
                .build();

        System.out.println("  TtsTrainingExample with speaker embedding:");
        System.out.println("    text:          \"" + speakerExample.getText() + "\"");
        System.out.println("    speakerEmb:    " + Arrays.toString(speakerExample.getSpeakerEmbedding().shape()));

        // Collect examples into a list for batch training
        List<TtsTrainingExample> trainingExamples = new ArrayList<>();
        trainingExamples.add(basicExample);
        trainingExamples.add(speakerExample);

        // Additional synthetic examples
        String[] sampleTexts = {
                "The quick brown fox jumps over the lazy dog.",
                "Deep learning enables machines to understand speech.",
                "This model was fine-tuned on custom voice data."
        };
        for (String text : sampleTexts) {
            INDArray audio = Nd4j.rand(DataType.FLOAT, 1, 22050 * 4);  // 4-second clips
            trainingExamples.add(TtsTrainingExample.builder()
                    .audioWaveform(audio)
                    .text(text)
                    .sampleRate(22050)
                    .build());
        }
        System.out.println("  Total training examples prepared: " + trainingExamples.size());

        // ============================================================
        // 6. BUILD SYNTHETIC TTS MODEL
        // ============================================================
        System.out.println("\n=== 6. Synthetic SameDiff Model for Pipeline Demonstration ===");

        // This is a minimal SameDiff model that matches the structural expectations
        // of a TTS model: 2D weight matrices that LoRA can target.
        // A real TTS model (e.g. SoundStorm, VITS, YourTTS) would be imported from
        // a checkpoint file using GGMLModelImport or KerasModelImport.
        SameDiff model = SameDiff.create();

        int textEncoderDim = 256;
        int decoderDim = 512;
        int melDim = 80;

        // Text encoder (frozen during voice cloning)
        SDVariable textEmbedWeight = model.var("text_encoder.embedding.weight",
                Nd4j.randn(DataType.FLOAT, 256, textEncoderDim).muli(0.02));
        SDVariable textProjWeight = model.var("text_encoder.proj.weight",
                Nd4j.randn(DataType.FLOAT, textEncoderDim, decoderDim).muli(0.02));

        // Speaker embedding (trainable during voice cloning)
        SDVariable speakerEmbWeight = model.var("speaker_embedding.weight",
                Nd4j.randn(DataType.FLOAT, 256, decoderDim).muli(0.01));  // 256 speakers

        // Decoder attention layers (LoRA targets — 2D weight matrices)
        SDVariable decoderAttnQ = model.var("decoder.layer0.self_attn.q_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, decoderDim).muli(0.02));
        SDVariable decoderAttnK = model.var("decoder.layer0.self_attn.k_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, decoderDim).muli(0.02));
        SDVariable decoderAttnV = model.var("decoder.layer0.self_attn.v_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, decoderDim).muli(0.02));
        SDVariable decoderAttnO = model.var("decoder.layer0.self_attn.o_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, decoderDim).muli(0.02));

        // Decoder MLP layers
        SDVariable decoderMlpGate = model.var("decoder.layer0.mlp.gate_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, decoderDim * 2).muli(0.02));
        SDVariable decoderMlpDown = model.var("decoder.layer0.mlp.down_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim * 2, decoderDim).muli(0.02));

        // Mel output projection
        SDVariable melProjWeight = model.var("mel_proj.weight",
                Nd4j.randn(DataType.FLOAT, decoderDim, melDim).muli(0.02));

        System.out.println("  Synthetic model variables:");
        for (String name : model.variableNames()) {
            long[] shape = model.getVariable(name).getArr().shape();
            System.out.println("    " + name + "  shape=" + Arrays.toString(shape));
        }
        System.out.println("  Total variables: " + model.variableNames().size());

        // ============================================================
        // 7. TTS TRAINING PIPELINE
        // ============================================================
        System.out.println("\n=== 7. TtsTrainingPipeline ===");

        // Create the pipeline with the base model, audio config, and training config.
        // When trainingConfig contains a PeftConfig (e.g., LoraConfig), the pipeline
        // wraps the model with PEFT adapters automatically.
        TtsTrainingPipeline pipelineNoPeft = TtsTrainingPipeline.create(
                model, ttsConfig, basicTrainingConfig);

        System.out.println("  Pipeline (no PEFT):");
        System.out.println("    getModel():          " + (pipelineNoPeft.getModel() != null ? "SameDiff[ok]" : "null"));
        System.out.println("    getTtsConfig():      " + (pipelineNoPeft.getTtsConfig() != null ? "TtsFineTuneConfig[ok]" : "null"));
        System.out.println("    getTrainingConfig(): " + (pipelineNoPeft.getTrainingConfig() != null ? "TtsTrainingConfig[ok]" : "null"));
        // getPeftModel() returns null when no PeftConfig was provided
        System.out.println("    getPeftModel():      " + pipelineNoPeft.getPeftModel()
                + "  (null — no PEFT configured)");

        // Pipeline with LoRA: the pipeline wraps the model with low-rank adapters.
        // getPeftModel() returns the PEFT-wrapped SameDiff model (non-null).
        TtsTrainingPipeline pipelineLora = TtsTrainingPipeline.create(
                model, ttsConfig, loraTrainingConfig);

        System.out.println("  Pipeline (with LoRA):");
        System.out.println("    getPeftModel():      " + (pipelineLora.getPeftModel() != null
                ? "SameDiff[LoRA-wrapped]" : "null")
                + "  (non-null — LoRA adapters attached)");

        // ---- Training method signatures (not called — requires a real model) ----
        // The following block shows how training would be invoked on a real model.
        // Calling these on a synthetic placeholder will throw an exception.
        System.out.println("  Training method signatures (not executed on synthetic model):");

        System.out.println("    // Train on a List<TtsTrainingExample>:");
        System.out.println("    //   pipeline.train(List<TtsTrainingExample> examples)");
        System.out.println("    //   -> Processes waveforms, computes mel-spectrograms,");
        System.out.println("    //      runs forward pass, computes loss, updates weights.");

        System.out.println("    // Train on a MultiDataSetIterator (pre-batched):");
        System.out.println("    //   pipeline.train(MultiDataSetIterator iterator, int totalSteps)");
        System.out.println("    //   -> Iterates for totalSteps gradient updates.");

        System.out.println("    // Merge LoRA weights and export final model:");
        System.out.println("    //   SameDiff merged = pipeline.mergeAndExport()");
        System.out.println("    //   -> Fuses LoRA A*B deltas into base weights,");
        System.out.println("    //      returns a standard SameDiff model (no LoRA overhead).");

        // mergeAndExport() is safe to call even without training — it merges whatever
        // adapter weights are currently set (random if untrained).
        System.out.println("  Calling mergeAndExport() on untrained pipeline (safe):");
        SameDiff mergedModel = pipelineLora.mergeAndExport();
        System.out.println("    mergeAndExport() returned: "
                + (mergedModel != null ? "SameDiff[merged]" : "null"));
        System.out.println("    Merged model variables: " + mergedModel.variableNames().size());

        // ============================================================
        // 8. AUDIO FEATURE EXTRACTOR
        // ============================================================
        System.out.println("\n=== 8. AudioFeatureExtractor ===");

        // AudioFeatureExtractor provides general-purpose audio feature extraction
        // independent of TtsFineTuneConfig. Useful for downstream tasks like
        // speaker verification, speech classification, or analysis.
        AudioFeatureExtractor featureExtractor = AudioFeatureExtractor.builder()
                .sampleRate(22050)    // input audio sample rate
                .fftSize(2048)        // STFT analysis window size
                .hopLength(512)       // STFT hop size (frame stride)
                .numMelBins(128)      // number of mel filterbank channels
                .numMfcc(13)          // number of MFCC coefficients to extract
                .build();

        System.out.println("  AudioFeatureExtractor config:");
        System.out.println("    sampleRate:  " + featureExtractor.getSampleRate());
        System.out.println("    fftSize:     " + featureExtractor.getFftSize());
        System.out.println("    hopLength:   " + featureExtractor.getHopLength());
        System.out.println("    numMelBins:  " + featureExtractor.getNumMelBins());
        System.out.println("    numMfcc:     " + featureExtractor.getNumMfcc());

        // Create a 1-second test audio clip at 22 kHz
        INDArray testAudio = Nd4j.rand(DataType.FLOAT, 1, 22050);

        // extractMfcc: STFT -> mel filterbank -> log -> DCT -> first numMfcc coefficients
        // Output shape: [batch, numMfcc, numFrames]
        INDArray mfcc = featureExtractor.extractMfcc(testAudio);
        System.out.println("  extractMfcc():            " + Arrays.toString(mfcc.shape())
                + "  [batch, numMfcc=13, numFrames]");

        // extractMelSpectrogram: STFT -> mel filterbank (linear scale)
        // Output shape: [batch, numMelBins, numFrames]
        INDArray melSpectrogram = featureExtractor.extractMelSpectrogram(testAudio);
        System.out.println("  extractMelSpectrogram():  " + Arrays.toString(melSpectrogram.shape())
                + "  [batch, numMelBins=128, numFrames]");

        // extractLogMelSpectrogram: STFT -> mel filterbank -> log(mel + offset)
        // Output shape: [batch, numMelBins, numFrames] — log-compressed for neural nets
        INDArray logMel = featureExtractor.extractLogMelSpectrogram(testAudio);
        System.out.println("  extractLogMelSpectrogram():" + Arrays.toString(logMel.shape())
                + "  [batch, numMelBins=128, numFrames]  (log-compressed)");

        System.out.println("\nTTS Training Pipeline example completed successfully.");
        System.out.println("To use this pipeline with a real TTS model:");
        System.out.println("  1. Import a pre-trained TTS model via GGMLModelImport or KerasModelImport");
        System.out.println("  2. Configure TtsFineTuneConfig matching the model's audio parameters");
        System.out.println("  3. Prepare TtsTrainingExample instances from your voice dataset");
        System.out.println("  4. Call pipeline.train(trainingExamples) or pipeline.train(iterator, steps)");
        System.out.println("  5. Call pipeline.mergeAndExport() to obtain the final merged model");
    }
}

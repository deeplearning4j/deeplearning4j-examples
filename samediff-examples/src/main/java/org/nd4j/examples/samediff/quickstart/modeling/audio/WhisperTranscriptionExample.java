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

import org.eclipse.deeplearning4j.audio.feature.WhisperMelSpectrogram;
import org.eclipse.deeplearning4j.audio.io.AudioLoader;
import org.eclipse.deeplearning4j.audio.whisper.WhisperConfig;
import org.eclipse.deeplearning4j.audio.whisper.WhisperDecoderResult;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModel;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader;
import org.eclipse.deeplearning4j.audio.whisper.WhisperModelDownloader.WhisperModelSize;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * REAL speech-to-text: synthesize spoken audio, run OpenAI Whisper (tiny) end to end,
 * and check the transcript against the known utterance.
 *
 * Where {@link WhisperSpeechToTextExample} tours the configuration surface (its
 * transcribe calls are documentation), this example executes the full chain:
 *
 *   speech WAV → {@link AudioLoader} (javax.sound WAV decode)
 *              → resample to 16kHz (native audio_resample op, inside the model)
 *              → {@link WhisperMelSpectrogram} (80-bin log-mel, shown explicitly too)
 *              → encoder ONNX → autoregressive decoder ONNX (KV-cached)
 *              → {@link WhisperDecoderResult} text + segments
 *
 * Audio input, in order of preference:
 *   1. -Dexample.wav.path=/path/to/speech.wav   (any mono/stereo WAV; auto-resampled)
 *   2. espeak-ng synthesis of a known sentence (if installed — Linux: dnf/apt install
 *      espeak-ng). Synthetic-but-real speech makes the transcript CHECKABLE: the
 *      example verifies keywords from the known utterance appear in the output.
 *
 * First run downloads whisper-tiny ONNX from huggingface.co/onnx-community
 * (encoder + decoder + decoder_with_past + tokenizer.json, ~200MB total) into
 * ~/.cache/dl4j-whisper-models. Later runs are offline.
 *
 * Model quality note: TINY is the smallest checkpoint — expect occasional word
 * errors on synthetic speech; the keyword check uses several words and passes on a
 * majority. Swap WhisperModelSize.BASE/SMALL for accuracy at more download/compute.
 *
 * System properties: -Dexample.wav.path=...  -Dexample.model.size=TINY|BASE|SMALL
 */
public class WhisperTranscriptionExample {

    private static final String UTTERANCE =
            "The quick brown fox jumps over the lazy dog";
    private static final String[] KEYWORDS = {"quick", "brown", "fox", "lazy", "dog"};

    public static void main(String[] args) throws Exception {
        WhisperModelSize size = WhisperModelSize.valueOf(
                System.getProperty("example.model.size", "TINY"));

        // ============================================================
        // 1. AUDIO INPUT — user WAV or synthesized speech
        // ============================================================
        System.out.println("=== 1. Audio input ===");
        File wav;
        boolean knownUtterance = false;
        String wavProp = System.getProperty("example.wav.path");
        if (wavProp != null && new File(wavProp).isFile()) {
            wav = new File(wavProp);
            System.out.println("  Using supplied WAV: " + wav.getAbsolutePath());
        } else {
            wav = synthesizeSpeech(UTTERANCE);
            if (wav == null) {
                System.out.println("  No WAV supplied and espeak-ng not available.");
                System.out.println("  Provide audio with -Dexample.wav.path=/path/to/speech.wav");
                System.out.println("  or install espeak-ng for automatic synthesis. Exiting.");
                return;
            }
            knownUtterance = true;
            System.out.println("  Synthesized with espeak-ng: \"" + UTTERANCE + "\"");
            System.out.println("  WAV: " + wav.getAbsolutePath() + " (" + wav.length() / 1024 + " KB)");
        }

        // Decode the WAV to a waveform tensor — shows the raw signal the model sees.
        INDArray samples = AudioLoader.loadWav(wav);
        int sampleRate = AudioLoader.getSampleRate(wav);
        double seconds = samples.length() / (double) sampleRate;
        System.out.println("  Waveform: " + Arrays.toString(samples.shape())
                + " samples @ " + sampleRate + " Hz ("
                + String.format("%.2f", seconds)
                + "s) — Whisper resamples to 16000 Hz internally");

        // ============================================================
        // 2. MODEL DOWNLOAD + LOAD
        // ============================================================
        System.out.println("\n=== 2. Whisper " + size + " (ONNX) ===");
        long t0 = System.currentTimeMillis();
        WhisperModel model = WhisperModel.fromDownload(size,
                WhisperModelDownloader.WhisperModelFormat.ONNX);
        System.out.println("  Downloaded/loaded in " + (System.currentTimeMillis() - t0) + "ms");

        // ============================================================
        // 3. THE MEL FRONT END, SHOWN EXPLICITLY
        // ============================================================
        // transcribe(File) does this internally; running it standalone shows the
        // exact model input: 80 log-mel bins x 3000 frames (30s padded window).
        System.out.println("\n=== 3. Log-mel front end ===");
        WhisperMelSpectrogram mel = new WhisperMelSpectrogram(size.getConfig());
        INDArray melFeatures = mel.extractFeaturesFromFile(wav);
        System.out.println("  Mel features: " + Arrays.toString(melFeatures.shape())
                + " (bins x frames; 100 frames/second, 30s window)");

        // ============================================================
        // 4. TRANSCRIBE
        // ============================================================
        System.out.println("\n=== 4. Transcription ===");
        t0 = System.currentTimeMillis();
        WhisperDecoderResult result = model.transcribe(wav);
        long transcribeMs = System.currentTimeMillis() - t0;

        String text = result.getText() != null ? result.getText().trim() : "";
        System.out.println("  Text: \"" + text + "\"");
        System.out.println("  Wall time: " + transcribeMs + "ms for "
                + String.format("%.2f", seconds) + "s of audio");

        // ============================================================
        // 5. CHECK AGAINST THE KNOWN UTTERANCE
        // ============================================================
        if (knownUtterance) {
            System.out.println("\n=== 5. Keyword check (known utterance) ===");
            String lower = text.toLowerCase(Locale.ROOT);
            int hits = 0;
            StringBuilder detail = new StringBuilder();
            for (String kw : KEYWORDS) {
                boolean hit = lower.contains(kw);
                if (hit) hits++;
                detail.append(kw).append(hit ? "=Y " : "=n ");
            }
            System.out.println("  " + detail);
            System.out.println("  " + hits + "/" + KEYWORDS.length + " keywords found — "
                    + (hits >= 3 ? "PASS (pipeline transcribes real speech)"
                                 : "WEAK (tiny model on synthetic speech; try BASE, or check audio)"));
        }

        model.close();
        System.out.println("\nWhisper transcription example completed.");
    }

    /** Synthesize the utterance with espeak-ng; returns null when unavailable. */
    private static File synthesizeSpeech(String text) {
        try {
            File wav = Files.createTempFile("whisper-utterance", ".wav").toFile();
            wav.deleteOnExit();
            // -s 120: slow, clearly articulated speech (whisper hears espeak's default
            // 175wpm formant synthesis as mush); -v en-us+m3: deeper US male variant
            // transcribes measurably better than the default voice; -g 8: wider word
            // gaps; -a 180: amplitude.
            Process p = new ProcessBuilder("espeak-ng", "-s", "120", "-v", "en-us+m3",
                    "-g", "8", "-a", "180",
                    "-w", wav.getAbsolutePath(), text)
                    .redirectErrorStream(true).start();
            if (!p.waitFor(30, TimeUnit.SECONDS) || p.exitValue() != 0 || wav.length() == 0) {
                return null;
            }
            return wav;
        } catch (Exception e) {
            return null;
        }
    }
}

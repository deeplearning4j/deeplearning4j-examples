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

package org.nd4j.examples.samediff.quickstart.operations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

/**
 * SameDiff Audio Operations (sd.audio()) - Complete API Example
 *
 * The SDAudio namespace provides differentiable audio DSP operations that can be
 * used directly in computation graphs for end-to-end audio processing pipelines.
 *
 * Operations covered:
 *   1.  aWeighting        - A-weighting for human hearing approximation
 *   2.  audioNormalize    - Peak or RMS normalization
 *   3.  audioResample     - Sample rate conversion
 *   4.  chromaFeatures    - Chroma (pitch class) feature extraction from spectrogram
 *   5.  griffinLim        - Phase reconstruction from magnitude spectrogram
 *   6.  melFilterbank     - Mel-scale triangular filterbank matrix (no SDVariable input)
 *   7.  melSpectrogram    - Mel-scaled spectrogram from raw audio
 *   8.  mfcc              - Mel-Frequency Cepstral Coefficients
 *   9.  pitchDetection    - Fundamental frequency estimation
 *  10.  preEmphasis       - High-frequency emphasis filter  y[n] = x[n] - c * x[n-1]
 *  11.  spectralCentroid  - Spectral "center of mass" from spectrogram
 *  12.  spectralRolloff   - Frequency below which X% of energy is concentrated
 *  13.  zeroCrossingRate  - Rate of sign changes in a signal
 */
public class AudioOpsExample {

    public static void main(String[] args) {

        // ============================================================
        // 1. A-WEIGHTING - Perceptual frequency weighting curve
        // ============================================================
        System.out.println("=== 1. A-Weighting ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: 1D array of frequencies in Hz
            SDVariable frequencies = sd.placeHolder("freqs", DataType.FLOAT, -1);

            // Named version: aWeighting(String name, SDVariable frequencies)
            SDVariable weights = sd.audio().aWeighting("weights", frequencies);

            // Unnamed version: aWeighting(SDVariable frequencies)
            SDVariable weightsUnnamed = sd.audio().aWeighting(frequencies);

            // Apply to standard frequency bins from 20 Hz to 20 kHz
            INDArray freqBins = Nd4j.linspace(DataType.FLOAT, 20, 20000, 100);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("freqs", freqBins), "weights");
            System.out.println("  A-weighting shape: " + result.get("weights").shapeInfoToString());
            System.out.println("  A-weighting at low freq (should be strongly negative dB): "
                    + result.get("weights").getFloat(0));
            System.out.println("  A-weighting near 1kHz (should be ~0 dB): "
                    + result.get("weights").getFloat(49));
            System.out.println("  Models human ear sensitivity -- attenuates low and very high frequencies");
        }

        // ============================================================
        // 2. AUDIO NORMALIZE - Peak or RMS amplitude normalization
        // ============================================================
        System.out.println("\n=== 2. Audio Normalize ===");
        {
            SameDiff sd = SameDiff.create();
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, 16000);

            // Full version: audioNormalize(String name, SDVariable input, double targetLevel, boolean useRms)
            // Peak normalization: scales so peak amplitude equals targetLevel
            SDVariable peakNorm = sd.audio().audioNormalize("peakNorm", audio, 1.0, false);

            // RMS normalization: scales so RMS amplitude equals targetLevel
            SDVariable rmsNorm = sd.audio().audioNormalize("rmsNorm", audio, 0.5, true);

            // Named default: audioNormalize(String name, SDVariable input) -- uses targetLevel=1.0, useRms=false
            SDVariable normNamed = sd.audio().audioNormalize("normNamed", audio);

            // Unnamed default: audioNormalize(SDVariable input)
            SDVariable normUnnamed = sd.audio().audioNormalize(audio);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, 16000).mul(0.1);  // quiet signal
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "peakNorm", "rmsNorm");
            System.out.println("  Peak-normalized max amplitude: " + result.get("peakNorm").amaxNumber());
            System.out.println("  RMS-normalized output shape: " + result.get("rmsNorm").shapeInfoToString());
            System.out.println("  useRms=false -> peak norm (max|x|=targetLevel)");
            System.out.println("  useRms=true  -> RMS norm  (sqrt(mean(x^2))=targetLevel)");
        }

        // ============================================================
        // 3. AUDIO RESAMPLE - Sample rate conversion
        // ============================================================
        System.out.println("\n=== 3. Audio Resample ===");
        {
            SameDiff sd = SameDiff.create();

            // Input audio at 44100 Hz (CD quality)
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, 44100);

            // audioResample(String name, SDVariable input, int origSampleRate, int targetSampleRate)
            // Downsample to 16000 Hz (telephony/speech recognition standard)
            SDVariable resampled16k = sd.audio().audioResample("resampled16k", audio, 44100, 16000);

            // Downsample to 22050 Hz (half CD rate, common for music analysis)
            SDVariable resampled22k = sd.audio().audioResample("resampled22k", audio, 44100, 22050);

            // Unnamed version: audioResample(SDVariable input, int origSampleRate, int targetSampleRate)
            SDVariable resampledUnnamed = sd.audio().audioResample(audio, 44100, 8000);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, 44100);  // 1 second at 44.1 kHz
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "resampled16k", "resampled22k");
            System.out.println("  Original samples: 44100 (1 sec @ 44.1 kHz)");
            System.out.println("  Resampled 16k length: " + result.get("resampled16k").length());
            System.out.println("  Resampled 22k length: " + result.get("resampled22k").length());
        }

        // ============================================================
        // 4. CHROMA FEATURES - Pitch class profile from spectrogram
        // ============================================================
        System.out.println("\n=== 4. Chroma Features ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: magnitude spectrogram [batch, freqBins, numFrames]
            // freqBins = fftSize/2+1 = 2048/2+1 = 1025 for fftSize=2048
            int freqBins = 1025;
            SDVariable spectrogram = sd.placeHolder("spectrogram", DataType.FLOAT, -1, freqBins, -1);

            // Full version: chromaFeatures(String name, SDVariable input, int sampleRate, int fftSize, int numChroma)
            // Output: [batch, numChroma, numFrames]
            SDVariable chroma = sd.audio().chromaFeatures("chroma", spectrogram, 22050, 2048, 12);

            // Default version: chromaFeatures(SDVariable input) -- uses 22050, 2048, 12
            SDVariable chromaDefault = sd.audio().chromaFeatures("chromaDefault", spectrogram);

            // Unnamed full version
            SDVariable chromaUnnamed = sd.audio().chromaFeatures(spectrogram, 22050, 2048, 12);

            System.out.println("  Chroma features: 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)");
            System.out.println("  Input:  magnitude spectrogram [batch, freqBins=" + freqBins + ", numFrames]");
            System.out.println("  Output: [batch, numChroma=12, numFrames]");
            System.out.println("  Useful for chord recognition, key detection, music structure analysis");
        }

        // ============================================================
        // 5. GRIFFIN-LIM - Phase reconstruction from magnitude spectrogram
        // ============================================================
        System.out.println("\n=== 5. Griffin-Lim ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: magnitude spectrogram [batch, freqBins, numFrames]
            // freqBins = fftSize/2+1 = 1025 for fftSize=2048
            SDVariable magSpec = sd.placeHolder("magSpec", DataType.FLOAT, -1, 1025, -1);

            // Full version: griffinLim(String name, SDVariable magnitudeSpectrogram,
            //               int fftSize, int hopLength, int numIterations)
            // Output: [batch, samples]
            SDVariable reconstructed = sd.audio().griffinLim("reconstructed", magSpec, 2048, 512, 32);

            // Default version: griffinLim(SDVariable magnitudeSpectrogram) -- uses 2048, 512, 32
            SDVariable reconstructedDefault = sd.audio().griffinLim("reconstructedDefault", magSpec);

            // Unnamed full version
            SDVariable reconstructedUnnamed = sd.audio().griffinLim(magSpec, 2048, 512, 32);

            System.out.println("  Griffin-Lim: iterative phase reconstruction algorithm");
            System.out.println("  Input:  magnitude spectrogram [batch, freqBins=1025, numFrames]");
            System.out.println("  Output: reconstructed waveform [batch, samples]");
            System.out.println("  numIterations=32: more iterations -> better phase estimate");
            System.out.println("  Use case: TTS vocoder, audio style transfer, spectrogram inversion");
        }

        // ============================================================
        // 6. MEL FILTERBANK - Triangular filterbank matrix (no SDVariable input)
        // ============================================================
        System.out.println("\n=== 6. Mel Filterbank ===");
        {
            SameDiff sd = SameDiff.create();

            // NOTE: melFilterbank does NOT take an SDVariable input -- only int/double params
            // Full version: melFilterbank(String name, int numMelBins, int fftSize, int sampleRate,
            //               double lowerEdgeHz, double upperEdgeHz)
            // Output: [numMelBins, fftSize/2+1]
            SDVariable filterbank = sd.audio().melFilterbank("fb", 128, 2048, 22050, 0.0, 8000.0);

            // Default version: melFilterbank(String name, int numMelBins, int fftSize, int sampleRate)
            //   -- uses lowerEdgeHz=0.0, upperEdgeHz=8000.0
            SDVariable fbDefault = sd.audio().melFilterbank("fbDefault", 128, 2048, 22050);

            // Unnamed full version
            SDVariable fbUnnamed = sd.audio().melFilterbank(128, 2048, 22050, 0.0, 8000.0);

            // Unnamed default version
            SDVariable fbUnnamedDefault = sd.audio().melFilterbank(128, 2048, 22050);

            Map<String, INDArray> result = sd.output(Collections.emptyMap(), "fb", "fbDefault");
            System.out.println("  Mel filterbank shape: " + result.get("fb").shapeInfoToString());
            System.out.println("  Default shape: " + result.get("fbDefault").shapeInfoToString());
            System.out.println("  Matrix dimensions: [numMelBins=" + 128 + ", fftSize/2+1=" + (2048/2+1) + "]");
            System.out.println("  Apply to linear spectrogram: melSpec = filterbank @ linearSpec");
            System.out.println("  Mel scale: equally spaced in perceptual (mel) frequency domain");
        }

        // ============================================================
        // 7. MEL SPECTROGRAM - Core time-frequency representation
        // ============================================================
        System.out.println("\n=== 7. Mel Spectrogram ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: raw waveform [batch, samples]
            int sampleRate = 22050;
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, sampleRate);

            // Full version: melSpectrogram(String name, SDVariable input, int sampleRate, int fftSize,
            //               int hopLength, int numMelBins, double lowerEdgeHz, double upperEdgeHz, double power)
            // Output: [batch, numMelBins, numFrames]
            SDVariable melSpec = sd.audio().melSpectrogram("melSpec", audio,
                    sampleRate, 2048, 512, 128, 0.0, 8000.0, 2.0);

            // Default version: melSpectrogram(SDVariable input) -- uses 22050, 2048, 512, 128, 0.0, 8000.0, 2.0
            SDVariable melSpecDefault = sd.audio().melSpectrogram("melSpecDefault", audio);

            // Unnamed full version
            SDVariable melSpecUnnamed = sd.audio().melSpectrogram(audio,
                    sampleRate, 2048, 512, 128, 0.0, 8000.0, 2.0);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, sampleRate);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "melSpec");
            System.out.println("  Mel spectrogram shape: " + result.get("melSpec").shapeInfoToString());
            System.out.println("  power=2.0 -> power spectrogram (magnitude squared)");
            System.out.println("  power=1.0 -> magnitude spectrogram");
            System.out.println("  numFrames = ceil(samples / hopLength)");
        }

        // ============================================================
        // 8. MFCC - Mel-Frequency Cepstral Coefficients
        // ============================================================
        System.out.println("\n=== 8. MFCC ===");
        {
            SameDiff sd = SameDiff.create();
            int sampleRate = 22050;

            // Input: raw waveform [batch, samples]
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, sampleRate);

            // Full version: mfcc(String name, SDVariable input, int sampleRate, int fftSize,
            //               int hopLength, int numMelBins, int numMfcc, double lowerEdgeHz, double upperEdgeHz)
            // Output: [batch, numMfcc, numFrames]
            SDVariable mfcc = sd.audio().mfcc("mfcc", audio,
                    sampleRate, 2048, 512, 128, 13, 0.0, 8000.0);

            // Default version: mfcc(SDVariable input) -- uses 22050, 2048, 512, 128, 13, 0.0, 8000.0
            SDVariable mfccDefault = sd.audio().mfcc("mfccDefault", audio);

            // Unnamed full version
            SDVariable mfccUnnamed = sd.audio().mfcc(audio,
                    sampleRate, 2048, 512, 128, 13, 0.0, 8000.0);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, sampleRate);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "mfcc");
            System.out.println("  MFCC shape: " + result.get("mfcc").shapeInfoToString());
            System.out.println("  numMfcc=13: standard for ASR (coefficients 0-12)");
            System.out.println("  MFCC 0 is log-energy; coefficients 1-12 capture spectral shape");
            System.out.println("  Standard pipeline: preEmphasis -> frame -> window -> FFT -> mel -> log -> DCT");
        }

        // ============================================================
        // 9. PITCH DETECTION - Fundamental frequency estimation
        // ============================================================
        System.out.println("\n=== 9. Pitch Detection ===");
        {
            SameDiff sd = SameDiff.create();
            int sampleRate = 22050;

            // Input: raw waveform [batch, samples]
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, sampleRate);

            // Full version: pitchDetection(String name, SDVariable input, int sampleRate,
            //               int frameLength, int hopLength, double minFreq, double maxFreq)
            // Output: [batch, numFrames]
            SDVariable pitch = sd.audio().pitchDetection("pitch", audio,
                    sampleRate, 2048, 512, 80.0, 1000.0);

            // Default version: pitchDetection(SDVariable input) -- uses 22050, 2048, 512, 80.0, 1000.0
            SDVariable pitchDefault = sd.audio().pitchDetection("pitchDefault", audio);

            // Unnamed full version
            SDVariable pitchUnnamed = sd.audio().pitchDetection(audio,
                    sampleRate, 2048, 512, 80.0, 1000.0);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, sampleRate);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "pitch");
            System.out.println("  Pitch shape: " + result.get("pitch").shapeInfoToString());
            System.out.println("  minFreq=80 Hz (approx lowest male voice fundamental)");
            System.out.println("  maxFreq=1000 Hz (approx highest singing pitch)");
            System.out.println("  Output: fundamental frequency in Hz per frame");
        }

        // ============================================================
        // 10. PRE-EMPHASIS - High-frequency boost filter
        // ============================================================
        System.out.println("\n=== 10. Pre-Emphasis ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: raw waveform [batch, samples]
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, 16000);

            // Full version: preEmphasis(String name, SDVariable input, double coefficient)
            // Formula: y[n] = x[n] - coefficient * x[n-1]
            SDVariable emphasized = sd.audio().preEmphasis("emphasized", audio, 0.97);

            // Default version: preEmphasis(SDVariable input) -- uses coefficient=0.97
            SDVariable emphasizedDefault = sd.audio().preEmphasis("emphasizedDefault", audio);

            // Unnamed full version
            SDVariable emphasizedUnnamed = sd.audio().preEmphasis(audio, 0.97);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, 16000);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "emphasized");
            System.out.println("  Pre-emphasis output shape: " + result.get("emphasized").shapeInfoToString());
            System.out.println("  Formula: y[n] = x[n] - 0.97 * x[n-1]");
            System.out.println("  Boosts high frequencies to compensate for natural spectral roll-off");
            System.out.println("  Standard preprocessing step before STFT, MFCC, or filterbank extraction");
        }

        // ============================================================
        // 11. SPECTRAL CENTROID - "Brightness" of a signal (from spectrogram)
        // ============================================================
        System.out.println("\n=== 11. Spectral Centroid ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: magnitude spectrogram [batch, freqBins, numFrames]
            int freqBins = 1025;  // fftSize/2+1 for fftSize=2048
            SDVariable spectrogram = sd.placeHolder("spectrogram", DataType.FLOAT, -1, freqBins, -1);

            // Full version: spectralCentroid(String name, SDVariable input, int sampleRate, int fftSize)
            // Output: [batch, numFrames]
            SDVariable centroid = sd.audio().spectralCentroid("centroid", spectrogram, 22050, 2048);

            // Default version: spectralCentroid(SDVariable input) -- uses 22050, 2048
            SDVariable centroidDefault = sd.audio().spectralCentroid("centroidDefault", spectrogram);

            // Unnamed full version
            SDVariable centroidUnnamed = sd.audio().spectralCentroid(spectrogram, 22050, 2048);

            System.out.println("  Input: magnitude spectrogram [batch, freqBins=" + freqBins + ", numFrames]");
            System.out.println("  Output: [batch, numFrames] -- centroid freq in Hz per frame");
            System.out.println("  Formula: centroid = sum(f_k * |X_k|) / sum(|X_k|)");
            System.out.println("  Higher values = brighter/higher-frequency content");
            System.out.println("  Use case: timbre analysis, music genre classification");
        }

        // ============================================================
        // 12. SPECTRAL ROLLOFF - Energy concentration boundary
        // ============================================================
        System.out.println("\n=== 12. Spectral Rolloff ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: magnitude spectrogram [batch, freqBins, numFrames]
            int freqBins = 1025;  // fftSize/2+1 for fftSize=2048
            SDVariable spectrogram = sd.placeHolder("spectrogram", DataType.FLOAT, -1, freqBins, -1);

            // Full version: spectralRolloff(String name, SDVariable input, int sampleRate,
            //               int fftSize, double rolloffPercent)
            // Output: [batch, numFrames]
            SDVariable rolloff = sd.audio().spectralRolloff("rolloff", spectrogram, 22050, 2048, 0.85);

            // Default version: spectralRolloff(SDVariable input) -- uses 22050, 2048, 0.85
            SDVariable rolloffDefault = sd.audio().spectralRolloff("rolloffDefault", spectrogram);

            // Unnamed full version
            SDVariable rolloffUnnamed = sd.audio().spectralRolloff(spectrogram, 22050, 2048, 0.85);

            System.out.println("  Input: magnitude spectrogram [batch, freqBins=" + freqBins + ", numFrames]");
            System.out.println("  Output: [batch, numFrames] -- rolloff frequency in Hz per frame");
            System.out.println("  rolloffPercent=0.85: freq below which 85% of spectral energy lies");
            System.out.println("  Low rolloff -> bass-heavy content; high rolloff -> treble-heavy content");
            System.out.println("  Use case: speech/music discrimination, genre classification");
        }

        // ============================================================
        // 13. ZERO CROSSING RATE - Signal oscillation frequency
        // ============================================================
        System.out.println("\n=== 13. Zero Crossing Rate ===");
        {
            SameDiff sd = SameDiff.create();

            // Input: raw waveform [batch, samples]
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, 16000);

            // Full version: zeroCrossingRate(String name, SDVariable input, int frameLength, int hopLength)
            // Output: [batch, numFrames]
            SDVariable zcr = sd.audio().zeroCrossingRate("zcr", audio, 2048, 512);

            // Default version: zeroCrossingRate(SDVariable input) -- uses 2048, 512
            SDVariable zcrDefault = sd.audio().zeroCrossingRate("zcrDefault", audio);

            // Unnamed full version
            SDVariable zcrUnnamed = sd.audio().zeroCrossingRate(audio, 2048, 512);

            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, 16000);
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("audio", audioData), "zcr");
            System.out.println("  Zero crossing rate shape: " + result.get("zcr").shapeInfoToString());
            System.out.println("  High ZCR -> noisy/unvoiced speech or high-frequency content");
            System.out.println("  Low ZCR  -> voiced speech or tonal (pitched) content");
            System.out.println("  Use case: voiced/unvoiced detection, silence detection, onset detection");
        }

        // ============================================================
        // PIPELINE EXAMPLE: Pre-emphasis -> Mel Spectrogram -> MFCC
        //   + Spectrogram-based features: centroid, rolloff, chroma
        // ============================================================
        System.out.println("\n=== Full Audio Feature Extraction Pipeline ===");
        {
            SameDiff sd = SameDiff.create();
            int sampleRate = 16000;
            int fftSize = 1024;
            int hopLength = 256;

            // -- Raw audio input --
            SDVariable rawAudio = sd.placeHolder("rawAudio", DataType.FLOAT, -1, sampleRate);

            // Step 1: Pre-emphasis filter (boost high frequencies)
            SDVariable emphasized = sd.audio().preEmphasis("step1_emphasis", rawAudio, 0.97);

            // Step 2: Peak normalize to unit amplitude
            SDVariable normalized = sd.audio().audioNormalize("step2_normalize", emphasized, 1.0, false);

            // Step 3: Compute mel spectrogram (raw audio -> mel-scaled time-frequency representation)
            SDVariable melSpec = sd.audio().melSpectrogram("step3_melspec", normalized,
                    sampleRate, fftSize, hopLength, 80, 0.0, 8000.0, 2.0);

            // Step 4: Extract MFCC from the normalized waveform
            SDVariable mfcc = sd.audio().mfcc("step4_mfcc", normalized,
                    sampleRate, fftSize, hopLength, 80, 13, 0.0, 8000.0);

            // Step 5: Zero crossing rate for voiced/unvoiced segmentation
            SDVariable zcr = sd.audio().zeroCrossingRate("step5_zcr", normalized, fftSize, hopLength);

            // Step 6: Pitch detection for prosody features
            SDVariable pitch = sd.audio().pitchDetection("step6_pitch", normalized,
                    sampleRate, fftSize, hopLength, 80.0, 800.0);

            // -- Spectrogram-based features (require a linear spectrogram) --
            // freqBins = fftSize/2+1 = 1024/2+1 = 513
            SDVariable linearSpec = sd.placeHolder("linearSpec", DataType.FLOAT, -1, 513, -1);

            // Step 7: Spectral centroid ("brightness")
            SDVariable centroid = sd.audio().spectralCentroid("step7_centroid", linearSpec,
                    sampleRate, fftSize);

            // Step 8: Spectral rolloff (energy distribution boundary)
            SDVariable rolloff = sd.audio().spectralRolloff("step8_rolloff", linearSpec,
                    sampleRate, fftSize, 0.85);

            // Step 9: Chroma features (pitch class profiles for harmonic analysis)
            SDVariable chroma = sd.audio().chromaFeatures("step9_chroma", linearSpec,
                    sampleRate, fftSize, 12);

            // -- Run the waveform-based pipeline --
            INDArray audioData = Nd4j.randn(DataType.FLOAT, 1, sampleRate);
            Map<String, INDArray> results = sd.output(
                    Collections.singletonMap("rawAudio", audioData),
                    "step3_melspec", "step4_mfcc", "step5_zcr", "step6_pitch");

            System.out.println("  Pipeline output shapes:");
            System.out.println("    Mel spectrogram:    " + results.get("step3_melspec").shapeInfoToString());
            System.out.println("    MFCC:               " + results.get("step4_mfcc").shapeInfoToString());
            System.out.println("    Zero crossing rate: " + results.get("step5_zcr").shapeInfoToString());
            System.out.println("    Pitch:              " + results.get("step6_pitch").shapeInfoToString());
            System.out.println("  Spectrogram-based ops (centroid, rolloff, chroma) are registered");
            System.out.println("  in the graph and operate on [batch, freqBins=513, numFrames] input");
        }

        // ============================================================
        // GRAPH SUMMARY - Show all ops registered in a combined graph
        // ============================================================
        System.out.println("\n=== Graph Summary: All 13 SDAudio Ops ===");
        {
            SameDiff sd = SameDiff.create();
            int sampleRate = 22050;
            int fftSize = 2048;
            int hopLength = 512;
            int freqBins = fftSize / 2 + 1;  // 1025

            // Audio waveform input: [batch, samples]
            SDVariable audio = sd.placeHolder("audio", DataType.FLOAT, -1, sampleRate);

            // Spectrogram input for spectrogram-based ops: [batch, freqBins, numFrames]
            SDVariable spectrogram = sd.placeHolder("spec", DataType.FLOAT, -1, freqBins, -1);

            // Frequency input for A-weighting
            SDVariable frequencies = sd.placeHolder("freqs", DataType.FLOAT, -1);

            // Register all 13 ops
            SDVariable op01 = sd.audio().aWeighting("op01_aWeighting", frequencies);
            SDVariable op02 = sd.audio().audioNormalize("op02_audioNormalize", audio, 1.0, false);
            SDVariable op03 = sd.audio().audioResample("op03_audioResample", audio, sampleRate, 16000);
            SDVariable op04 = sd.audio().chromaFeatures("op04_chromaFeatures", spectrogram, sampleRate, fftSize, 12);
            SDVariable op05 = sd.audio().griffinLim("op05_griffinLim", spectrogram, fftSize, hopLength, 32);
            SDVariable op06 = sd.audio().melFilterbank("op06_melFilterbank", 128, fftSize, sampleRate, 0.0, 8000.0);
            SDVariable op07 = sd.audio().melSpectrogram("op07_melSpectrogram", audio, sampleRate, fftSize, hopLength, 128, 0.0, 8000.0, 2.0);
            SDVariable op08 = sd.audio().mfcc("op08_mfcc", audio, sampleRate, fftSize, hopLength, 128, 13, 0.0, 8000.0);
            SDVariable op09 = sd.audio().pitchDetection("op09_pitchDetection", audio, sampleRate, fftSize, hopLength, 80.0, 1000.0);
            SDVariable op10 = sd.audio().preEmphasis("op10_preEmphasis", audio, 0.97);
            SDVariable op11 = sd.audio().spectralCentroid("op11_spectralCentroid", spectrogram, sampleRate, fftSize);
            SDVariable op12 = sd.audio().spectralRolloff("op12_spectralRolloff", spectrogram, sampleRate, fftSize, 0.85);
            SDVariable op13 = sd.audio().zeroCrossingRate("op13_zeroCrossingRate", audio, fftSize, hopLength);

            System.out.println("  SDAudio ops registered in graph:");
            String[] opNames = {
                "op01_aWeighting", "op02_audioNormalize", "op03_audioResample",
                "op04_chromaFeatures", "op05_griffinLim", "op06_melFilterbank",
                "op07_melSpectrogram", "op08_mfcc", "op09_pitchDetection",
                "op10_preEmphasis", "op11_spectralCentroid", "op12_spectralRolloff",
                "op13_zeroCrossingRate"
            };
            System.out.println("  " + Arrays.toString(opNames));
            System.out.println("\n  Graph variable count: " + sd.variables().size());
            System.out.println("  Graph function count: " + sd.getOps().size());
            System.out.println("\n  Inputs summary:");
            System.out.println("    Waveform ops (audio [batch, samples]):          normalize, resample,");
            System.out.println("                                                     melSpectrogram, mfcc,");
            System.out.println("                                                     pitchDetection, preEmphasis,");
            System.out.println("                                                     zeroCrossingRate");
            System.out.println("    Spectrogram ops (spec [batch, freqBins, time]):  chromaFeatures, griffinLim,");
            System.out.println("                                                     spectralCentroid, spectralRolloff");
            System.out.println("    Frequency ops (freqs [N]):                       aWeighting");
            System.out.println("    Parameter-only ops (no SDVariable input):        melFilterbank");
        }

        System.out.println("\nAll 13 SDAudio operations demonstrated successfully.");
    }
}

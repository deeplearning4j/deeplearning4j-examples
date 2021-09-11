/*******************************************************************************
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

package org.deeplearning4j.examples.wip.advanced.modelling.melodl4j;

import java.util.List;

/**
 * Converts a sequence of Notes to textual form for use by LSTM learning.
 *
 * @author Don Smith
 */
public class MelodyStrings {
    public static final String COMMENT_STRING = "//";
    // The following strings are used to build the symbolic representation of a melody
    // The next two strings contain chars used to indicate pitch deltas.
    public static final String noteGapCharsPositive = "0123456789abc"; // A pitch delta of "0" indicates delta=0.
    public static final String noteGapCharsNegative = "ABCDEFGHIJKLM"; // A pitch delta of "A" indicates delta=-1.
    // R is used to indicate the beginning of a rest
    public static int durationDeltaParts = 8;
    public static final String durationChars = "defghijkmnopqrstuvwzyz!@#$%^&*-_"; // 32 divisions.  We omit lower-case L to avoid confusion with one.
    //                                          12345678901234567890123456789012
    public static final String allValidCharacters = getValidCharacters();
    // 13+13+1+32 = 59 possible characters.
    // 'd' indicates the smallest pitch duration allowed (typically a 1/32 note or so).
    // 'e' is a duration twice that of 'd'
    // 'f' is a duration three times that of 'd', etc.
    // If there is a rest between notes, we append 'R' followed by a char for the duration of the rest.
    public static final char restChar = 'R';

    /**
     * @return characters that may occur in a valid melody string
     */
    private static String getValidCharacters() {
        StringBuilder sb = new StringBuilder();
        sb.append(noteGapCharsPositive);
        sb.append(noteGapCharsNegative);
        sb.append(durationChars);
        sb.append('R');
        return sb.toString();
    }

    public static String convertToMelodyString(List<Note> noteSequence) {
        double averageNoteDuration = computeAverageDuration(noteSequence);
        double durationDelta = averageNoteDuration / durationDeltaParts;
        Note previousNote = null;
        StringBuilder sb = new StringBuilder();
        for (Note note : noteSequence) {
            if (previousNote == null) {
                // The first pitch is excluded. Only its duration is included
                char noteDurationChar = computeDurationChar(note.getDurationInTicks(), durationDelta);
                sb.append(noteDurationChar);
            } else {
                long restDuration = note.getStartTick() - previousNote.getEndTick();
                if (restDuration > 0) {
                    char restDurationChar = computeDurationChar(restDuration, durationDelta);
                    sb.append(restChar);
                    sb.append(restDurationChar);
                }
                int pitchGap = note.getPitch() - previousNote.getPitch();
                while (pitchGap >= noteGapCharsPositive.length()) {
                    pitchGap -= noteGapCharsPositive.length();
                }
                while (pitchGap < -noteGapCharsNegative.length()) {
                    pitchGap += noteGapCharsNegative.length();
                }
                sb.append(getCharForPitchGap(pitchGap));
                long noteDuration = note.getDurationInTicks();
                char noteDurationChar = computeDurationChar(noteDuration, durationDelta);
                sb.append(noteDurationChar);
            }
            previousNote = note;
        }
        return sb.toString();
    }

    private static char getCharForPitchGap(int pitchGap) {
        return pitchGap >= 0 ? noteGapCharsPositive.charAt(pitchGap) : noteGapCharsNegative.charAt(-1 - pitchGap);
    }

    private static char computeDurationChar(long duration, double durationDelta) {
        int times = Math.min((int) Math.round(duration / durationDelta), durationChars.length() - 1);
        if (times < 0) {
            System.err.println("WARNING: Duration = " + duration);
            times = 0;
        }
        return durationChars.charAt(times);
    }

    private static double computeAverageDuration(List<Note> noteSequence) {
        double sum = 0;
        for (Note note : noteSequence) {
            sum += note.getDurationInTicks();
        }
        if (sum == 0) {
            for (Note note : noteSequence) {
                System.out.println(note);
            }
            throw new IllegalStateException("0 sum");
        }
        return sum / noteSequence.size();
    }
}

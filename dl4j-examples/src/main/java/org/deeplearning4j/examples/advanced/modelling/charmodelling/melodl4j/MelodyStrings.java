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

package org.deeplearning4j.examples.advanced.modelling.charmodelling.melodl4j;

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
    // We use this ordering of characters so that the pitch gap order is the same as the ASCII order.
    public static final char lowestPitchGapChar = 'A';
    public static final char REST_CHAR = ' ';

    // As written now, it allows pitch gaps between -12 and +12, inclusive.
    // If you want to change the allowed gap, you will have to change the characters in PITCH_GAP_CHARS_POSITIVE
    // and PITCH_GAP_CHARS_NEGATIVE

    // There are thirteen chars in pitchGapCharsPositive because the first one ('M') indicates a zero pitch gap.
    // "M" indicates delta=0. "N" indicates delta=1, 'O' indicates delta=2, etc.
    public static final String PITCH_GAP_CHARS_POSITIVE = "MNOPQRSTUVWXY";

    public static int MAX_POSITIVE_PITCH_GAP = PITCH_GAP_CHARS_POSITIVE.length()-1; // -1 because the first char is for zero.
    public static final char ZERO_PITCH_DELTA_CHAR = PITCH_GAP_CHARS_POSITIVE.charAt(0);
    public static final char MAX_POSITIVE_PITCH_DELTA_CHAR = PITCH_GAP_CHARS_POSITIVE.charAt(PITCH_GAP_CHARS_POSITIVE.length()-1);

    // The order of characters below is intended to simplify the learning of pitch gaps, because increasing
    // characters correspond to increasing pitch gaps.  (Whether this convention simplifies learning pitches
    // depends on which learning algorithm is used.) The string must end with 'A'.
    // "L" indicates delta=-1. "K" indicates delta = -2, 'J' indicates delta = -3, etc.
    public static final String PITCH_GAP_CHARS_NEGATIVE = "LKJIHGFEDCBA";

    public static final char FIRST_PITCH_CHAR_NEGATIVE =  PITCH_GAP_CHARS_NEGATIVE.charAt(0);

    // durationDeltaParts determines how short durations can get. The shortest duration is 1/durationDeltaParts
    // as long as the average note length in the piece.
    public static int durationDeltaParts = 8;
    public static final String DURATION_CHARS = "]^_`abcdefghijklmnopqrstuvwxyz{|"; // 32 divisions, in ASCII order

    public static final char FIRST_DURATION_CHAR = DURATION_CHARS.charAt(0);
    public static final String allValidCharacters = getValidCharacters();
    // 13+13+1+32 = 59 possible characters.
    // ']' indicates the smallest pitch duration allowed (typically a 1/32 note or so).
    // '^' is a duration twice that of ']'
    // '_' is a duration three times that of ']', etc.
    // If there is a rest between notes, we append ' ' followed by a char for the duration of the rest.

    /**
     * @return characters that may occur in a valid melody string
     */
    private static String getValidCharacters() {
        StringBuilder sb = new StringBuilder();
        sb.append(PITCH_GAP_CHARS_POSITIVE);
        sb.append(PITCH_GAP_CHARS_NEGATIVE);
        sb.append(DURATION_CHARS);
        sb.append(REST_CHAR);
        return sb.toString();
    }
    public static boolean isValidMelodyString(String string) {
        for(int i=0;i<string.length();i++) {
            if (i%2==0) {
                if (!isDurationChar(string.charAt(i))) {
                    return false;
                }
            } else {
                if (!isPitchCharOrRest(string.charAt(i))) {
                    return false;
                }
            }
        }
        return true;
    }

    public static int getPitchDelta(final char ch) {
        if (ch >= ZERO_PITCH_DELTA_CHAR && ch <= MAX_POSITIVE_PITCH_DELTA_CHAR) {
            return ch - ZERO_PITCH_DELTA_CHAR;
        }
        if (ch >= 'A' && ch <= FIRST_PITCH_CHAR_NEGATIVE) {
            return - (1 + (FIRST_PITCH_CHAR_NEGATIVE - ch));
        }
        return 0;
    }
    public static char getCharForPitchGap(int pitchGap) {
        while (pitchGap>MAX_POSITIVE_PITCH_GAP) {
            pitchGap -= MAX_POSITIVE_PITCH_GAP;
        }
        while (pitchGap < -MAX_POSITIVE_PITCH_GAP) {
            pitchGap += MAX_POSITIVE_PITCH_GAP;
        }
        return (char) (ZERO_PITCH_DELTA_CHAR + pitchGap);
    }

    public static int getDurationInTicks(char ch, int resolutionDelta) {
        int diff = Math.max(0,ch - FIRST_DURATION_CHAR);
        return diff * resolutionDelta;
    }

    public static boolean isDurationChar(char ch) {
        return ch>=']' && ch <= '|';
    }
    public static boolean isPitchCharOrRest(char ch) {
        return ch == REST_CHAR || ch >= 'A' && ch <= 'Z';
    }
    public static boolean isPitchChar(char ch) {
        return ch >= 'A' && ch <= 'Z';
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
                    sb.append(REST_CHAR);
                    sb.append(restDurationChar);
                }
                int pitchGap = note.getPitch() - previousNote.getPitch();
                while (pitchGap > MAX_POSITIVE_PITCH_GAP) {
                    pitchGap -= MAX_POSITIVE_PITCH_GAP;
                }
                while (pitchGap < -MAX_POSITIVE_PITCH_GAP) {
                    pitchGap += MAX_POSITIVE_PITCH_GAP;
                }
                sb.append(getCharForPitchGap(pitchGap));
                long noteDuration = note.getDurationInTicks();
                char noteDurationChar = computeDurationChar(noteDuration, durationDelta);
                sb.append(noteDurationChar);
            }
            previousNote = note;
        }
        String result= sb.toString();
        if (!isValidMelodyString(result)) {
            System.err.println("Invalid melody string: " + result);
        }
        return result;
    }


    private static char computeDurationChar(long duration, double durationDelta) {
        int times = Math.min((int) Math.round(duration / durationDelta), DURATION_CHARS.length() - 1);
        if (times < 0) {
            System.err.println("WARNING: Duration = " + duration);
            times = 0;
        }
        char ch = DURATION_CHARS.charAt(times);
        if (!isDurationChar(ch)) {
            throw new IllegalStateException("Invalid duration char " + ch + " for duration " + duration + ", " + durationDelta);
        }
        return ch;
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

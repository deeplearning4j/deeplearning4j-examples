package org.deeplearning4j.examples.recurrent.character.melodl4j;

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
    public static final String pitchGapCharsNegative = "MLKJIHGFEDCBA"; // "M" indicates delta=-1. "L" indicates -2,...
    public static final String pitchGapCharsPositive = "NOPQRSTUVWXYZ"; // "N" indicates delta=0. "O" indicates 1, ...
    public static final int numberOfPitchGapCharsIncludingRest =  1+ pitchGapCharsNegative.length() + pitchGapCharsPositive.length();
    public static String allPitchGapCharsIncludingRest = REST_CHAR + pitchGapCharsNegative + pitchGapCharsPositive;
    // ' ' is used to indicate the beginning of a rest
    public static int durationDeltaParts = 8; //12345678901234567890123456789012
    public static final String durationChars = "]^_`abcdefghijklmnopqrstuvwzyz{|"; // 32 divisions, in ASCII order
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
        sb.append(pitchGapCharsPositive);
        sb.append(pitchGapCharsNegative);
        sb.append(durationChars);
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
    public static boolean isDurationChar(char ch) {
        return ch>=']' && ch <= '|';
    }
    public static boolean isPitchCharOrRest(char ch) {
        return ch == REST_CHAR || ch >= 'A' && ch <= 'Z';
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
                while (pitchGap >= pitchGapCharsPositive.length()) {
                    pitchGap -= pitchGapCharsPositive.length();
                }
                while (pitchGap < -pitchGapCharsNegative.length()) {
                    pitchGap += pitchGapCharsNegative.length();
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

    private static char getCharForPitchGap(int pitchGap) {
        char ch= pitchGap >= 0 ? pitchGapCharsPositive.charAt(pitchGap) : pitchGapCharsNegative.charAt(-1 - pitchGap);
        if (!isPitchCharOrRest(ch)) {
            throw new IllegalStateException("Got char " + ch + " for pitchGap " + pitchGap);
        }
        return ch;
    }

    private static char computeDurationChar(long duration, double durationDelta) {
        int times = Math.min((int) Math.round(duration / durationDelta), durationChars.length() - 1);
        if (times < 0) {
            System.err.println("WARNING: Duration = " + duration);
            times = 0;
        }
        char ch = durationChars.charAt(times);
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

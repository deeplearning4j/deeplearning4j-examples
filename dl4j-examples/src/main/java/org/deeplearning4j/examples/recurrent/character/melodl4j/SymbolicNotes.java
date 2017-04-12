package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.util.HashMap;
import java.util.Map;

public class SymbolicNotes {
    private static final String durationChars = "tsiqhw";  // x=64, t=32nd, s = 16th, i=eigth, q=quarter, h=half, w=whole
    private static Map<String,Integer> noteMap = new HashMap<>();
    private static final String[] pitches= {"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    static {
        noteMap.put("C", 0);
        noteMap.put("C#", 1);
        noteMap.put("Db", 1);
        noteMap.put("D", 2);
        noteMap.put("D#", 3);
        noteMap.put("Eb", 3);
        noteMap.put("E", 4);
        noteMap.put("Fb", 4);
        noteMap.put("E#", 5);
        noteMap.put("F", 5);
        noteMap.put("F#", 6);
        noteMap.put("Gb", 6);
        noteMap.put("G", 7);
        noteMap.put("G#", 8);
        noteMap.put("Ab", 8);
        noteMap.put("A", 9);
        noteMap.put("A#", 10);
        noteMap.put("Bb", 10);
        noteMap.put("B", 11);
    }
    //--------------------------------------------------------
    // Assumes quarter note duration
    public static String pitchToJFugueNoteString(int rawPitch) {
        int key = rawPitch%12;
        int scale=rawPitch/12;
        String pitchString = pitches[key];
        return pitchString + scale;
    } //  Middle C = C5 = 60
    public static String pitchToJFugueNoteStringWithDefaultDuration(int rawPitch) {
        return pitchToJFugueNoteString(rawPitch) + "q";
    }
    //----------------------
    public static char getMelodyDurationCharacterFromJFuguePatternString(String noteSymbolAndScale) {// C5 or Ab4 or or C5q or D3w  or C5q.
        int duration=getDurationInMultiplesOf32ndNoteFromJFuguePatternString(noteSymbolAndScale);
        return Midi2MelodyStrings.durationChars.charAt(duration);
    }
    //----------------------------
    public static String melodyToJFugue(final String melodyString, int startPitch) {
        StringBuilder sb = new StringBuilder();
        char ch = melodyString.charAt(0);
        sb.append(pitchToJFugueNoteString(startPitch));
        int index=0;
        if (Midi2MelodyStrings.isDurationChar(ch)) {
            sb.append(durationCharToJFugueTempo(ch));
            index++;
        } else {
            sb.append("q"); // default
        }
        sb.append(' ');
        int lastPitch=startPitch;
        while (index<melodyString.length()) {
            ch=melodyString.charAt(index);
            index++;
            if (Midi2MelodyStrings.isPitchDeltaChar(ch)) {
                int delta = Midi2MelodyStrings.getPitchDeltaFromMelodyChar(ch);
                lastPitch+= delta;
                sb.append(pitchToJFugueNoteString(lastPitch));
            } else if (ch=='R') {
                sb.append(ch);
            } else {
                System.err.println("WARNING: bad char (" + ch + ") in melody");
                continue; // Skip duration calculation below
            }
            ch=melodyString.charAt(index);
            if (Midi2MelodyStrings.isDurationChar(ch)) {
                sb.append(durationCharToJFugueTempo(ch));
                index++;
            } else {
                sb.append('q');
            }
            sb.append(' ');
        }
        return sb.toString();
    }
    public static String durationCharToJFugueTempo(char ch) {
        int index= Midi2MelodyStrings.durationChars.indexOf(ch);
        if (index<0) { // invalid
            return "q"; // default
        }
        switch (index) {
            case 1:
                return "t";
            case 2:
                return "s";
            case 3:
                return "s.";
            case 4:
                return "i";
            case 5:
                return "it";
            case 6:
               return "is";
            case 7:
                return "ist";
            case 8:
                return "q";
            case 9:
                return "qt";
            case 10:
                return "qs";
            case 11:
                return "qst";
            case 12:
                return "qi";
            case 13:
                return "qit";
            case 14:
                return "qis";
            case 15:
                return "qist";
            case 16:
                return "h";
            case 32:
                return "w";
            default:
                return "q";
        }
    }
    //---------------------
    // Default to quarter note (4), up to a maximum of 32 (a whole note)
    public static int getDurationInMultiplesOf32ndNoteFromJFuguePatternString(String noteSymbolAndScale) {// Rq or C5 or Ab4 or or C5q or D3w  or C5q.
        final String originalNoteSymbolAndScale=noteSymbolAndScale;
        boolean dotted = noteSymbolAndScale.endsWith(".");
        if (dotted) {
            noteSymbolAndScale = noteSymbolAndScale.substring(0,noteSymbolAndScale.length()-1);
        }
        char lastChar = noteSymbolAndScale.charAt(noteSymbolAndScale.length()-1);
        int index= durationChars.indexOf(lastChar);
        if (index<0) {
            //throw new IllegalArgumentException("Illegal duration char " + lastChar + " in " + originalNoteSymbolAndScale);
            return 4; // default is quarter note
        }
        int duration=1;
        while (index>0) {
            duration+= duration;
            index--;
        }
        if (dotted) {
            duration+= duration/2;
        }
        if (duration>32) {
            duration=32; // max is whole note
        }
        return duration;
    }
    //---------------------
    public static int getPitchFromJFuguePatternString(String jfugue) {// C5 or Ab4 or or C5q or D3w  or C5q.
        final String originalNoteSymbolAndScale=jfugue;
        int len=jfugue.length();
        if (len==0) {
            return 0;
        }
        char lastChar = jfugue.charAt(len-1);
        while (lastChar == '.' || durationChars.indexOf(lastChar)>=0) {
            jfugue= jfugue.substring(0,len-1);
            len--;
            if (len==0) {
                throw new IllegalArgumentException("Missing pitch in " + originalNoteSymbolAndScale);
            }
            lastChar = jfugue.charAt(len-1);
        }
        if (!Character.isDigit(lastChar)) {
            throw new IllegalArgumentException("Invalid noteSymbol because of missing digit at end: " + jfugue);
        }
        int scale=Integer.parseInt(""+lastChar);
        String noteSymbol = jfugue.substring(0, len-1);
        Integer value = noteMap.get(noteSymbol);
        if (value==null) {
            throw new IllegalArgumentException("Invalid noteSymbol " + jfugue);
        }
        return scale*12+value.intValue();
    }
    //----------------
    public static void main(String [] args) {
        try {
            for(int i=20;i<=72;i++) {
                String asString = pitchToJFugueNoteString(i);
                System.out.println(i + " " + asString + " " + getPitchFromJFuguePatternString(asString));
            }
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }
}

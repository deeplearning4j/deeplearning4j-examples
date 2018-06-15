package org.deeplearning4j.examples.recurrent.character.harmonies;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;
import org.deeplearning4j.examples.recurrent.character.melodl4j.Note;

/**
 * This class converts Midi to two-party harmony string, saving them in a file (See printTwoPartHarmonies).
 * <p>
 * If you set the variable writeInstrumentsToHarmonyStrings to true, the harmony strings will containn instrument numbers and be twice as long.
 * You should then modify GravesLSTMForTwoPartHarmonies by setting useInstruments to true.
 *
 * @author Don Smith (ThinkerFeeler@gmail.com)
 */
public class MidiHarmonyUtility {
    public static Set<Integer> instrument1RestrictionSet = new TreeSet<>();
    ; // If you add numbers to this, the first voice must have one of the said instruments
    public static Set<Integer> instrument2RestrictionSet = new TreeSet<>(); // If you add numbers to this, the second voice must have one the said instrument s
    public static final char REST_CHAR = ' ';
    public static final String PITCH_CHARACTERS_FOR_HARMONY = makePitchCharactersForHarmony();  // 49 chars. This does NOT include REST_CHAR.
    public static final char FIRST_VALID_PITCH_CHAR= PITCH_CHARACTERS_FOR_HARMONY.charAt(0);
    public static final char LAST_VALID_PITCH_CHAR = PITCH_CHARACTERS_FOR_HARMONY.charAt(PITCH_CHARACTERS_FOR_HARMONY.length()-1);
    public static final int MIN_ALLOWED_PITCH = 36;
    public static final int MAX_ALLOWED_PITCH = 84;

    private static final int MIN_ALLOWED_NUMBER_OF_NOTES = 100;
    private static final long SMALLEST_TIME_INTERVAL_IN_MICROSECONDS = 1_000_000 / 16;
    private static final double MIN_PROPORTION_NON_SILENCE = 0.25;
    private static final double MAX_GAP_PROPORTION = 0.10;
    // Space represents a rest
    private static final Note REST_NOTE = new Note(0, -1, -1, -1, 0);

    private static String makePitchCharactersForHarmony() {
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<49;i++) { // 4 scales plus C of the next scale
            sb.append((char)('A'+i));
        }
        return sb.toString();
    }

    public static boolean isValidPitchCharacter(char ch) {
        return ch == ' '
            || ch >= FIRST_VALID_PITCH_CHAR && ch <= LAST_VALID_PITCH_CHAR;
    }

    public static char getCharForPitch(int pitch) {
        if (pitch == 0) {
            return REST_CHAR;
        }
        while (pitch < MIN_ALLOWED_PITCH) {
            pitch+=12;
        }
        while (pitch>MAX_ALLOWED_PITCH) {
            pitch-=12;
        }
        int offset = pitch - MIN_ALLOWED_PITCH;
        return (char) (FIRST_VALID_PITCH_CHAR + offset);
    }

    /**
     *
     * @param ch symbolic melody string char
     * @return corresponding pitch, or 0 if ch is the REST_CHAR
     */
    public static int getPitchForChar(char ch) {
        if (ch == REST_CHAR) {
            return 0;
        }
        return MIN_ALLOWED_PITCH + (ch - FIRST_VALID_PITCH_CHAR);
    }
    private static class GetNoteOrSilenceByTickFromNoteList {
        private final List<Note> notes;
        private long currentTick = 0;
        private int currentIndex = 0;

        public GetNoteOrSilenceByTickFromNoteList(final List<Note> notes) {
            this.notes = notes;
        }

        /**
         * @return null if we're at the end, REST_NOTE if there's silence, otherwise the Note
         */
        public Note getCurrentNote() {
            if (currentIndex == notes.size()) {
                return null;
            }
            Note note = notes.get(currentIndex);
            if (note.getStartTick() > currentTick) {
                return REST_NOTE;
            }
            return note;
        }

        public void advanceTicks(long ticks) {
            if (currentIndex == notes.size()) {
                return;
            }
            long newCurrentTick = ticks + currentTick;
            while (currentIndex < notes.size()) {
                Note note = notes.get(currentIndex);
                long tickAtEndOfCurrentNote = note.getStartTick() + note.getDurationInTicks();
                if (newCurrentTick >= tickAtEndOfCurrentNote) {
                    currentIndex++;
                } else {
                    break;
                }
            }
            currentTick = newCurrentTick;
        }
    }

    private static final int MAX_PITCH = 127;


    private static int defaultShortestDuration = 30;
    static boolean forceNonZeroVolumeToBeAtLeastHalf = false;
    private final List<TreeMap<Integer, TreeMap<Integer, List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes;

    public MidiHarmonyUtility(final List<TreeMap<Integer, TreeMap<Integer, List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes) {
        this.perTrackMapFromChannelToInstrumentToListOfNotes = perTrackMapFromChannelToInstrumentToListOfNotes;
    }


    private static int adjustPitch(int pitch) {
        if (pitch == 0) {
            return pitch;
        }
        while (pitch > MAX_ALLOWED_PITCH) {
            pitch -= 12;
        }
        while (pitch < MIN_ALLOWED_PITCH) {
            pitch += 12;
        }
        return pitch;
    }

    private static long ticksOfNonSilence(List<Note> notes) {
        long ticks = 0;
        for (Note note : notes) {
            if (note.getPitch() > 0 && note.getVelocity() > 0) {
                ticks += (note.getEndTick() - note.getStartTick());
            }
        }
        return ticks;
    }

    private static long longestGapInTicks(List<Note> notes) {
        long longestGap = 0;
        long lastEndTick = 0;
        for (Note note : notes) {
            if (note.getPitch() > 0 && note.getVelocity() > 0) {
                longestGap = Math.max(longestGap, note.getStartTick() - lastEndTick);
                lastEndTick = note.getEndTick();
            }
        }
        return longestGap;
    }
      public void printTwoPartHarmonies(final String midiFileName, final double microsecondsPerTick, PrintWriter writer) throws IOException {
        // We want to numerically encode the fact that C4 and C5 have C in common, as well as the fact that E is close to F.
        // Representing notes by pitch alone (frequency or midi pitch) encodes the latter but not the former.

        // Seems like a circle is the correct model for the pitch.    Angle encodes the note of scale (C, C# D, etc) and radius encodes scale number (C4, C5, etc).
        //Similarly, a point in a sphere encodes a note:   latitude encodes note of a scale, longitude encodes scale number, and distance from center encodes volume.

        // But the fact that 360 degrees is the same as 0 degrees, and both are close to 10 degrees, is not encoded numerically in a single number. You need a
        // function (sine or cosine).

        final List<List<Note>> voicesEachMonophonicAndOneInstrument = generateMonophonicVoicesOfOneInstrument();
        sortVoicesByAveragePitchIgnoringSpaces(voicesEachMonophonicAndOneInstrument);
        long shortestDurationOfANoteInTicks = Long.MAX_VALUE;
        for (List<Note> list : voicesEachMonophonicAndOneInstrument) {
            for (Note note : list) {
                if (note.getDurationInSeconds(microsecondsPerTick) < MidiMusicExtractor.minimumDurationInSecondsOfNoteToIncludeInOutput) {
                    continue;
                }
                shortestDurationOfANoteInTicks = Math.min(shortestDurationOfANoteInTicks, note.getDurationInTicks());
            }
        }
        int endXOfAllVoices = 0;
        System.out.println("voicesEachMonophonicAndOneInstrument.size() = " + voicesEachMonophonicAndOneInstrument.size());
        for (List<Note> list : voicesEachMonophonicAndOneInstrument) {
            endXOfAllVoices = Math.max(endXOfAllVoices, (int) ((list.get(list.size() - 1).getEndTick()) / shortestDurationOfANoteInTicks));
        }
        double ticksPerMicrosecond = 1.0 / microsecondsPerTick;
        long tickDelta = Math.round(ticksPerMicrosecond * SMALLEST_TIME_INTERVAL_IN_MICROSECONDS);
        int numberOfVoices = voicesEachMonophonicAndOneInstrument.size();
        int countRejectedDueToSilence = 0;
        int countRejectedDueToTooFewNotes = 0;
        int countRejectedDueToLongGap = 0;
        for (int voice1 = 0; voice1 < numberOfVoices; voice1++) {
            List<Note> notes1 = voicesEachMonophonicAndOneInstrument.get(voice1);
            if (notes1.size() < MIN_ALLOWED_NUMBER_OF_NOTES) {
                countRejectedDueToTooFewNotes++;
                continue;
            }
            if (instrument1RestrictionSet.size() > 0 && !instrument1RestrictionSet.contains(notes1.get(0).getInstrument())) {
                continue;
            }
            final long endTick1 = notes1.get(notes1.size() - 1).getEndTick();
            final long ticksOfNonSilence1 = ticksOfNonSilence(notes1);
            if ((0.0 + ticksOfNonSilence1) / endTick1 < MIN_PROPORTION_NON_SILENCE) {
                countRejectedDueToSilence++;
                continue;
            }
            if ((longestGapInTicks(notes1) + 0.0) / endTick1 > MAX_GAP_PROPORTION) {
                countRejectedDueToLongGap++;
                continue;
            }
            for (int voice2 = voice1 + 1; voice2 < numberOfVoices; voice2++) {
                List<Note> notes2 = voicesEachMonophonicAndOneInstrument.get(voice2);
                if (notes2.size() < MIN_ALLOWED_NUMBER_OF_NOTES) {
                    countRejectedDueToTooFewNotes++;
                    continue;
                }
                if (instrument2RestrictionSet.size() > 0 && !instrument2RestrictionSet.contains(notes2.get(0).getInstrument())) {
                    continue;
                }
                final long endTick2 = notes2.get(notes2.size() - 1).getEndTick();
                final long ticksOfNonSilence2 = ticksOfNonSilence(notes2);
                if ((0.0 + ticksOfNonSilence2) / Math.max(endTick1, endTick2) < MIN_PROPORTION_NON_SILENCE) {
                    countRejectedDueToSilence++;
                    continue;
                }
                if ((0.0 + ticksOfNonSilence1) / Math.max(endTick1, endTick2) < MIN_PROPORTION_NON_SILENCE) {
                    countRejectedDueToSilence++;
                    continue;
                }
                if ((longestGapInTicks(notes2) + 0.0) / endTick2 > MAX_GAP_PROPORTION) {
                    countRejectedDueToLongGap++;
                    continue;
                }

                GetNoteOrSilenceByTickFromNoteList selector1 = new GetNoteOrSilenceByTickFromNoteList(notes1);
                GetNoteOrSilenceByTickFromNoteList selector2 = new GetNoteOrSilenceByTickFromNoteList(notes2);
                StringBuilder sb = new StringBuilder();
                while (true) {
                    Note note1 = selector1.getCurrentNote();
                    Note note2 = selector2.getCurrentNote();
                    if (note1 == null || note2 == null) {
                        break;
                    }
                    int pitch1 = adjustPitch(note1.getPitch());
                    int pitch2 = adjustPitch(note2.getPitch());
                    char ch1 = pitch1 == 0 ? REST_CHAR : PITCH_CHARACTERS_FOR_HARMONY.charAt(pitch1 - MIN_ALLOWED_PITCH);
                    char ch2 = pitch2 == 0 ? REST_CHAR : PITCH_CHARACTERS_FOR_HARMONY.charAt(pitch2 - MIN_ALLOWED_PITCH);
                    sb.append(ch1);
                    sb.append(ch2);
                    selector1.advanceTicks(tickDelta);
                    selector2.advanceTicks(tickDelta);
                }
                String line = sb.toString();
                // 	Remove spaces/silence
                String line2 = PlayTwoPartHarmonies.removeSilences(line, 16);
                if ((line.length() - line2.length()) % 2 != 0) {
                    throw new IllegalStateException();
                }
                line = line2;
                if (line.length() >= MIN_ALLOWED_NUMBER_OF_NOTES) {
                    writer.println(line);
                }
            }
        }
        System.out.println("countRejectedDueToSilence = " + countRejectedDueToSilence + ", countRejectedDueToTooFewNotes = " + countRejectedDueToTooFewNotes
            + ", countRejectedDueToLongGap = " + countRejectedDueToLongGap);
    }

    public static double getAveragePitchIgnoringSpaces(List<Note> notes) {
        int sum = 0;
        int count = 0;
        for (Note note : notes) {
            if (note != REST_NOTE && note.getVelocity() > 0 && note.getPitch() > 0) {
                count++;
                sum += note.getPitch();
            }
        }
        return count == 0 ? 0 : (0.0 + sum) / count;
    }

    private static int getIndex(final Object obj, final Object[] objects) {
        for (int i = 0; i < objects.length; i++) {
            if (obj == objects[i]) {
                return i;
            }
        }
        return -1;
    }

    private static boolean validateOrder(List<List<Note>> voices) {
        double lastAvg = 0.0;
        for (List<Note> notes : voices) {
            double avg = getAveragePitchIgnoringSpaces(notes);
            if (avg < lastAvg) {
                return false;
            }
            lastAvg = avg;
        }
        return true;
    }

    // tom-adsfund suggested ordering voices like this (for Bach, where the lower voice and upper voice differ).
    private static void sortVoicesByAveragePitchIgnoringSpaces(List<List<Note>> voicesEachMonophonicAndOneInstrument) {
        final double averagePitches[] = new double[voicesEachMonophonicAndOneInstrument.size()];
        final Object[] voicesList = new Object[voicesEachMonophonicAndOneInstrument.size()];
        for (int i = 0; i < voicesEachMonophonicAndOneInstrument.size(); i++) {
            voicesList[i] = voicesEachMonophonicAndOneInstrument.get(i);
            averagePitches[i] = getAveragePitchIgnoringSpaces(voicesEachMonophonicAndOneInstrument.get(i));
        }
        voicesEachMonophonicAndOneInstrument.sort(new Comparator<List<Note>>() {
            @Override
            public int compare(List<Note> notes1, List<Note> notes2) {
                int index1 = getIndex(notes1, voicesList);
                int index2 = getIndex(notes2, voicesList);
                return Double.compare(averagePitches[index1], averagePitches[index2]);
            }
        });

        assert (validateOrder(voicesEachMonophonicAndOneInstrument));
    }

    private static int adjustVolume127(int volume127) {
        if (forceNonZeroVolumeToBeAtLeastHalf) {
            if (volume127 > 0) {
                return Math.max(63, volume127);
            }
        }
        return volume127;
    }

    private static String getFilenameWithoutSuffix(String name) {
        int index = name.lastIndexOf('.');
        if (index > 0) {
            return name.substring(0, index);
        }
        return name;
    }

    // Generates a list of lists of notes such that each list of notes is monophonic (no harmony) and consists of one instrument.
    // There may be multiple lists of notes for the same instrument.
    private List<List<Note>> generateMonophonicVoicesOfOneInstrument() {
        List<List<Note>> result = new ArrayList<>();
        int count = 0;
        for (Note note : getAllNotesFromPerTrackMapFromChannelToInstrumentToListOfNotes()) {
            //if (count<20) {System.out.println(note);} count++;
            addNoteToVoiceOfSameInstrumentForWhichItOverlapsNoExistingNote(note, result);
        }
        return result;
    }

    private void addNoteToVoiceOfSameInstrumentForWhichItOverlapsNoExistingNote(Note note, List<List<Note>> result) {
        int instrument = note.getInstrument();
        for (List<Note> existingList : result) {
            Note firstNote = existingList.get(0);
            if (firstNote.getInstrument() == instrument && !MidiMusicExtractor.overlapsStrict(note, existingList)) {
                //System.out.println(note);
                existingList.add(note);
                return;
            }
        }
        List<Note> newList = new ArrayList<>();
        result.add(newList);
        newList.add(note);
    }

    private List<Note> getAllNotesFromPerTrackMapFromChannelToInstrumentToListOfNotes() {
        List<Note> allNotes = new ArrayList<>();
        for (TreeMap<Integer, TreeMap<Integer, List<Note>>> map1 : perTrackMapFromChannelToInstrumentToListOfNotes) {
            for (TreeMap<Integer, List<Note>> map2 : map1.values()) {
                for (List<Note> list : map2.values()) {
                    allNotes.addAll(list);
                }
            }
        }
        allNotes.sort(new Comparator<Note>() {
            @Override
            public int compare(Note n1, Note n2) {
                return n1.compareTo(n2);
            }
        });
        return allNotes;
    }

      private static List<Note> findOrCreatListForInstrument(int instrument, List<List<Note>> existingInstrumentNotes) {
        for (List<Note> list : existingInstrumentNotes) {
            if (list.get(0).getInstrument() == instrument) {
                return list;
            }
        }
        List<Note> notes = new ArrayList<>();
        existingInstrumentNotes.add(notes);
        return notes;
    }

    // Let I be the number of distinct instruments appearing in the notes of the file
    // This method returns I lists of notes, one for each instrument. That is, all notes for the same instrument are put into a single track.
    private static List<List<Note>> convertVoiceNotesToInstrumentNotes(List<List<Note>> voiceNotes) {
        List<List<Note>> result = new ArrayList<>();
        for (List<Note> voices : voiceNotes) {
            if (voices.isEmpty()) {
                continue;
            }
            int instrument = voices.get(0).getInstrument();
            List<Note> notes = findOrCreatListForInstrument(instrument, result);
            for (Note note : voices) {
                notes.add(note);
            }
        }
        for (List<Note> notes : result) {
            notes.sort(null);
        }
        return result;
    }

    private static void setInstrument(Track track, int channel, int instrument) throws InvalidMidiDataException {
        if (instrument < 0 || instrument > 128) {
            System.err.println("instrument = " + instrument);
        }
        MidiMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE, channel, instrument, 0);
        track.add(new MidiEvent(midiMessage, 0));
    }


    private static Sequence convertVoicesToSequence(String name, List<List<Note>> voiceNotes1)
        throws FileNotFoundException, InvalidMidiDataException {
        List<List<Note>> instrumentNotesList = convertVoiceNotesToInstrumentNotes(voiceNotes1);
        Sequence sequence = new Sequence(Sequence.PPQ, 120 /*resolution*/);
        int countNotes = 0;
        int channel = 0;
        for (List<Note> instrumentNotes : instrumentNotesList) {
            Track track = sequence.createTrack();
            Note firstNote = instrumentNotes.get(0);
            setInstrument(track, channel, firstNote.getInstrument());
            for (Note note : instrumentNotes) {
                //System.out.println(note);
                note.setChannel(channel);
                note.addMidiEvents(track);
                countNotes++;
            }
            channel++;
            if (channel == 9) {
                channel++; // percussion channel
            }
            if (channel == 16) {
                System.err.println("Warning: more than 16 channels for " + name + ", ignoring extra channels");
                break;
            }
        }
        // Convert each list of instrument notes to a track
        System.out.println("Made sequence with " + sequence.getTracks().length + " tracks and " + countNotes + " notes from " + name);
        return sequence;
    }
}

package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequence;
import javax.sound.midi.Sequencer;
import javax.sound.midi.Track;

/*
 * @author Donald A. Smith
 *
 * A NoteSequence is a sequence of notes played on a single track.
 *
 */
public class NoteSequence implements Comparable<NoteSequence> {
    private static boolean trace = false;
    private long startTick;
    private int trackNumber;
    private int channel;
    private List<Note> notes = new ArrayList<Note>();
    private List<InstrumentChange> instrumentChanges = new ArrayList<InstrumentChange>();
    private final int resolution;
    private int instrument;
    private double averageNoteDuration = -1;

    public NoteSequence(long startTick, int track, int channel, int resolution) {
        this.startTick = startTick;
        this.trackNumber = track;
        this.channel = channel;
        this.resolution = resolution;
    }

    public int getResolution() {
        return resolution;
    }

    public int getMaxRawNote() {
        int max = Integer.MIN_VALUE;
        for (Note note : getNotes()) {
            int val = note.getRawNote();
            if (val > max) {
                max = val;
            }
        }
        return max;
    }

    public double getAverageNoteDuration() {
        if (averageNoteDuration < 0) {
            averageNoteDuration = getAverageNoteDurationExpensive();
        }
        return averageNoteDuration;
    }

    public double getAverageNoteDurationExpensive() {
        long sumDurations = 0;
        int noteCount = 0;
        for (Note note : getNotes()) {
            noteCount++;
            sumDurations += note.getDuration();
        }
        return noteCount == 0 ? 0 : (0.0 + sumDurations) / noteCount;
    }

    public int getMaxPitchGapAbsolute() {
        int max = Integer.MIN_VALUE;
        int lastPitch = Integer.MIN_VALUE;
        for (Note note : getNotes()) {
            int val = note.getRawNote();
            if (lastPitch == Integer.MIN_VALUE) {
                lastPitch = val;
                continue;
            }
            int gap = Math.abs(val - lastPitch);
            if (gap > max) {
                max = gap;
            }
            lastPitch = val;
        }
        return max;
    }

    public int getMinRawNote() {
        int min = Integer.MAX_VALUE;
        for (Note note : getNotes()) {
            int val = note.getRawNote();
            if (val < min) {
                min = val;
            }
        }
        return min;
    }

    public boolean isValid() {
        int min = getMinPitch();
        int max = getMaxPitch();
        if (min >= max || min < 0 || getMaxPitchGapAbsolute() > 16) {
            return false;
        }
        return true;
    }

    public void addInstrumentChange(int instrumentNumber, long startTick) {
        if (instrumentNumber == instrument) {
            if (trace) {
                System.out.println("Duplicate instrument change to " + Midi2MelodyStrings.programs[instrumentNumber]);
            }
            return;
        }
        if (trace) {
            System.out.println("Adding instrument change for " + instrumentNumber
                + " (" + Midi2MelodyStrings.programs[instrumentNumber] + ") for channel " + channel + " at tick " + startTick);
        }
        instrumentChanges.add(new InstrumentChange(instrumentNumber, startTick, channel));
        instrument = instrumentNumber;
    }

    public Sequence toSequence() throws InvalidMidiDataException {
        Sequence sequence = new Sequence(Sequence.PPQ, resolution);
        Track track = sequence.createTrack();
        if (trace) {
            System.out.println("Playing track " + trackNumber + ", channel " + channel);
        }

        for (InstrumentChange change : instrumentChanges) {
            change.addMidiEvents(track);
        }
        for (Note note : notes) {
            note.addMidiEvents(track);
        }
        return sequence;
    }

    public void play(Sequencer sequencer) throws MidiUnavailableException, InvalidMidiDataException {
        Sequence sequence = toSequence();
        sequencer.setSequence(sequence);
        sequencer.setTickPosition(0);
        sequencer.open();
        sequencer.start();
    }

    public long getStartTick() {
        return startTick;
    }

    public long getEndTick() {
        return notes.get(notes.size() - 1).getEndTick();
    }

    public int getNumberOfDistinctPitches() {
        int counts[] = new int[128];
        int count = 0;
        for (Note note : getNotes()) {
            if (counts[note.getRawNote()] == 0) {
                count++;
            }
            counts[note.getRawNote()]++;
        }
        if (count < 3) {
            System.out.print(count + " ");
        }
        return count;
    }

    public long getDuration() {
        return getEndTick() - getStartTick();
    }

    public double getProportionSilence() {
        long totalDuration = getDuration();
        long totalRests = 0;
        Note lastNote = null;
        for (Note note : notes) {
            if (lastNote != null) {
                totalRests += note.getStartTick() - lastNote.getEndTick();
            }
            lastNote = note;
        }
        return (0.0 + totalRests) / totalDuration;
    }

    public long getLongestNoteDuration() {
        long longest = 0;
        for (Note note : getNotes()) {
            if (note.getDuration() > longest) {
                longest = note.getDuration();
            }
        }
        return longest;
    }

    public Iterable<Note> getNotes() {
        final Iterator<Note> iterator = notes.iterator();
        return new Iterable<Note>() {
            @Override
            public Iterator<Note> iterator() {
                return iterator;
            }
        };
    }

    public long getShortetNoteDuration() {
        long shortest = Long.MAX_VALUE;
        for (Note note : getNotes()) {
            long duration = note.getDuration();
            if (duration > 0 && duration < shortest) {
                shortest = note.getDuration();
            }
        }
        return shortest;
    }

    public long getLongestRest() {
        long longest = 0;
        Note lastNote = notes.get(0);
        for (Note note : notes) {
            long rest = note.getStartTick() - lastNote.getEndTick();
            if (rest > longest) {
                longest = rest;
            }
            lastNote = note;
        }
        return longest;
    }

    public int getTrack() {
        return trackNumber;
    }

    public int getChannel() {
        return channel;
    }

    public void verifyMonotonicIncreasing() {
        NoteOrInstrumentChange previous = null;
        for (NoteOrInstrumentChange noteOrInstrumentChange : notes) {
            if (previous != null) {
                if (previous.getStartTick() > noteOrInstrumentChange.getStartTick()) {
                    System.err.println("Not monitonic: " + previous + " and " + noteOrInstrumentChange);
                }
            }
            previous = noteOrInstrumentChange;
        }
    }

    // Return count removed
    public int removeAllButHigherOrLowerNotes(boolean higher) {
        //throw new RuntimeException("Not implemented yet");
        int countRemoved = 0;
        Iterator<Note> iterator = notes.iterator();
        while (iterator.hasNext()) {
            Note note = iterator.next();
            if (aHigherOrLowerPitchedNoteOverlapsThisNote(note, higher)) {
                iterator.remove();
                countRemoved++;
            }
        }
        return countRemoved;
    }

    private boolean aHigherOrLowerPitchedNoteOverlapsThisNote(Note note1, boolean higher) {
        for (Note note2 : getNotes()) {
            if (note2.getStartTick() >= note1.getEndTick()) {
                break;
            }
            if ((higher ? note2.getRawNote() > note1.getRawNote() : note2.getRawNote() < note1.getRawNote()) && ticksOverlapInTime(note1, note2)) {
                return true;
            }

        }
        return false;
    }

    private boolean ticksDontOverlapInTime(Note note1, Note note2) {
        return note1.getEndTick() <= note2.getStartTick() || note2.getEndTick() <= note1.getStartTick();
    }

    private boolean ticksOverlapInTime(Note note1, Note note2) {
        return !ticksDontOverlapInTime(note1, note2);
    }

    public int countOfNotesHavingPolyphony() {
        verifyMonotonicIncreasing();
        Set<Note> notesOn = new HashSet<Note>();
        int count = 0;
        for (Note note : getNotes()) {
            Iterator<Note> iterator = notesOn.iterator();
            while (iterator.hasNext()) {
                Note onNote = iterator.next();
                if (note.getStartTick() >= onNote.endTick()) {
                    iterator.remove();
                } else {
                    count++;
                }
            }
            notesOn.add(note);
        }
        return count;
    }

    @Override
    public int compareTo(NoteSequence other) {
        int diff = trackNumber - other.trackNumber;
        if (diff != 0) {
            return diff;
        }
        diff = channel - other.channel;
        if (diff != 0) {
            return diff;
        }
        long diffLong = startTick - other.startTick;
        if (diffLong > 0) {
            return 1;
        }
        if (diffLong < 0) {
            return -1;
        }
        return 0;
    }

    public boolean equals(Object other) {
        return compareTo((NoteSequence) other) == 0;
    }

    public void removeLeadingSilence() {
        long firstRealNoteTick = getFirstRealNoteTick();
        if (firstRealNoteTick >= 0) {
            for (NoteOrInstrumentChange note : notes) {
                if (note.getStartTick() >= firstRealNoteTick) {
                    note.setStartTick(1 + note.getStartTick() - firstRealNoteTick);
                }
            }
        }
    }

    private long getFirstRealNoteTick() {
        return notes.size() > 0 ? notes.get(0).getStartTick() : -1;
    }

    public void add(Note note) {
        averageNoteDuration = -1;
        notes.add(note);
    }

    public long getLastTick() {
        return notes.isEmpty() ? 0L : notes.get(notes.size() - 1).getStartTick();
    }

    public void toString(StringBuilder sb, boolean verbose) {
        sb.append("NoteSequence with track = " + trackNumber);
        sb.append(", channel = " + channel);
        sb.append(", noteCount = " + notes.size());
        sb.append(", count of polyphonic notes = " + countOfNotesHavingPolyphony());
        sb.append(", startTick = " + startTick);
        sb.append(", and " + notes.size() + " notes");
        if (verbose) {
            for (NoteOrInstrumentChange note : notes) {
                sb.append("  ");
                sb.append(note);
                sb.append("\n");
            }
        }
    }

    public int getMinPitch() {
        int minPitch = Integer.MAX_VALUE;
        for (Note note : notes) {
            if (note.getRawNote() < minPitch) {
                minPitch = note.getRawNote();
            }
        }
        return minPitch;
    }

    public int getMaxPitch() {
        int maxPitch = 0;
        for (Note note : notes) {
            if (note.getRawNote() > maxPitch) {
                maxPitch = note.getRawNote();
            }
        }
        return maxPitch;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        toString(sb, false);
        return sb.toString();
    }

    public String toString(boolean verbose) {
        StringBuilder sb = new StringBuilder();
        toString(sb, verbose);
        return sb.toString();
    }

    public int getLength() {
        return notes.size();
    }

    public Note get(int i) {
        return notes.get(i);
    }

    public int getLengthOfLongestSequenceOfRepeatedNotes() {
        int count = 0;
        int max = 0;
        int lastRawNote = -1;
        for (Note note : getNotes()) {
            int rawNote = note.getRawNote();
            if (rawNote == lastRawNote) {
                count++;
                if (count > max) {
                    max = count;
                }
            } else {
                count = 0;
            }
            lastRawNote = rawNote;
        }
        return max;
    }

    public int getNumberOfRepeatedNotes() {
        int count = 0;
        int lastRawNote = -1;
        for (Note note : getNotes()) {
            int rawNote = note.getRawNote();
            if (rawNote == lastRawNote) {
                count++;
            }
            lastRawNote = rawNote;
        }
        return count;
    }

    public double getNumberOfNotes() {
        return notes.size();
    }
}


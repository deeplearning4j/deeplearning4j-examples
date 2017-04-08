package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.util.Set;
import java.util.TreeSet;

/*
 * @author Donald A. Smith
 */
public class Piece {
    Set<NoteSequence> noteSequences = new TreeSet<NoteSequence>();
    int noteCount = 0;
    long totalNoteDurationInTicks = 0;
    final long totalPieceDurationInTicks;
    final int resolution;

    public Piece(long duration, int resolution) {
        totalPieceDurationInTicks = duration;
        this.resolution = resolution;
    }

    public Set<NoteSequence> findOpenTracksUsing(int channel) {
        Set<NoteSequence> set = new TreeSet<NoteSequence>();
        for (NoteSequence ns : noteSequences) {
            if (ns.getChannel() == channel) {
                set.add(ns);
            }
        }
        return set;
    }

    public NoteSequence findTrack(int track, int channel) {
        for (NoteSequence noteSequence : noteSequences) {
            if (noteSequence.getTrack() == track && noteSequence.getChannel() == channel) {
                return noteSequence;
            }
        }
        return null;
    }

    public double getDensity() {
        return (0.0 + totalNoteDurationInTicks) / totalPieceDurationInTicks;
    }

    public void addNote(int track, int channel, Note note) {
        noteCount++;
        NoteSequence noteSequence = findTrack(track, channel);
        if (noteSequence == null) {
            noteSequence = new NoteSequence(note.startTick, track, channel, resolution);
            noteSequences.add(noteSequence);
        }
        note.indexInNoteSequence = noteSequence.getLength();
        noteSequence.add(note);
        totalNoteDurationInTicks += note.getDuration();
    }

    public void toString(StringBuilder sb, boolean verbose) {
        sb.append(noteCount + " notes and " + noteSequences.size() + " sequences:\n");
        for (NoteSequence ns : noteSequences) {
            sb.append("  ");
            ns.toString(sb, verbose);
            sb.append("\n");
        }
        sb.append("Note density = " + getDensity() + "\n");
    }
}

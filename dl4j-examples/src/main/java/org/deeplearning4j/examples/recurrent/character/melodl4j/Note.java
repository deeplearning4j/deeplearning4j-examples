package org.deeplearning4j.examples.recurrent.character.melodl4j;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

public class Note implements Comparable<Note> {
    private final int pitch;
    private final long startTick;
    private long endTick;
    private final Integer instrument;
    private final Integer channel;
    private final int velocity;
    public Note(int pitch, long startTick, int instrument, int channel, int velocity) {
        this.pitch=pitch;
        this.startTick=startTick;
        this.instrument=instrument;
        this.channel=channel;
        this.velocity=velocity;
    }

    @Override
    public String toString() {
        return "Pitch " + pitch + " starting at " + startTick + " ending at " + endTick + " on channel " + channel + " with instrument " + instrument;
    }
    public long getEndTick() {
        return endTick;
    }
    public void setEndTick(long endTick) {
        this.endTick = endTick;
    }
    public int getPitch() {
        return pitch;
    }
    public long getStartTick() {
        return startTick;
    }
    public double getStartSeconds(double microsecondsPerTick) {
        return 0.001*microsecondsPerTick*getStartTick();
    }
    public double getEndSeconds(double microsecondsPerTick) {
        return 0.001*microsecondsPerTick*getEndTick();
    }
    public int getInstrument() {
        return instrument;
    }
    public int getChannel() {
        return channel;
    }
    public int getVelocity() {
        return velocity;
    }
    public long getDuration() {
        return endTick-startTick;
    }
    @Override
    public int compareTo(Note other) {
        long diffL=startTick-other.startTick;
        if (diffL!=0) {
            return diffL>0?1:-1;
        }
        int diff=pitch-other.pitch;
        if (diff!=0) {
            return diff;
        }
        diff = channel-other.channel;
        if (diff!=0) {
            return diff;
        }
        return 0;
    }
    @Override
    public boolean equals(Object obj) {
        Note other=(Note)obj;
        return startTick==other.startTick  && pitch==other.pitch && channel==other.channel;
    }
    public void addMidiEvents(Track track) throws InvalidMidiDataException {
        MidiMessage midiMessageStart=new ShortMessage(ShortMessage.NOTE_ON,channel,pitch,velocity);
        track.add(new MidiEvent(midiMessageStart,startTick));
        MidiMessage midiMessageEnd=new ShortMessage(ShortMessage.NOTE_OFF,channel,pitch,0);
        track.add(new MidiEvent(midiMessageEnd,endTick));
    }
} // class Note

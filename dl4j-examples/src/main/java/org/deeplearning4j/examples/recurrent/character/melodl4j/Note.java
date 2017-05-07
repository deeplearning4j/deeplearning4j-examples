package org.deeplearning4j.examples.recurrent.character.melodl4j;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

public class Note extends NoteOrInstrumentChange {
	private final int rawNote; // midi note value
	private final int velocity;
	private final int channel;
	int indexInNoteSequence;
	private long durationInTicks; // set later

	public Note(long startTick,int rawNote, int velocity, int channel) {
		this.startTick=startTick;
		this.rawNote=rawNote;
		this.velocity=velocity;
		this.channel=channel;
	}
	public int getRawNote() {
		return rawNote;
	}
	public int getKey() {
		return rawNote%12;
	}
	public void setDuration(long durationInTicks) {
		this.durationInTicks=durationInTicks;
	}
	public int getOctave() {
		return rawNote/12 -1;
	}
	public int getChannel() {
		return channel;
	}
	public int getVelocity() {
		return velocity;
	}
	public int interval(Note other) {
		return getRawNote()-other.getRawNote();
	}
	public long endTick() {
		return startTick+durationInTicks;
	}
	@Override
	public String toString() {
		long endTick = startTick + durationInTicks;
		return "Note[startTick: " + startTick + ", endTick = " + endTick + ", duration: " + durationInTicks
				+ ", rawNote:" + rawNote
				+ ", note: " + getKey() + ", octave: " + getOctave() + ", channel: " + channel + ", velocity: " + velocity + "] ";
	}
	public long getDuration() {
		return durationInTicks;
	}
	@Override
	public void addMidiEvents(Track track) throws InvalidMidiDataException {
		MidiMessage midiMessageStart=new ShortMessage(ShortMessage.NOTE_ON,channel,rawNote,velocity);
		track.add(new MidiEvent(midiMessageStart,startTick));
		MidiMessage midiMessageEnd=new ShortMessage(ShortMessage.NOTE_OFF,channel,rawNote,0);
		track.add(new MidiEvent(midiMessageEnd,startTick+durationInTicks));
	}
	public long getEndTick() {
		return startTick+durationInTicks;
	}
}

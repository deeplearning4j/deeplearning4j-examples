package org.deeplearning4j.examples.recurrent.character.melodl4j;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.Track;

public abstract class NoteOrInstrumentChange {
	protected long startTick;
	abstract void addMidiEvents(Track track) throws InvalidMidiDataException;
	public long getStartTick() {
		return startTick;
	}
	public void setStartTick(long tick) {
		startTick=tick;
	}
}

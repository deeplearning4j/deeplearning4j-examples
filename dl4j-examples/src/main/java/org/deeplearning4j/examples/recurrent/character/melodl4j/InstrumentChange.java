package org.deeplearning4j.examples.recurrent.character.melodl4j;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

public class InstrumentChange extends NoteOrInstrumentChange {
	private final int instrumentNumber;
	private final int channel;
	public InstrumentChange(int instrumentNumber, long tick, int channel) {
		this.instrumentNumber=instrumentNumber;
		this.startTick=tick;
		this.channel=channel;
	}
	@Override
	public String toString() {
		return "Change instrument to " + instrumentNumber + " (" + Midi2MelodyStrings.programs[instrumentNumber] + ") at " + startTick;
	}
	public int getInstrumentNumber() {
		return instrumentNumber;
	}
	@Override
	public void addMidiEvents(Track track) throws InvalidMidiDataException {
		MidiMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE,channel,instrumentNumber,0);
		System.out.println("Adding instrument change to track for channel " + channel + " and instrumentName = " + Midi2MelodyStrings.programs[instrumentNumber]);
		track.add(new MidiEvent(midiMessage, startTick));
	}
}

package org.deeplearning4j.examples.recurrent.character.harmonies;

/**
 * This class was borrowed and modified from JFugue  http://www.jfugue.org/, which operates under an Apache License
 */

import java.io.BufferedReader;

/*
 * JFugue - API for Music Programming
 * Copyright (C) 2003-2008  Karl Helgason and David Koelle
 *
 * http://www.jfugue.org
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MetaMessage;
import javax.sound.midi.MidiDevice;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Receiver;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Synthesizer;
import javax.sound.midi.Track;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import com.sun.media.sound.AudioSynthesizer;

@SuppressWarnings("restriction")
public class Midi2WavRenderer {
	public final static int MIDI_PERCUSSION_CHANNEL=9;
    private AudioSynthesizer synth;
    
    static {
    	PlayMusic.loadSoundBank(); 
    }
    
    public Midi2WavRenderer() throws MidiUnavailableException, InvalidMidiDataException, IOException {
        this.synth = findAudioSynthesizer(); // MidiSystem.getSynthesizer();
    }

    public static void createMp3File(Sequence sequence,String outputMp3FilePath, int transposeHalfSteps) throws Exception {
    	File temp = File.createTempFile("midi-wav", ".wav");
    	createWavFile(sequence, temp.getAbsolutePath(),transposeHalfSteps);
    	StringBuilder sb= new StringBuilder();
    	File outFile = new File(outputMp3FilePath);
    	if (outFile.exists()) {outFile.delete();}
    	PlayMusic.convertWavToMp3(temp.getAbsolutePath(), outputMp3FilePath, sb);
    	System.out.println("Wrote MP3 to " + outputMp3FilePath + (sb.length()==0 ? "": " with error: " + sb.toString()));
    }
    public static void createWavFile(Sequence sequence,String outputWaveFilePath, int transposeHalfSteps) throws Exception {
    	if (transposeHalfSteps!=0) {
    		transpose(sequence, transposeHalfSteps);
    	}
    	transpose(sequence,-4);
    	Midi2WavRenderer renderer = new Midi2WavRenderer();
    	renderer.createWavFile(sequence,new File(outputWaveFilePath));
    }
    public static void createWavFile(String inputMidiFilePath, String outputWaveFilePath, int transposeHalfSteps) throws Exception {
    	Sequence sequence =MidiSystem.getSequence(new File(inputMidiFilePath));
    	createWavFile(sequence,outputWaveFilePath,transposeHalfSteps);
    }

    private static void transpose(final Sequence sequence, final int halfSteps) throws InvalidMidiDataException {
    	for(Track track:sequence.getTracks()) {
    		for (int eventNumber = 0; eventNumber < track.size(); eventNumber++) {
				MidiEvent event = track.get(eventNumber);
				long tick = event.getTick();
				MidiMessage message = event.getMessage();
				if (message instanceof ShortMessage) {
					ShortMessage shortMessage = (ShortMessage) message;
					if (shortMessage.getChannel()==MIDI_PERCUSSION_CHANNEL) {
						continue;
					}
					switch (shortMessage.getCommand()) {
					case ShortMessage.NOTE_ON: case ShortMessage.NOTE_OFF: 
						int pitch=shortMessage.getData1();
						pitch+=halfSteps;
						shortMessage.setMessage(shortMessage.getCommand(),shortMessage.getChannel(),pitch,shortMessage.getData2());
						break;
						default:
							break;
					}
				}
    		}
    	}
	} // void transpose

	/**
     * Creates a WAV file based on the Sequence, using the default soundbank.
     *  
     * @param sequence
     * @param outputWavFile
     * @throws MidiUnavailableException
     * @throws InvalidMidiDataException
     * @throws IOException
     */
    public void createWavFile(Sequence sequence, File outputWavFile) throws MidiUnavailableException, InvalidMidiDataException, IOException {
       //AudioFormat format = new AudioFormat(96000, 24, 2, true, false);
        AudioFormat format = new AudioFormat(44100, 16, 1, true, false);  // worked!
      //  AudioFormat format = new AudioFormat(22050, 16, 1, true, false);  // das
        Map<String, Object> p = new HashMap<String, Object>();
        p.put("interpolation", "sinc");
        p.put("max polyphony", "1024");
        synth.close();
        AudioInputStream stream = synth.openStream(format, p);

        // Play Sequence into AudioSynthesizer Receiver.
        double total = send(sequence, synth.getReceiver());

        // Calculate how long the WAVE file needs to be.
        long len = (long) (stream.getFormat().getFrameRate() * (total + 4));
        stream = new AudioInputStream(stream, stream.getFormat(), len);

        // Write WAVE file to disk.
        AudioSystem.write(stream, AudioFileFormat.Type.WAVE, outputWavFile);
        this.synth.close();
    }
    public void createWavAndSendToOutputStream(Sequence sequence, OutputStream outputStream) throws MidiUnavailableException, InvalidMidiDataException, IOException {
        //AudioFormat format = new AudioFormat(96000, 24, 2, true, false);
         AudioFormat format = new AudioFormat(44100, 16, 1, true, false);  // das
         Map<String, Object> p = new HashMap<String, Object>();
         p.put("interpolation", "sinc");
         p.put("max polyphony", "1024");
         synth.close();
         AudioInputStream stream = synth.openStream(format, p);

         // Play Sequence into AudioSynthesizer Receiver.
         double total = send(sequence, synth.getReceiver());

         // Calculate how long the WAVE file needs to be.
         long len = (long) (stream.getFormat().getFrameRate() * (total + 4));
         stream = new AudioInputStream(stream, stream.getFormat(), len);

         AudioSystem.write(stream, AudioFileFormat.Type.WAVE, outputStream);
         this.synth.close();
     }
         

	/**
	 * Find available AudioSynthesizer.
	 */
	private AudioSynthesizer findAudioSynthesizer() throws MidiUnavailableException {
		// First check if default synthesizer is AudioSynthesizer.
		Synthesizer synth = MidiSystem.getSynthesizer();
		if (synth instanceof AudioSynthesizer) {
			return (AudioSynthesizer)synth;
		}

		// If default synthesizer is not AudioSynthesizer, check others.
		MidiDevice.Info[] midiDeviceInfo = MidiSystem.getMidiDeviceInfo();
		for (int i = 0; i < midiDeviceInfo.length; i++) {
			MidiDevice dev = MidiSystem.getMidiDevice(midiDeviceInfo[i]);
			if (dev instanceof AudioSynthesizer) {
				return (AudioSynthesizer) dev;
			}
		}
		// No AudioSynthesizer was found, return null.
		return null;
	}

	/**
	 * Send entry MIDI Sequence into Receiver using timestamps.
	 */
	private double send(Sequence seq, Receiver recv) {
		float divtype = seq.getDivisionType();
		assert (seq.getDivisionType() == Sequence.PPQ);
		Track[] tracks = seq.getTracks();
		int[] trackspos = new int[tracks.length];
		int mpq = 500000;
		int seqres = seq.getResolution();
		long lasttick = 0;
		long curtime = 0;
		while (true) {
			MidiEvent selevent = null;
			int seltrack = -1;
			for (int i = 0; i < tracks.length; i++) {
				int trackpos = trackspos[i];
				Track track = tracks[i];
				if (trackpos < track.size()) {
					MidiEvent event = track.get(trackpos);
					if (selevent == null
							|| event.getTick() < selevent.getTick()) {
						selevent = event;
						seltrack = i;
					}
				}
			}
			if (seltrack == -1)
				break;
			trackspos[seltrack]++;
			long tick = selevent.getTick();
			if (divtype == Sequence.PPQ)
				curtime += ((tick - lasttick) * mpq) / seqres;
			else
				curtime = (long) ((tick * 1000000.0 * divtype) / seqres);
			lasttick = tick;
			MidiMessage msg = selevent.getMessage();
			if (msg instanceof MetaMessage) {
				if (divtype == Sequence.PPQ)
					if (((MetaMessage) msg).getType() == 0x51) {
						byte[] data = ((MetaMessage) msg).getData();
						mpq = ((data[0] & 0xff) << 16)
								| ((data[1] & 0xff) << 8) | (data[2] & 0xff);
					}
			} else {
				if (recv != null)
					recv.send(msg, curtime);
			}
		}
		return curtime / 1000000.0;
	}
	//...........
	private static String[] loadMelodies(String inputFilePath, int count) throws IOException {
		String [] strings = new String[count];
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFilePath)));
		for(int i=0;i<count;i++) {
			String line=reader.readLine();
			if (line==null) {
				break;
			} else {
				strings[i]=line;
			}
		}
		reader.close();
		return strings;
	}

	//..............
	public static void main(String [] args) {
		// In JFugue, Middle C is 60 = C5
		try { //  36  37 38 39 40 41  42  43  44 45  46 47 48
			  //   C  C#  D  D# E  F  F#   g  G#  A  A#  B  C    SCALE
			//saveMelodyToWav("s", "d:/tmp/piano/C4.wav",48);
			//saveMelodiesToWav("d:/tmp/melodies/matches.txt", 8, "d:/tmp/matches1.wav");
			createWavFile("d:/Music/MIDI/clean_midi/The Beatles/All You Need Is Love.3.mid","d:/tmp/All-You-Need-Is-Love3.wav",0);
		} catch (Exception exc) {
			exc.printStackTrace();
			System.exit(1);
		}
	}
}

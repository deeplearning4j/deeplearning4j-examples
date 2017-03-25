package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.Random;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequencer;
import javax.sound.midi.Soundbank;
import javax.sound.midi.Synthesizer;
import org.apache.commons.io.FileUtils;

public class PlaySymbolic {
	private static Random random = new Random();
    final static String tempDir = System.getProperty("java.io.tmpdir");
    static {
        loadSoundBank(); // Do this statically, because it will persist.
    }
    //-----------------------------------
    public static void main(String[] args)  {
        try {
            String pathToMelodiesFile= args.length==0?  getPathToExampleMelodiesFile() : args[0];
            playMelodies(pathToMelodiesFile,"Acoustic Grand Piano",15);
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }
    //-----------------------------------------------
    public static void playMelodies(String inFilepath, int secondsToPlay) throws IOException, MidiUnavailableException, InvalidMidiDataException {
        playMelodies(inFilepath,"Acoustic Grand Piano",secondsToPlay);
    }
    /* Plays midi file through the system's midi sequencer.
     *  @param inFilepath points to a local file containing symbolic melodies
     *  @param instrumentName : See the list in Midi2Symbolic.java. If null, defaults to "Acoustic Grand Piano".
     *  @param secondsToPlay -- seconds after which to stop the music playback.
     */
	public static void playMelodies(String inFilepath, String instrumentName, int secondsToPlay) throws IOException, MidiUnavailableException, InvalidMidiDataException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inFilepath)));
		int instrumentNumber = instrumentName==null? 0: Midi2Symbolic.getInstrument(instrumentName);
		int lineNumber=0;
		while (true) {
			String line=reader.readLine();
			if (line==null) {
				break;
			}
			lineNumber++;
			line=line.trim();
			if (line.equals("") || line.startsWith("#")) {
				continue;
			}
			System.out.println("Playing " + lineNumber + " : " + line);
			if (instrumentName==null) {
				instrumentNumber= inFilepath.contains("pop") ? chooseRandomInstrumentPop() :chooseRandomInstrumentClassical();
			}
			int lastRawNote =  48+random.nextInt(6);
            playMelody(line, lastRawNote,instrumentNumber, secondsToPlay);
		}
		reader.close();
	}
	private static void sleep(long mls) {
		try {Thread.sleep(mls);} catch (InterruptedException exc) {System.err.println("Interrupted"); Thread.interrupted();}
	}
	private static int chooseRandomInstrumentPop() {
		switch (random.nextInt(3)) {
			case 0:  return Midi2Symbolic.getInstrument("Acoustic Guitar (steel)");
			case 1:  return Midi2Symbolic.getInstrument("Acoustic Guitar (nylon)");
			default: return 0;
		}
	}
	private static int chooseRandomInstrumentClassical() {
		switch (random.nextInt(10)) {
			case 0:  return Midi2Symbolic.getInstrument("Violin");
			case 1:  return Midi2Symbolic.getInstrument("Cello");
			case 2:  return Midi2Symbolic.getInstrument("Church Organ");
			case 3:  return Midi2Symbolic.getInstrument("Flute");
			case 4:  return Midi2Symbolic.getInstrument("Choir Aahs");
			case 5:  return Midi2Symbolic.getInstrument("String Ensemble 1");
			case 6:  return Midi2Symbolic.getInstrument("Acoustic Guitar (nylon)");
			case 7:  return Midi2Symbolic.getInstrument("Trumpet");
			default: return 0;
		}
	}

	// return -1 if it's not a duration
	private static int getDurationInTicks(char ch, int resolutionDelta) {
		int indexOf=Midi2Symbolic.durationChars.indexOf(ch);
		return indexOf<0? -1: resolutionDelta*(1+indexOf);
	}
	// return -1 if it's not a pitch
	private static int getPitchDelta(char ch) {
		int index = Midi2Symbolic.noteGapCharsPositive.indexOf(ch);
		if (index>=0) {
			return index;
		}
		index = Midi2Symbolic.noteGapCharsNegative.indexOf(ch);
		if (index<0) {
			return -1;
		}
		return -(index+1);
	}
	private static boolean isDurationChar(char ch) {
		return ch!='R' &&  Midi2Symbolic.durationChars.indexOf(ch)>=0;
	}
	private static boolean isPitchChar(char ch) {
		return ch!='R' && !isDurationChar(ch);
	}
	public static void playMelody(String melody, int secondsToPlay) throws Exception {
	    //0 is Acoustic Grand Piano
        playMelody(melody,48,0,secondsToPlay);
    }
	// This method will try to play a melody even if the string is malformed.  The neural networks sometimes output invalid melody strings.
	private static void playMelody(String melody, int lastRawNote, int instrumentNumber, int secondsToPlay) throws MidiUnavailableException, InvalidMidiDataException {
		// First char is noteDuration
		// Next: are  restDuration, pitch, noteDuration
		int channel=0;
		if (instrumentNumber == Midi2Symbolic.getInstrument("Flute")) {
			lastRawNote+=12;
		}
		if (instrumentNumber == Midi2Symbolic.getInstrument("Violin")) {
			lastRawNote+=6;
		}
		int velocity=80;
		int track=2;
		final int resolution=480;
		final int resolutionDelta = resolution/16;
		int index = 0; //getIndexOfFirstPitchDuration(line);
		NoteSequence ns = new NoteSequence(0, track, channel, resolution);
		ns.addInstrumentChange(instrumentNumber, 0);
		//ns.add(note);
		index++;
		int noteDurationInTicks=0;
		long tick = 0; // end of note
		while (index<melody.length()-2) {
			char ch = melody.charAt(index);
			if (ch=='R') {
				index++;
				ch = melody.charAt(index);
				if (isDurationChar(ch)) {
					tick+= getDurationInTicks(ch,resolutionDelta);
					index++;
				} else {
					System.out.print('R'); // Badly formed melody string
				}
			} else if (isPitchChar(ch)) {
				index++;
				int pitchDelta = getPitchDelta(ch);
				lastRawNote += pitchDelta;
				while (lastRawNote<30) {
					System.out.print('<');
					lastRawNote+=12; // This is a hack to prevent melodies from becoming inaudible
				}
				while (lastRawNote>=80) {
					System.out.print('>');
					lastRawNote-=12; // This is a hack to prevent melodies from becoming inaudible
				}
				ch = melody.charAt(index);
				if (isDurationChar(ch)) {
					noteDurationInTicks=getDurationInTicks(ch,resolutionDelta);
					index++;
				} else {
					System.out.print('D'); // Badly formed melody string
					noteDurationInTicks = 4*resolutionDelta;
				}
				Note note = new Note(tick, lastRawNote,velocity,channel);
				note.setDuration(noteDurationInTicks);
				ns.add(note);
				tick += noteDurationInTicks;
			} else {
				System.out.print(ch);
				index++;
			}
		}
		int numberDistinct = ns.getNumberOfDistinctPitches();
		if (numberDistinct<3) {
			System.err.println("Warning: only " + numberDistinct + " distinct notes, skipping: " +melody);
			sleep(2000);
			return;
		}
		Sequencer sequencer = MidiSystem.getSequencer();
		System.out.println(",  with " + ns.getLength() + " notes");
		ns.play(sequencer);
		long tickLength=sequencer.getTickLength();
		//JOptionPane.showMessageDialog(null, "Click enter to abort.");
		long startTime=System.currentTimeMillis();
		while (sequencer.getTickPosition()< tickLength) {
			sleep(100);
			long now = System.currentTimeMillis();
			if (now-startTime>secondsToPlay*1000) {
				sequencer.stop();
				tickLength=0;
			}
		}
		sleep(2000); // So that there's a noticeable gap between melodies
	}
    //---------------------------------
    private static void loadSoundBank() {// Download for higher quality MIDI
        final String filename="GeneralUser_GS_SoftSynth.sf2";  // FreeFont.sf2   Airfont_340.dls
        final String soundBankLocation = tempDir + "/" + filename;
        File file = new File(soundBankLocation);
        try {
            if (!file.exists()) {
                System.out.println("Downloading soundbank (first time only!). This may take a while.");
                FileUtils.copyURLToFile(new URL("http://truthsite.org/music/" + filename), file);
                System.out.println("Soundbank downloaded to " + file.getAbsolutePath());
            }
            Synthesizer synth =MidiSystem.getSynthesizer();
            Soundbank deluxeSoundbank;
            deluxeSoundbank=MidiSystem.getSoundbank(file);
            synth.loadAllInstruments(deluxeSoundbank);
            System.out.println("Loaded soundbank from " + soundBankLocation);
        } catch (Exception exc) {
            System.err.println("Unable to load soundbank " + soundBankLocation + " due to " + exc.getMessage()
                + ", using default soundbank.");
        }
    }
	private static String getPathToExampleMelodiesFile() throws Exception {
        String filename="composed-in-the-style-of-bach.txt"; // These melodies were composed by MelodyModelingExample.java.
        //filename = "composed-in-the-style-of-pop.txt";
        String fileLocation = tempDir + "/" + filename;
        File file = new File(fileLocation);
        if( !file.exists() ){
            FileUtils.copyURLToFile(new URL("http://truthsite.org/music/" + filename), file);
            System.out.println("Melody file downloaded to " + file.getAbsolutePath());
        } else {
            System.out.println("Using existing melody file at " + file.getAbsolutePath());
        }
        return fileLocation;
    }
}

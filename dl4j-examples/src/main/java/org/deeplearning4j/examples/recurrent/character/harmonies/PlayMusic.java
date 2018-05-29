package org.deeplearning4j.examples.recurrent.character.harmonies;

import org.deeplearning4j.examples.recurrent.character.melodl4j.Note;
import org.deeplearning4j.examples.recurrent.character.melodl4j.PlayMelodyStrings;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Pattern;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequence;
import javax.sound.midi.Sequencer;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Soundbank;
import javax.sound.midi.Synthesizer;
import javax.sound.midi.Track;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

/*
 *  This plays melody strings and MIDI sequences using your operating system's software synthesizer.
 *
 * @author Donald A. Smith (ThinkerFeeler@gmail.com)
 */
public class PlayMusic {
    private static Random random = new Random();
    private final static String tempDir = System.getProperty("java.io.tmpdir");
    private static Map<String, Integer> instrumentsByName = new HashMap<>();
    //http://en.wikipedia.org/wiki/General_MIDI
    public static final String[] programs = {
        "Acoustic Grand Piano",    //0
        "Bright Acoustic Piano",   //1
        "Electric Grand Piano",    //2
        "Honky-tonk Piano",        //3
        "Electric Piano 1",        //4
        "Electric Piano 2",        //5
        "Harpsichord",             //6
        "Clavinet",                //7
        "Celesta",                 //8
        "Glockenspiel",            //9
        "Music Box",               //10
        "Vibraphone",              //12
        "Marimba",                 //13
        "Xylophone",               //14
        "Tubular Bells",           //15
        "Dulcimer",                //16
        "Drawbar Organ",           //17
        "Percussive Organ",        //18
        "Rock Organ",              //19
        "Church Organ",            //20
        "Reed Organ",              //21
        "Accordion",               //22
        "Harmonica",               //23
        "Tango Accordion",         //24
        "Acoustic Guitar (nylon)", //25
        "Acoustic Guitar (steel)", //26
        "Electric Guitar (jazz)",  //27
        "Electric Guitar (clean)", //28
        "Electric Guitar (muted)", //29
        "Overdriven Guitar",       //30
        "Distortion Guitar",       //31
        "Guitar Harmonics",        //32
        "Acoustic Bass",           //33
        "Electric Bass (finger)",  //34
        "Electric Bass (pick)",    //35
        "Fretless Bass",           //36
        "Slap Bass 1",             //37
        "Slap Bass 2",             //38
        "Synth Bass 1",            //39
        "Synth Bass 2",            //40
        "Violin",                  //41
        "Viola",                   //42
        "Cello",                   //43
        "Contrabass",              //44
        "Tremolo Strings",         //45
        "Pizzicato Strings",       //46
        "Orchestral Harp",         //47
        "Timpani",                 //48
        "String Ensemble 1",       //49
        "String Ensemble 2",       //50
        "Synth Strings 1",         //51
        "Synth Strings 2",         //52
        "Choir Aahs",              //53
        "Voice Oohs",              //54
        "Synth Choir",             //55
        "Orchestra Hit",           //56
        "Trumpet",                 //57
        "Trombone",                //58
        "Tuba",                    //59
        "Muted Trumpet",           //60
        "French Horn",             //61
        "Brass Section",           //62
        "Synth Brass 1",           //63
        "Synth Brass 2",           //64
        "Soprano Sax",             //65
        "Alto Sax",                //66
        "Tenor Sax",               //67
        "Baritone Sax",            //68
        "Oboe",                    //69
        "English Horn",            //70
        "Bassoon",                 //71
        "Clarinet",                //71
        "Piccolo",                 //72
        "Flute",                   //73
        "Recorder",                //74
        "Pan Flute",               //75 <-- max, For instruments higher, we treat them as guitar if we encode instruments.
        "Blown Bottle",            //76
        "Shakuhachi",              //77
        "Whistle",                 //78
        "Ocarina",                 //79
        "Lead 1 (square)",         //80
        "Lead 2 (sawtooth)",       //81
        "Lead 3 (calliope)",       //82
        "Lead 4 (chiff)",          //83
        "Lead 5 (charang)",        //84
        "Lead 6 (voice)",          //85
        "Lead 7 (fifths)",         //86
        "Lead 8 (bass + lead)",    //87
        "Pad 1 (new age)",         //88
        "Pad 2 (warm)",            //89
        "Pad 3 (polysynth)",       //90
        "Pad 4 (choir)",           //91
        "Pad 5 (bowed)",           //92
        "Pad 6 (metallic)",        //93
        "Pad 7 (halo)",            //94
        "Pad 8 (sweep)",           //95
        "FX 1 (rain)",             //96
        "FX 2 (soundtrack)",       //97
        "FX 3 (crystal)",          //98
        "FX 4 (atmosphere)",       //99
        "FX 5 (brightness)",       //100
        "FX 6 (goblins)",          //101
        "FX 7 (echoes)",           //102
        "FX 8 (sci-fi)",           //103
        "Sitar",                   //104
        "Banjo",                   //105
        "Shamisen",                //106
        "Koto",                    //107
        "Kalimba",                 //108
        "Bagpipe",                 //109
        "Fiddle",                  //110
        "Shanai",                  //111
        "Tinkle Bell",             //112
        "Agogo",                   //113
        "Steel Drums",             //114
        "Woodblock",               //115
        "Taiko Drum",              //116
        "Melodic Tom",             //117
        "Synth Drum",              //118
        "Reverse Cymbal",          //119
        "Guitar Fret Noise",       //120
        "Breath Noise",            //121
        "Seashore",                //122
        "Bird Tweet",              //123
        "Telephone Ring",          //124
        "Helicopter",              //125
        "Applause",                //126
        "Gunshot"                  //127
    };
    private static final NumberFormat numberFormat = NumberFormat.getInstance();
    //----------------------------------
    public static int getInstrument(String name) {
        Integer instrument = instrumentsByName.get(name);
        if (instrument == null) {
            System.err.println("WARNING: no instrument for name " + name);
            return 0;
        }
        return instrument.intValue();
    }

    static {
    	for(int instrumentNumber = 0; instrumentNumber< programs.length; instrumentNumber++) {
    		instrumentsByName.put(programs[instrumentNumber], instrumentNumber);
    	}
        numberFormat.setMaximumFractionDigits(1);
    }


	private static boolean isLinux() {
		return System.getProperty("os.name").contains("Linux");
	}
	/**
	 *
	 * @param absoluteDiskPathForWav
	 * @param absoluteDiskPathForMp3
	 * @return true if successful
	 */
	public static boolean convertWavToMp3(String absoluteDiskPathForWav, String absoluteDiskPathForMp3, StringBuilder errorMessage)  {
		String commandPrefix = isLinux()? "/home/donalds/lame-3.99.5/frontend/lame -V 1": "ffmpeg -i";
		String command = commandPrefix + " " + absoluteDiskPathForWav + " \"" + absoluteDiskPathForMp3 + "\"";
		try {
			System.err.println(command);
			final Process process=Runtime.getRuntime().exec(command);
			final InputStream errorInputStream=process.getErrorStream();
			int rc=process.waitFor();
			if (rc==0) {
				if (!new File(absoluteDiskPathForWav).delete()) {
					errorMessage.append("WARNING: unable to delete " + absoluteDiskPathForWav);
				}
				return true;
			}
			if (errorInputStream!=null) {
				BufferedReader reader = new BufferedReader(new InputStreamReader(errorInputStream));
				while (true) {
					String line=reader.readLine();
					if (line==null) {
						break;
					}
					errorMessage.append("\nError from process:  " + line);
				}
				reader.close();
			}
			errorMessage.append("Unable to convert " + absoluteDiskPathForWav + " to " + absoluteDiskPathForMp3 + ",\n rc = " + rc);
		} catch (Exception exc) {
			errorMessage.append("Unable to convert " + absoluteDiskPathForWav + " to " + absoluteDiskPathForMp3 + " due to " + exc.getMessage());
		}
		return false;
	}
 	private static void sleep(long mls) {
        try {
            Thread.sleep(mls);
        } catch (InterruptedException exc) {
            System.err.println("Interrupted");
            Thread.interrupted();
        }
    }

    public static void playMidiFile(String path, double tempoFactor) throws MidiUnavailableException, InvalidMidiDataException, IOException {
    	Sequence sequence=MidiSystem.getSequence(new File(path));
    	playSequence(sequence,tempoFactor);
    }

    public static void playSequence(Sequence sequence, double tempoFactor) throws MidiUnavailableException, InvalidMidiDataException {
    	playSequence(sequence,tempoFactor, 10000);
    }
    public static void playSequence(Sequence sequence, double tempoFactor, int maxSeconds) throws MidiUnavailableException, InvalidMidiDataException {
    	loadSoundBank();
    	Sequencer sequencer = MidiSystem.getSequencer();
    	sequencer.setSequence(sequence);
    	sequencer.setTickPosition(0);
    	sequencer.open();
    	sequencer.setTempoFactor((float) tempoFactor);
    	sequencer.start();
    	long startTime= System.currentTimeMillis();
    	long endTime=startTime + maxSeconds*1000L;
    	long previousSeconds = 0;
    	while (true) {
    		sleep(1000);
    		long now=System.currentTimeMillis();
    		long milliseconds = now-startTime;
    		long seconds = milliseconds/1000;
    		if (seconds!=previousSeconds) {
    			System.out.print(seconds + " ");
    			previousSeconds=seconds;
    			if (seconds==30) {
    				System.out.println();
    			}
    		}
    		if (sequencer.getMicrosecondPosition() >= sequence.getMicrosecondLength() || now>=endTime) {
    			System.out.println("\nBreaking after " + seconds + " seconds");
    			sequencer.stop();
    			break;
    		}
    	}
    }

    public static Sequence makeSequence(List<Note> ns, int instrumentNumber) throws InvalidMidiDataException {
    	  Sequence sequence = new Sequence(Sequence.PPQ, 120 /*resolution*/);
    	  Track track = sequence.createTrack();
    	  int channel = ns.get(0).getChannel();
    	  MidiMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE,channel,instrumentNumber,0);
    	  track.add(new MidiEvent(midiMessage, 0));
    	  for(Note note:ns) {
    		  //System.out.println(note);
    		note.addMidiEvents(track);
    	  }
    	  return sequence;
	}

	//---------------------------------
    private static boolean soundBackLoaded=false;
    protected static void loadSoundBank() {// Download for higher quality MIDI
    	if (soundBackLoaded) {
    		return;
    	}
    	soundBackLoaded=true;
    	long startTime=System.currentTimeMillis();
    	final String filename = "GeneralUser_GS_SoftSynth.sf2";  // FreeFont.sf2   Airfont_340.dls
    	final String soundBankLocation = PlayTwoPartHarmonies.ROOT_DIR_PATH+ File.separator + filename;
    	File file= new File(soundBankLocation);
        try {
            if (!file.exists()) {
                System.out.println("Downloading soundbank (first time only!). This may take a while.");
                PlayMelodyStrings.copyURLContentsToFile(new URL("http://truthsite.org/music/" + filename), file);
                System.out.println("Soundbank downloaded to " + file.getAbsolutePath());
            }
            Synthesizer synth = MidiSystem.getSynthesizer();
            Soundbank deluxeSoundbank;
            deluxeSoundbank = MidiSystem.getSoundbank(file);
            synth.loadAllInstruments(deluxeSoundbank);
            float seconds = 0.001f*(System.currentTimeMillis()-startTime);
            System.out.println("Loaded soundbank from " + file.getAbsolutePath() + " in " + seconds + " seconds");
        } catch (Exception exc) {
            System.err.println("Unable to load soundbank " + file.getAbsolutePath() + " due to " + exc.getMessage()
                + ", using default soundbank.");
        }
    }

  	 public static void chooseMidiFileAndPlay() throws MidiUnavailableException, InvalidMidiDataException, IOException {
	    	JFileChooser chooser = new JFileChooser(new File("D:/Music/MIDI/clean_midi"));
	    	chooser.setFileFilter(new FileFilter(){
				@Override
				public boolean accept(File file) {
					String name=file.getName().toLowerCase();
					return file.isDirectory() || name.endsWith(".mid") || name.endsWith("midi");
				}

				@Override
				public String getDescription() {
					return "Midi files";
				}});
	    	if (chooser.showDialog(null, "Play")!=JFileChooser.APPROVE_OPTION) {
	    		return;
	    	}

	    	File chosenFile = chooser.getSelectedFile();
	    	if (chosenFile!=null) {
	    		System.out.println("Playing " + chosenFile.getAbsolutePath());
	    		playMidiFile(chosenFile.getAbsolutePath(), 1.0);
	    	}
	    }
	  //-----------------------------------
    public static void main(String[] args) {
       // args = new String[] {  "d:/tmp/beatles-melodies-input.txt"};
        try {
        	chooseMidiFileAndPlay();
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }
}

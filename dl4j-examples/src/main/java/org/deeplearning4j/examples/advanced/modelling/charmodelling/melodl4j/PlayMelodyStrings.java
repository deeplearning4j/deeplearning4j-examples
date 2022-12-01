/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.charmodelling.melodl4j;

import javax.sound.midi.*;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;
import java.io.BufferedReader;
import java.io.BufferedInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/*
 *  This plays melody strings using your operating system's MIDI synthesizer.
 *  There's a public static method for playing melody strings in files and
 *  another method for playing melody strings passed in as java.lang.Strings.
 *
 *  If you run main, it will open a JFileChooser pointing to the midi-learning subdirectory of your home directory,
 *  prompting you to choose a txt file (typically starting with prefix "composition-").
 *
 *  The format for the melody strings is determined by MelodyStrings.java and MidiMelodyExtractor.java.
 *
 *  In a valid melody string, each pitch or rest character should be followed by
 *  a duration. But during learning, some melody strings have invalid syntax.
 *  This class will ignore invalid characters in the melody strings.
 *
 * @author Donald A. Smith
 */
public class PlayMelodyStrings {
    private static Random random = new Random();
    private final static String tmpDir = System.getProperty("java.io.tmpdir");
    private static Map<String, Integer> instrumentsByName = new HashMap<>();

    private static Pattern INSTRUMENT_PATTERN = Pattern.compile(".*Instrument = (\\d+).*");
    private static Pattern START_NOTE_PATTERN = Pattern.compile(".*StartNote = (\\d+).*");
    //http://en.wikipedia.org/wiki/General_MIDI
    public static final String[] programs = {
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Electric Grand Piano",
        "Honky-tonk Piano",
        "Electric Piano 1",
        "Electric Piano 2",
        "Harpsichord",
        "Clavinet",
        "Celesta",
        "Glockenspiel",
        "Music Box",
        "Vibraphone",
        "Marimba",
        "Xylophone",
        "Tubular Bells",
        "Dulcimer",
        "Drawbar Organ",
        "Percussive Organ",
        "Rock Organ",
        "Church Organ",
        "Reed Organ",
        "Accordion",
        "Harmonica",
        "Tango Accordion",
        "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)",
        "Electric Guitar (clean)",
        "Electric Guitar (muted)",
        "Overdriven Guitar",
        "Distortion Guitar",
        "Guitar Harmonics",
        "Acoustic Bass",
        "Electric Bass (finger)",
        "Electric Bass (pick)",
        "Fretless Bass",
        "Slap Bass 1",
        "Slap Bass 2",
        "Synth Bass 1",
        "Synth Bass 2",
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Tremolo Strings",
        "Pizzicato Strings",
        "Orchestral Harp",
        "Timpani",
        "String Ensemble 1",
        "String Ensemble 2",
        "Synth Strings 1",
        "Synth Strings 2",
        "Choir Aahs",
        "Voice Oohs",
        "Synth Choir",
        "Orchestra Hit",
        "Trumpet",
        "Trombone",
        "Tuba",
        "Muted Trumpet",
        "French Horn",
        "Brass Section",
        "Synth Brass 1",
        "Synth Brass 2",
        "Soprano Sax",
        "Alto Sax",
        "Tenor Sax",
        "Baritone Sax",
        "Oboe",
        "English Horn",
        "Bassoon",
        "Clarinet",
        "Piccolo",
        "Flute",
        "Recorder",
        "Pan Flute",
        "Blown Bottle",
        "Shakuhachi",
        "Whistle",
        "Ocarina",
        "Lead 1 (square)",
        "Lead 2 (sawtooth)",
        "Lead 3 (calliope)",
        "Lead 4 (chiff)",
        "Lead 5 (charang)",
        "Lead 6 (voice)",
        "Lead 7 (fifths)",
        "Lead 8 (bass + lead)",
        "Pad 1 (new age)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth)",
        "Pad 4 (choir)",
        "Pad 5 (bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
        "FX 1 (rain)",
        "FX 2 (soundtrack)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes)",
        "FX 8 (sci-fi)",
        "Sitar",
        "Banjo",
        "Shamisen",
        "Koto",
        "Kalimba",
        "Bagpipe",
        "Fiddle",
        "Shanai",
        "Tinkle Bell",
        "Agogo",
        "Steel Drums",
        "Woodblock",
        "Taiko Drum",
        "Melodic Tom",
        "Synth Drum",
        "Reverse Cymbal",
        "Guitar Fret Noise",
        "Breath Noise",
        "Seashore",
        "Bird Tweet",
        "Telephone Ring",
        "Helicopter",
        "Applause",
        "Gunshot"
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
        for (int instrumentNumber = 0; instrumentNumber < programs.length; instrumentNumber++) {
            instrumentsByName.put(programs[instrumentNumber], instrumentNumber);
        }
        numberFormat.setMaximumFractionDigits(1);
    }

    private static List<String> readStringsFromFile(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        List<String> result = new ArrayList<>();
        while (true) {
            String line=reader.readLine();
            if (line==null) {
                break;
            }
            line=line.trim();
            if (line.length()<20 || line.startsWith("---") || line.startsWith("#")) {
                continue;
            }
            result.add(line);
        }
        reader.close();
        return result;
    }

    //-----------------------------------------------
    public static void playMelodies(String inFilepath, double secondsToPlay) throws IOException, MidiUnavailableException, InvalidMidiDataException {
        playMelodies(inFilepath, 0, secondsToPlay);
    }

    /* Plays midi file through the system's midi sequencer, starting from the last line in the file.
     *  @param inFilepath points to a local file containing symbolic melodies
     *  @param instrumentName : See the list in Midi2MelodyStrings.java. If null, defaults to "Acoustic Grand Piano".
     *  @param secondsToPlay -- seconds after which to stop the music playback.
     */
    public static void playMelodies(String inFilepath, int instrumentNumber, double secondsToPlay) throws IOException, MidiUnavailableException, InvalidMidiDataException {
        System.out.println("Playing melodies from " + inFilepath);
        List<String>  lines = readStringsFromFile(inFilepath);
        int startNote = 45;
        for(int lineNumber=lines.size()-1;lineNumber>=0;lineNumber--) {
            String line = lines.get(lineNumber);
            line = line.trim();
            if (line.equals("")) {
                continue;
            }
            if (line.startsWith(MelodyStrings.COMMENT_STRING)) {
                System.out.println(line);
                instrumentNumber = getInstrumentNumberFromLine(line, instrumentNumber);
                startNote = getStartNoteFromLine(line, startNote);
                System.out.println("Using instrument " + programs[instrumentNumber] + " and startNote " + startNote);
                continue;
            }
            System.out.println("\nPlaying " + lineNumber + " : " + line);
            playMelody(line, startNote, instrumentNumber, secondsToPlay);
            sleep(2000); // so there's a noticeable gap between melodies
        }
        System.exit(0);
    }

    private static int getInstrumentNumberFromLine(String line, int defaultInstrumentNumber) {
        Matcher matcher = INSTRUMENT_PATTERN.matcher(line);
        if (matcher.find()) {
            String instrumentNumberString = matcher.group(1);
            return Integer.parseInt(instrumentNumberString);
        }
        return defaultInstrumentNumber;
    }

    private static int getStartNoteFromLine(String line, int theDefault) {
        Matcher matcher = START_NOTE_PATTERN.matcher(line);
        if (matcher.find()) {
            String instrumentNumberString = matcher.group(1);
            return Integer.parseInt(instrumentNumberString);
        }
        return theDefault;
    }

    private static void sleep(long mls) {
        try {
            Thread.sleep(mls);
        } catch (InterruptedException exc) {
            System.err.println("Interrupted");
            Thread.interrupted();
        }
    }

    private static int chooseRandomInstrumentPop() {
        switch (random.nextInt(3)) {
            case 0:
                return getInstrument("Acoustic Guitar (steel)");
            case 1:
                return getInstrument("Acoustic Guitar (nylon)");
            default:
                return 0;
        }
    }

    private static int chooseRandomInstrumentClassical() {
        switch (random.nextInt(10)) {
            case 0:
                return getInstrument("Violin");
            case 1:
                return getInstrument("Cello");
            case 2:
                return getInstrument("Church Organ");
            case 3:
                return getInstrument("Flute");
            case 4:
                return getInstrument("Choir Aahs");
            case 5:
                return getInstrument("String Ensemble 1");
            case 6:
                return getInstrument("Acoustic Guitar (nylon)");
            case 7:
                return getInstrument("Trumpet");
            default:
                return 0;
        }
    }

    /*
     * If the melody string contains a tab, the instrument used and start pitch will be obtained from the header before the tab.
     * Otherwise, it will use startPitch 55 and Acoustic Grand Piano (instrument number 0)
     */
    public static void playMelody(final String melody, final double secondsToPlay) throws Exception {
        int startPitch = 55;
        int instrumentNumber = 0; // Acoustic Grand Piano
        playMelody(melody, startPitch, instrumentNumber, secondsToPlay);
    }
    public static void playMelodies(final List<String>melodies, final double secondsToPlay) throws Exception {
        int startPitch = 55;
        int instrumentNumber = 0; // Acoustic Grand Piano
        for(String melody:melodies) {
            playMelody(melody, startPitch, instrumentNumber, secondsToPlay);
            try {Thread.sleep(1000);} catch (InterruptedException exc) {Thread.interrupted();}
        }
    }

    public static List<Note> createNoteSequence(String melody, int instrumentNumber, int startNote) {
        int lastRawNote = startNote;
        // First char is noteDuration
        // Next: are  restDuration, pitch, noteDuration
        int channel = 0;
        int velocity = 95;
        int track = 2;
        final int resolution = 480;
        final int resolutionDelta = resolution / 16;
        int index = 0; //getIndexOfFirstPitchDuration(line);
        List<Note> ns = new ArrayList<>();
        long tick = 0;
        if (MelodyStrings.isDurationChar(melody.charAt(index))) {
            //  Note(int pitch, long startTick, int instrument, int channel, int velocity)
            Note note = new Note(startNote, tick, instrumentNumber, channel, velocity);
            long duration = MelodyStrings.getDurationInTicks(melody.charAt(index), resolutionDelta);
            note.setEndTick(tick + duration);
            ns.add(note);
            index++;
            tick += duration;
        }
        int noteDurationInTicks = 0;

        while (index < melody.length() - 1) {
            char ch = melody.charAt(index);
            if (ch == 'R') {
                index++;
                ch = melody.charAt(index);
                if (MelodyStrings.isDurationChar(ch)) {
                    tick += MelodyStrings.getDurationInTicks(ch, resolutionDelta);
                    index++;
                } else {
                    System.out.print('R'); // Badly formed melody string
                }
            } else if (MelodyStrings.isPitchChar(ch)) {
                index++;
                int pitchDelta = MelodyStrings.getPitchDelta(ch);
                lastRawNote += pitchDelta;
                while (lastRawNote < 30) {
                    System.out.print('<');
                    lastRawNote += 12; // This is a hack to prevent melodies from becoming inaudible
                }
                while (lastRawNote >= 80) {
                    System.out.print('>');
                    lastRawNote -= 12; // This is a hack to prevent melodies from becoming inaudible
                }
                ch = melody.charAt(index);
                if (MelodyStrings.isDurationChar(ch)) {
                    noteDurationInTicks = MelodyStrings.getDurationInTicks(ch, resolutionDelta);
                    index++;
                } else {
                    System.out.print('D'); // Badly formed melody string
                    noteDurationInTicks = 4 * resolutionDelta;
                }
                //                 Note(int pitch, long startTick, int instrument, int channel, int velocity)
                Note note = new Note(lastRawNote, tick, instrumentNumber, channel, velocity);
                note.setEndTick(tick + noteDurationInTicks);
                ns.add(note);
                tick += noteDurationInTicks;
            } else {
                System.out.print(ch);
                index++;
            }
        }
        return ns;
    }

    // This method will try to play a melody even if the string is malformed.  The neural networks sometimes output invalid substrings, especially at the beginning of learning.
    public static void playMelody(String melody, int startNote, int instrumentNumber, double secondsToPlay) throws MidiUnavailableException, InvalidMidiDataException {
        loadSoundBank();
        List<Note> ns = createNoteSequence(melody, instrumentNumber, startNote);
        Sequencer sequencer = MidiSystem.getSequencer();
        Sequence sequence = makeSequence(ns, instrumentNumber);
        sequencer.setSequence(sequence);
        long tickLength = sequencer.getTickLength();
        //JOptionPane.showMessageDialog(null, "Click enter to abort.");
        long startTimeInMilliseconds = System.currentTimeMillis();
        sequencer.setTickPosition(0);
        sequencer.open();
        sequencer.setTempoFactor(3.5f);
        sequencer.start();
        System.out.println();
        while (sequencer.getTickPosition() < tickLength) {
            sleep(1000);
            long nowInMilliseconds = System.currentTimeMillis();
            long secondsPlayed = (nowInMilliseconds - startTimeInMilliseconds)/1000;
            System.out.print(secondsPlayed + " ");
            if (secondsPlayed >= secondsToPlay) {
                sequencer.stop();
                break;
            }
        } // while
        System.out.println();
    }

    public static void playSequence(Sequence sequence, double tempoFactor) throws MidiUnavailableException, InvalidMidiDataException {
        loadSoundBank();
        Sequencer sequencer = MidiSystem.getSequencer();
        sequencer.setSequence(sequence);
        sequencer.setTickPosition(0);
        sequencer.open();
        sequencer.setTempoFactor((float) tempoFactor);
        sequencer.start();
    }

    private static Sequence makeSequence(List<Note> ns, int instrumentNumber) throws InvalidMidiDataException {
        Sequence sequence = new Sequence(Sequence.PPQ, 120 /*resolution*/);
        Track track = sequence.createTrack();
        int channel = ns.get(0).getChannel();
        MidiMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE, channel, instrumentNumber, 0);
        track.add(new MidiEvent(midiMessage, 0));
        for (Note note : ns) {
            //System.out.println(note);
            note.addMidiEvents(track);
        }
        return sequence;
    }

    //---------------------------------
    private static boolean soundBackLoaded = false;

    private static void loadSoundBank() {// Download for higher quality MIDI
        if (soundBackLoaded) {
            return;
        }
        soundBackLoaded = true;
        final String filename = "GeneralUser_GS_SoftSynth.sf2";  // FreeFont.sf2   Airfont_340.dls
        final String soundBankLocation = tmpDir + "/" + filename;
        File file = new File(soundBankLocation);
        try {
            if (!file.exists()) {
                System.out.println("Downloading soundbank (first time only!). This may take a while.");
                copyURLContentsToFile(new URL("http://truthsite.org/music/" + filename), file);
                System.out.println("Soundbank downloaded to " + file.getAbsolutePath());
            }
            Synthesizer synth = MidiSystem.getSynthesizer();
            Soundbank deluxeSoundbank;
            deluxeSoundbank = MidiSystem.getSoundbank(file);
            synth.loadAllInstruments(deluxeSoundbank);
            System.out.println("Loaded soundbank from " + soundBankLocation);
        } catch (Exception exc) {
            System.err.println("Unable to load soundbank " + soundBankLocation + " due to " + exc.getMessage()
                + ", using default soundbank.");
        }
    }

    private static String getPathToExampleMelodiesFile() throws Exception {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(MelodyModelingExample.homeMidiDir));
        fileChooser.setFileFilter(new FileFilter(){
            @Override
            public boolean accept(File f) {
                final String name = f.getName().toLowerCase();
                return name.endsWith("txt");
            }

            @Override
            public String getDescription() {
                return "txt files containing MIDI melody strings";
            }
        });
        if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
            return fileChooser.getSelectedFile().getAbsolutePath();
        } else {
            System.exit(1);
            return null;
        }
    }
    public static void copyURLContentsToFile(URL url, File file) throws IOException {
        final int blockSize = 256;
        BufferedInputStream bis = new BufferedInputStream(url.openStream(), blockSize);
        FileOutputStream fos = new FileOutputStream(file);
        long totalRead = 0;
        byte bytes[] = new byte[blockSize];
        while (true) {
            int read = bis.read(bytes);
            if (read < 0) {
                break;
            }
            totalRead += read;
            fos.write(bytes, 0, read);
        }
        bis.close();
        fos.close();
        System.out.println("Read " + numberFormat.format(totalRead) + " bytes from " + url + " into " + file.getAbsolutePath());
    }

    public static void main1(String[] args) {
        try {
            getPathToExampleMelodiesFile();
        } catch (Throwable thr) {
            thr.printStackTrace();
        }
    }

    //-----------------------------------
    public static void main(String[] args) {
//        String filename="composition-bach-midi.zip-1669841367457.txt";
//        MelodyModelingExample.makeSureFileIsInTmpDir(filename);
//        args = new String[] {MelodyModelingExample.tmpDir + "/" + filename};
        try {
            String pathToMelodiesFile = args.length == 0 ? getPathToExampleMelodiesFile() : args[0];
            playMelodies(pathToMelodiesFile, 30); /// Note: by default it plays 30 seconds of each melody
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }
}

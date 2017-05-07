package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MetaMessage;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;


/*
 *  Parses MIDI files and outputs music in symbolic format, for input to MelodyModelingExample.
 *
 *  If a MIDI track has polyphony, this class will extract two melodies: one for the upper
 *  part of the harmony, and another for the lower part of the harmony.
 *
 *  If passed a directory it extracts melodies from all files appearing in the directory.
 *
 *  It skips tracks that have too much silence or too little variety of pitches.
 *
 *  @author Donald A. Smith (ThinkerFeeler@gmail.com)
 */
public class Midi2MelodyStrings {
    private static boolean trace = false;
    // The following strings are used to build the symbolic representation of a melody
    // The next two strings contain chars used to indicate pitch deltas.
    public static final String noteGapCharsPositive = "0123456789abc"; // A pitch delta of "0" indicates delta=0.
    public static final String noteGapCharsNegative = "!@#$%^&*(ABCD"; // A pitch delta of "!" indicates delta=-1.
    // R is used to indicate the beginning of a rest
    public static int durationDeltaParts = 8;
    public static final String durationChars = "defghijklmnopqrstuvwzyzEFGHIJKLM"; // 32 divisions.
                                             // 012345678
    // 'd' indicates the smallest pitch duration allowed (typically a 1/32 note or so).
    // 'e' is a duration twice that of 'd'
    // 'f' is a duration three times that of 'd', etc.

    private static final int MINIMUM_NUMBER_OF_DISTINCT_PITCHES = 4;
    private static final double MAXIMUM_PROPORTION_OF_REPEATED_NOTES = 0.7;
    private static final double MAXIMUM_ALLOWED_PROPORTION_OF_SILENCE = 0.66;
    private static final int MAXIMUM_ALLOWED_NUMBER_OF_REPEATED_NOTES = 10;
    //                                          12345678901234567890123456789012

    private Piece piece;
    private HashMap<String, Integer> trackAndChannelToInstrumentIndexMap = new HashMap<String, Integer>();
    private static Map<String, Integer> instrumentsByName = new HashMap<>();
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

    static {
        for (int i = 0; i < programs.length; i++) {
            instrumentsByName.put(programs[i], new Integer(i));
        }
    }

    public static final int NOTE_ON = 0x90;
    public static final int NOTE_OFF = 0x80;
    private static final double MILLION = 1000000.0;

    //-------------------------------------------------------------------
    public static void main(String[] args) { // You can download midi files from http://truthsite.org/music/bach-midi.zip and http://www.musedata.org
        try {
            showSequencesForAllMidiFiles("d:/music/MIDI/classical/haydn", "d:/tmp/haydn-midi-melodies.txt", 10, false);
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
    }

    //----------------------------------
    public static int getInstrument(String name) {
        Integer instrument = instrumentsByName.get(name);
        if (instrument == null) {
            System.err.println("WARNING: no instrument for name " + name);
            return 0;
        }
        return instrument.intValue();
    }

    //-------------------------------------------------------------------
    private static void myShow(Sequence sequence) {
        System.out.println("Length in seconds = " + sequence.getMicrosecondLength() / MILLION);
        System.out.println("Tick length in seconds= " + sequence.getTickLength());
        Track[] tracks = sequence.getTracks();
        for (int i = 0; i < tracks.length; i++) {
            Track track = tracks[i];
            System.out.print("Track " + i + ": " + track.size() + " events:  ");
            for (int j = 0; j < track.size(); j++) {
                System.out.print(track.get(j) + " ");
            }
            System.out.println();
            //if (i==2) {sequence.deleteTrack(track);}
        }
        System.out.println(tracks.length + " tracks");
    }

    //-------------------------------------------------------------------
    private static int findDurationInTicksOfNote(int i, Note note1, Track track, int trackNumber, ShortMessage message) {
        for (int j = i + 1; j < track.size(); j++) {
            MidiEvent event2 = track.get(j);
            MidiMessage message2 = event2.getMessage();
            if (message2 instanceof ShortMessage) {
                ShortMessage shortMessage2 = (ShortMessage) message2;
                int command = shortMessage2.getCommand();
                if (command != NOTE_OFF && command != NOTE_ON && command != ShortMessage.SYSTEM_RESET) {
                    continue;
                }
                int channel2 = shortMessage2.getChannel();
                if (channel2 != note1.getChannel()) {
                    continue;
                }
                int key2 = shortMessage2.getData1();
                int octave2 = (key2 / 12) - 1;
                if (octave2 != note1.getOctave()) {
                    continue;
                }
                int noteKey2 = key2 % 12;
                if (noteKey2 != note1.getKey()) {
                    continue;
                }
// In some cases, apparently, for a given piano key (say, C6), there are multiple NOTE_ONs with different velocities, all ended by a single NOTE_OFF.
// But we can end the first note when the subsequent one begins.
//                if (command==NOTE_ON && shortMessage2.getData2()!=0) {
//                	continue;
//                	//System.err.println("WARNING: ending NOTE_ON with velocity " + shortMessage2.getData2() + ", start velocity = " + note1.velocity);
//                }
                return (int) (event2.getTick() - note1.startTick);
            }
        }
        System.err.println("Warning: unended note in track: " + trackNumber + ": " + note1 + ", message = " + show(message));
        return 0;
    }

    private static String decodeCommand(int cmd) {
        switch (cmd) {
            case ShortMessage.ACTIVE_SENSING:
                return "ACTIVE_SENSING";
            case ShortMessage.CHANNEL_PRESSURE:
                return "CHANNEL_PRESSURE";
            case ShortMessage.CONTINUE:
                return "CONTINUE";
            case ShortMessage.CONTROL_CHANGE:
                return "CONTROL_CHANGE";
            case ShortMessage.END_OF_EXCLUSIVE:
                return "END_OF_EXCLUSIVE";
            case ShortMessage.MIDI_TIME_CODE:
                return "MIDI_TIME_CODE";
            case ShortMessage.NOTE_OFF:
                return "NOTE_OFF";
            case ShortMessage.NOTE_ON:
                return "NOTE_ON";
            case ShortMessage.PITCH_BEND:
                return "PITCH_BEND";
            case ShortMessage.POLY_PRESSURE:
                return "POLY_PRESSURE";
            case ShortMessage.PROGRAM_CHANGE:
                return "PROGRAM_CHANGE";
            case ShortMessage.SONG_POSITION_POINTER:
                return "SONG_POSITION_POINTER";
            case ShortMessage.SONG_SELECT:
                return "SONG_SELECT";
            case ShortMessage.START:
                return "START";
            case ShortMessage.STOP:
                return "STOP";
            case ShortMessage.SYSTEM_RESET:
                return "SYSTEM_RESET";
            case ShortMessage.TIMING_CLOCK:
                return "TIMING_CLOCK";
            case ShortMessage.TUNE_REQUEST:
                return "TUNE_REQUEST";
            default:
                return "(unknown)";
        }
    }

    public static String show(ShortMessage msg) {
        return "ShortMessage: command = " + decodeCommand(msg.getCommand()) // + "(" + msg.getCommand() + "), "
            + ", data1 = " + msg.getData1() + ", data2= " + msg.getData2();
    }

    //----------------
    public static boolean isDurationChar(char ch) {
        return durationChars.indexOf(ch)>=0;
    }
    public static boolean isPitchDeltaChar(char ch) {
        return noteGapCharsNegative.indexOf(ch)>=0 || noteGapCharsPositive.indexOf(ch)>=0;
    }
    public static boolean isRestChar(char ch) {
        return ch=='R';
    }
    public static int getPitchDeltaFromMelodyChar(char ch) {
        int delta = noteGapCharsPositive.indexOf(ch);
        if (delta>=0) {
            return delta;
        }
        delta = noteGapCharsNegative.indexOf(ch);
        if (delta>=0) {
            return -(delta+1);
        }
        System.err.println("WARNING: Bad pitch delta char (" + ch + "), using default of 0");
        return 0; // default: bad data
    }
    //-------------------
    private static long pow(int base, int n) {
        long result = 1;
        for (int i = 0; i < n; i++) {
            result *= base;
        }
        return result;
    }

    //---------------------------------------------
    //From http://stackoverflow.com/questions/3850688/reading-midi-files-in-java
    // returns ticklength
    private void loadPiece(File file) throws InvalidMidiDataException, IOException {
        Sequence sequence = MidiSystem.getSequence(file);
        if (trace) {
            System.out.println("Resolution = " + sequence.getResolution() + ", divisionType = " + sequence.getDivisionType());
        }
        piece = new Piece(sequence.getTickLength(), sequence.getResolution());
        int trackNumber = 0;
        for (Track track : sequence.getTracks()) {
            trackNumber++;
            for (int noteIndex = 0; noteIndex < track.size(); noteIndex++) {
                MidiEvent event = track.get(noteIndex);
                long tick = event.getTick();
                MidiMessage message = event.getMessage();
                if (message instanceof ShortMessage) {
                    ShortMessage shortMessage = (ShortMessage) message;
                    int channel = shortMessage.getChannel();
                    if (shortMessage.getCommand() == ShortMessage.NOTE_ON) {
                        int midiNoteValue = shortMessage.getData1();
                        int velocity = shortMessage.getData2();
                        if (velocity == 0) {
                            continue;
                        }
                        //System.out.println("Note on, " + noteName + octave + " key=" + key + " velocity: " + velocity);
                        Note note = new Note(tick, midiNoteValue, velocity, channel);
                        note.startTick = tick;
                        long duration = findDurationInTicksOfNote(noteIndex, note, track, trackNumber, shortMessage);
                        if (duration > 0) {
                            note.setDuration(duration);
                            piece.addNote(trackNumber, channel, note);
                        }
                    } else if (shortMessage.getCommand() == NOTE_OFF) {
                        // We can ignore these because of the method call to findDurationInTicksOfNote
                    } else if (shortMessage.getCommand() == ShortMessage.PROGRAM_CHANGE) {
                        int programNumber = shortMessage.getData1();
                        String hashKey = trackNumber + "-" + channel;
                        NoteSequence ns = piece.findTrack(trackNumber, channel);
                        if (ns == null) {
                            if (trace) {
                                System.out.println("Creating noteSequence from PROGRAM_CHANGE for track "
                                    + trackNumber + ", channel " + channel + ", instrument = " + programs[programNumber]);
                            }
                            ns = new NoteSequence(tick, trackNumber, channel, sequence.getResolution());
                            piece.noteSequences.add(ns);
                        }
                        ns.addInstrumentChange(programNumber, tick);

                        trackAndChannelToInstrumentIndexMap.put(hashKey, programNumber);
                    } else if (shortMessage.getCommand() == ShortMessage.CONTROL_CHANGE) {
                        // See http://www.midi.org/techspecs/midimessages.php#3
                        //System.out.println("Control change on channel " + channel + ": "+ dataString);
                    } else if (shortMessage.getCommand() == ShortMessage.CHANNEL_PRESSURE) {
                        //System.out.println("Channel pressure " + dataString);
                    } else if (shortMessage.getCommand() == ShortMessage.POLY_PRESSURE) {
                        //System.out.println("Poly pressure " + dataString);
                    } else if (shortMessage.getCommand() == ShortMessage.PITCH_BEND) {
                        //System.out.println("Pitch bend " + dataString);
                    }
                } else if (message instanceof MetaMessage) {
                    // http://cs.fit.edu/~ryan/cse4051/projects/midi/midi.html#settempo
                    // Includes Set Tempo and Time Signature
                    MetaMessage metaMessage = (MetaMessage) message;
                    //System.out.println("Meta message of type " + metaMessage.getType());
                    int type = metaMessage.getType();
                    final byte[] data = metaMessage.getData();
//Data bytes are Most Significant byte first, 8 bits per byte.
// http://sites.uci.edu/camp2014/2014/05/19/timing-in-midi-files/
                    if (type == 0x58) { // Time Signature:   FF 58 04 nn dd cc bb
                        int nn = data[0];
                        int dd = data[1];
                        int cc = data[2]; // clocks per tick
                        //int bb=data[3];
                        String signature = nn + "/" + pow(2, dd);
/*
Four bytes: nn dd cc bb
Time signature of the form: nn/2^dd
eg: 6/8 would be specified using nn=6, dd=3

The parameter cc is the number of MIDI Clocks per metronome tick.

Normally, there are 24 MIDI Clocks per quarter note. However, some software allows this to be set by the user.
The parameter bb defines this in terms of the number of 1/32 notes which make up the usual 24 MIDI Clocks (the 'standard' quarter note).
 */
                        if (trace) {
                            System.out.println("Time Signature with data = " + toString(data) + ", signature = " + signature + ", clocks/tick = " + cc);
                        }
                    } else if (type == 0x51) { // Set Tempo:  FF 51 03 tt tt tt
                        int tempoInMicroSecondsPerQuarterNote = toInt(data, data.length);
                        if (trace) {
                            System.out.println("Tempo = " + tempoInMicroSecondsPerQuarterNote + " microseconds per quarter note");
                        }
/*
This sets the tempo in microseconds per quarter note. This means a change in the unit-length of a delta-time tick. (note 1)

If not specified, the default tempo is 120 beats/minute, which is equivalent to tttttt=500000
 */
                    }
                } else {
                    if (trace) {
                        System.out.println("Other message: " + message.getClass());
                    }
                }
            }
        }
    }

    // Most significant byte first
    public static int toInt(byte[] bytes, int length) {
        int result = 0; //(bytes[0] << 24) | (0xff0000 & (bytes[1] << 16)) | (0xff00 & ((bytes[2] << 8))) | (0xff & bytes[3]);
        for (byte b : bytes) {
            result = result * 256 + (0xff & b);
        }
        return result;
    }

    private static String toString(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++) {
            if (i > 0) {
                sb.append(" ");
            }
            sb.append(bytes[i]);
        }
        return sb.toString();
    }

    //----------
    // 45 is A
    private static Sequence makeScaleStartingAt(int startNote) throws InvalidMidiDataException {
        int resolution = 192;
        float divisionType = 0;
        int channel = 1;
        int durationOfNotes = 1000;
        Sequence sequence = new Sequence(divisionType, resolution);
        Track track = sequence.createTrack();
        int velocity = 100;
        int instrument = instrumentsByName.get("Flute");
        System.out.println("instrument = " + instrument);
        for (int i = 0; i < 3; i++) {
            MidiMessage instrumentChange = new ShortMessage(ShortMessage.PROGRAM_CHANGE, i, instrument, 0);
            track.add(new MidiEvent(instrumentChange, 0));
        }
        int steps[] = {2, 2, 1, 2, 2, 2, 1, 0};
        long count = 0;
        int noteValue = startNote;
        for (int step : steps) {
            MidiMessage midiMessageStart = new ShortMessage(ShortMessage.NOTE_ON, channel, noteValue, velocity);
            MidiMessage midiMessageStop = new ShortMessage(ShortMessage.NOTE_OFF, channel, noteValue, velocity);
            long startTime = count * durationOfNotes;
            MidiEvent startEvent = new MidiEvent(midiMessageStart, startTime);
            track.add(startEvent);
            MidiEvent stopEvent = new MidiEvent(midiMessageStop, startTime + durationOfNotes);
            track.add(stopEvent);
            count++;
            noteValue += step;
        }
        myShow(sequence);
        return sequence;
    }

    //---------------------
    private static int countMidiFilesProcessed = 0;
    private static int countNoteSequencesProcessed = 0;
    private static int countNoteSequencesProcessedWithZeroPolyphony = 0;
    private static double sumShortestLongestDurationRatios = 0.0;
    private static double smallestShortestLongestDurationRatio = Double.MAX_VALUE;
    private static int countShortestLongestDurationRatios = 0;
    private static final double MAXIMUM_ALLOWED_REST_PROPORTION = 0.2;

    // By default we show the top pitch of polyphonic tracks. If bottomPitchOnly is true we show the bottom pitch and skip monophonic tracks
    private static void showSequencesForAllMidiFiles(File file, PrintWriter writer, int minLengthInNotes, boolean bottomPitchOnly) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            for (File child : children) {
                showSequencesForAllMidiFiles(child, writer, minLengthInNotes, bottomPitchOnly);
            }
        } else {
            if (!file.getAbsolutePath().toLowerCase().endsWith("mid")) {
                return;
            }
            Midi2MelodyStrings loader = new Midi2MelodyStrings();
            try {
                //writer.println("\nFor " + file);
                loader.loadPiece(file);
                countMidiFilesProcessed++;
                for (NoteSequence noteSequence : loader.piece.noteSequences) {
                    countNoteSequencesProcessed++;
                    int removed = noteSequence.removeAllButHigherOrLowerNotes(!bottomPitchOnly);
                    if (removed > 0) {
                        System.out.println("Removed " + removed + " notes");
                        if (noteSequence.countOfNotesHavingPolyphony() > 0) {
                            throw new IllegalStateException();
                        }
                    } else {
                        if (bottomPitchOnly) {
                            continue;
                        }
                    }
                    if (noteSequence.getLength() > minLengthInNotes) {
                        long duration = noteSequence.getDuration();
                        double longestRest = noteSequence.getLongestRest();
                        double longestRestProportion = longestRest / duration;
                        double proportionSilence = noteSequence.getProportionSilence();

                        if (noteSequence.isValid() && longestRestProportion <= MAXIMUM_ALLOWED_REST_PROPORTION
                            && noteSequence.getNumberOfDistinctPitches() >= MINIMUM_NUMBER_OF_DISTINCT_PITCHES
                            && (0.0 + noteSequence.getNumberOfRepeatedNotes()) / noteSequence.getNumberOfNotes() < MAXIMUM_PROPORTION_OF_REPEATED_NOTES
                            && noteSequence.getLengthOfLongestSequenceOfRepeatedNotes() <= MAXIMUM_ALLOWED_NUMBER_OF_REPEATED_NOTES
                            && proportionSilence < MAXIMUM_ALLOWED_PROPORTION_OF_SILENCE) {
                            countNoteSequencesProcessedWithZeroPolyphony++;
                            long shortestNote = noteSequence.getShortetNoteDuration();
                            long longestNote = noteSequence.getLongestNoteDuration();
                            double ratio = (1.0 * shortestNote) / longestNote;
                            sumShortestLongestDurationRatios += ratio;
                            countShortestLongestDurationRatios++;
                            if (ratio < smallestShortestLongestDurationRatio) {
                                smallestShortestLongestDurationRatio = ratio;
                            }
                            System.out.println("shortest = " + shortestNote + ", longest = " + longestNote + ", ratio = " + ratio);
                            showTrainingExamplesSymbolic(noteSequence, writer);
                        }
                        //writer.println(noteSequence);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.err.println("While loading " + file + ", got " + e.getMessage());
            }
        }
    }

    public static char getCharForPitchGap(int pitchGap) {
        return pitchGap >= 0 ? noteGapCharsPositive.charAt(pitchGap) : noteGapCharsNegative.charAt(-1 - pitchGap);
    }

    private static char computeDurationChar(long duration, double durationDelta) {
        int times = Math.min((int) Math.round(duration / durationDelta), durationChars.length() - 1);
        return durationChars.charAt(times);
    }

    private static void showTrainingExamplesSymbolic(NoteSequence noteSequence, PrintWriter writer) {
        double averageNoteDuration = noteSequence.getAverageNoteDuration();
        double durationDelta = averageNoteDuration / durationDeltaParts;
        Note lastNote = null;

        for (Note note : noteSequence.getNotes()) {
            if (lastNote == null) {
                long noteDuration = note.getDuration();
                char noteDurationChar = computeDurationChar(noteDuration, durationDelta);
                writer.print(noteDurationChar);
            } else {
                long restDuration = note.getStartTick() - lastNote.getEndTick();
                if (restDuration > 0) {
                    char restDurationChar = computeDurationChar(restDuration, durationDelta);
                    writer.print('R');
                    writer.print(restDurationChar);
                }
                int pitchGap = note.getRawNote() - lastNote.getRawNote();
                while (pitchGap > 12) {
                    pitchGap -= 12;
                }
                while (pitchGap < -12) {
                    pitchGap += 12;
                }
                writer.print(getCharForPitchGap(pitchGap));
                long noteDuration = note.getDuration();
                char noteDurationChar = computeDurationChar(noteDuration, durationDelta);
                writer.print(noteDurationChar);
            }
            lastNote = note;
        }
        writer.println();
    }

    private static void showSequencesForAllMidiFiles(String dir, String outPath, int minLengthInNotes, boolean bottomOnly) throws IOException {
        PrintWriter printWriter = new PrintWriter(outPath);
        //showHeader(printWriter,k);
        showSequencesForAllMidiFiles(new File(dir), printWriter, minLengthInNotes, bottomOnly);
        printWriter.close();
        System.out.println(countMidiFilesProcessed + " midi files");
        System.out.println(countNoteSequencesProcessed + " note sequences");
        System.out.println(countNoteSequencesProcessedWithZeroPolyphony + " with zero polyphony");
        double meanShortestLongestRatio = sumShortestLongestDurationRatios / countShortestLongestDurationRatios;
        System.out.println("smallestShortestLongestDurationRatio = " + smallestShortestLongestDurationRatio + ", mean shortest/longest ratio = " + meanShortestLongestRatio);
    }
}


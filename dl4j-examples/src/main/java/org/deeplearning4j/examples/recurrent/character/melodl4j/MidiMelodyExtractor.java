package org.deeplearning4j.examples.recurrent.character.melodl4j;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

/*
 *  MidiMelodyExtractor extracts all monophonic melodies from the midifiles and converts the melodies to strings using MelodyStrings.java.
 *  Each such melody has no harmony and consists of a single instrument playing in a single track and channel.
 *  
 *  To handle notes that change volume, there's a flag combineSamePitchNotes that lets you control whether the various notes having
 *  the same pitch are combined into one longer note or separated into multiple short notes.
 *  
 *  The percussion channel is ignored.   Melodies with too few notes or two few pitches are ignored. (See constants minSizeInNotesOfMelody and minDistinctPitches).
 *  
 *  By default, channels playing a bass instrument are skipped for purposes of extracting melodies (controlled by skipBassesForMelody flag).
 *  
 *  The flag extractMelodyFromPolyphonicNoteList determines what to do with lists of notes that contain polyphony (harmony). 
 *  If extractMelodyFromPolyphonicNoteList is true, the program extracts a list of monophonic notes from the polyphonic notes, by choosing the first 
 *  non-overlapping note of each chord.  If extractMelodyFromPolyphonicNoteList is false, the list of notes are skipped.
 *  
 *  The main method processes all files in inputDirectoryPath and outputs two files, "analysis.txt" and "melodies.txt" to outputDirectoryPath.
 *  
 *  Invoke the static method processDirectoryAndWriteMelodyAndAnalysisFiles to run this from another program.
 *  
 *  The program may print out many warning messages. These are due to bad data in MIDI files, such as an END_NOTE message with no corresponding 
 *  START_NOTE message.
 */
public class MidiMelodyExtractor  {
	public static boolean combineSamePitchNotes=true;
	public static boolean skipBassesForMelody = true; 
	public static int minSizeInNotesOfMelody=8;
	public static int minDistinctPitches=6;
	public static final String DEFAULT_INPUT_DIRECTORY_PATH="d:/Music/MIDI/CLASSICAL";
	public static final String DEFAULT_OUTPUT_DIRECTORY_PATH = System.getProperty("user.home"); // d:/tmp
	public static boolean extractMelodyFromPolyphonicNoteList=true; // If false, polyphonic note lists are skipped and not turned into melodies
	//...
    protected static final int PERCUSSION_CHANNEL = 9; // 10 in normal MIDI
    private static final NumberFormat numberFormat = NumberFormat.getInstance();
	private final String path;
	private final Sequence sequence;
	private final int [] countOfNoteInstancesPerTrack;
	private final int [] countPolyphonyPerTrack;
	private int maxChannelNumber=-1;
	private final Track[] tracks;
	private int totalNotesInstances=0;
	private int totalNotesInstancesTreatedAsDifferent=0;
	private int countOfTrackChannelInstrumentNoteSequences=0;
	private int countMelodies=0;
	private static int totalCountOfMelodies=0;
	private final List<Set<Integer>> setOfPitchesPerTrack = new ArrayList<>();
	private final List<TreeMap<Integer,TreeMap<Long,Integer>>> perTrackMapFromChannelToTickToInstrument  = new ArrayList<>();
	
	// Given the list of Notes in a track, we break it into sublists consisting of one channel and one instrument (TODO: with small gaps)
	private final List<TreeMap<Integer,TreeMap<Integer,List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes = new ArrayList<>();
	private static int totalCountPolyphonicNoteLists=0;
	private static int totalCountMonophonicNoteLists=0;
	//.................
	private static int totalCountOfUnEndedNotes=0;
	private static int couldFindStartNoteCount=0;
	private static int couldntFindStartNoteCount=0;
	private static int countMidiFiles=0;
	private static int countWarnings=0;
	static {
		numberFormat.setMaximumFractionDigits(2);
	}
	//-----------------------------
	public MidiMelodyExtractor(File file) throws InvalidMidiDataException, IOException {
		sequence= MidiSystem.getSequence(file);
		this.path=file.getAbsolutePath();
		// microsecondsPerTick=(0.0+sequence.getMicrosecondLength())/sequence.getTickLength();
		tracks=sequence.getTracks();
		countOfNoteInstancesPerTrack = new int[tracks.length];
		countPolyphonyPerTrack = new int[tracks.length];
		processTracks();
	}
	
	public void printMelodies(PrintStream melodiesPrintStream) {
		//private final List<TreeMap<Integer,TreeMap<Integer,List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes = new ArrayList<>();
		for(int trackIndex=0;trackIndex<perTrackMapFromChannelToInstrumentToListOfNotes.size();trackIndex++) {
			TreeMap<Integer,TreeMap<Integer,List<Note>>> mapFromChannelToMapFromInstrumentToListOfNotes = perTrackMapFromChannelToInstrumentToListOfNotes.get(trackIndex);
			if (mapFromChannelToMapFromInstrumentToListOfNotes!=null && mapFromChannelToMapFromInstrumentToListOfNotes.size()>0) {
				for(Map.Entry<Integer,TreeMap<Integer,List<Note>>> channelAndMapFromInstrumentAndListOfNotes: mapFromChannelToMapFromInstrumentToListOfNotes.entrySet()) {
					int channel=channelAndMapFromInstrumentAndListOfNotes.getKey();
					for(Map.Entry<Integer,List<Note>> instrumentAndListOfNotes: channelAndMapFromInstrumentAndListOfNotes.getValue().entrySet()) {
						Integer instrument=instrumentAndListOfNotes.getKey();
						String program=PlayMelodyStrings.programs[instrument];
						if (skipBassesForMelody && (program.contains("Bass") || program.contains("bass"))) {
							continue;
						}
						List<Note> notes = instrumentAndListOfNotes.getValue();
						if (noteListIsTooBoring(notes)) {
							continue;
						}
						int polyphony=computePolyphony(notes);
						if (polyphony>0) {
							totalCountPolyphonicNoteLists++;
							if (extractMelodyFromPolyphonicNoteList) {
								notes = extractMonophonicMelodyFromPolyphonicNoteList(notes);
							} else {
								continue;
							}
						} else {
							totalCountMonophonicNoteLists++;
						}
						String melodyString=MelodyStrings.convertToMelodyString(notes);
						countMelodies++;
						totalCountOfMelodies++;
						int startNote = notes.get(0).getPitch();
						melodiesPrintStream.println(MelodyStrings.COMMENT_STRING + " Track = trackIndex, Channel = "
								+ channel + ", Instrument = " + instrument + ", StartNote = " + startNote 
								+ (polyphony>0? ", polyphony = " + polyphony : "")
								);
						melodiesPrintStream.println(melodyString);
					}
				}
			}
		}
	}
	private static List<Note> extractMonophonicMelodyFromPolyphonicNoteList(List<Note> notes) {
		final List<Note> result = new ArrayList<>();
		for(Note note:notes) {
			if (!overlaps(note,result)) {
				result.add(note);
			}
		}
		return result;
	}

	private boolean noteListIsTooBoring(List<Note> notes) {
		if (notes.size()<minSizeInNotesOfMelody) {
			return true;
		}
		if (countDistinctPitches(notes)< minDistinctPitches) {
			return true;
		}
		return false;
	}

	private static int countDistinctPitches(List<Note> notes) {
		Set<Integer> pitches=new TreeSet<>();
		for(Note note:notes) {
			pitches.add(note.getPitch());
		}
		return pitches.size();
	}

	public void printAnalysis(PrintStream printStream) {
		printStream.println();
		printStream.println(path);
		printStream.println("tracks = " + tracks.length + ", totalNoteInstances = " + totalNotesInstances 
				+ ", totalNotesInstancesTreatedAsDifferent = " + totalNotesInstancesTreatedAsDifferent
				+ ", resolution = " + sequence.getResolution() + ", microsecondLength " + sequence.getMicrosecondLength()
				+ ", tickLength = " + sequence.getTickLength()  + ", countOfTrackChannelInstrumentNoteSequences = " + countOfTrackChannelInstrumentNoteSequences
				+ ", countMelodies = " + countMelodies
				);
		for(int i=0;i<tracks.length;i++) {
			if (countOfNoteInstancesPerTrack[i]>0) {
				printStream.println("Track " + i + " has " + countOfNoteInstancesPerTrack[i] + " notes, "
			+ setOfPitchesPerTrack.get(i).size() + " distinct, " + countPolyphonyPerTrack[i] + " polyphonic");
				printStream.println("channel to tick to instrument = " + perTrackMapFromChannelToTickToInstrument.get(i));
			}
		}
	}

	private void processTracks() throws InvalidMidiDataException {
		for (int i = 0; i < tracks.length; i++) {
			setOfPitchesPerTrack.add(new TreeSet<>());
			perTrackMapFromChannelToTickToInstrument.add(new TreeMap<>());
		}
		for (int trackIndex = 0; trackIndex < tracks.length; trackIndex++) {
			Track track = tracks[trackIndex];
			final TreeMap<Integer, TreeMap<Long, Integer>> mapFromChannelToTickToInstrument = perTrackMapFromChannelToTickToInstrument
					.get(trackIndex);
			final Set<Note> activeNotes = new TreeSet<>(); // This is needed to
															// find notes whose
															// velocity change
			final List<Note> notes = new ArrayList<>();
			long totalDurationOfEndedNotes = 0;
			int countOfEndedNotes = 0;
			for (int i = 0; i < track.size(); i++) {
				final MidiEvent event = track.get(i);
				final Long tick = event.getTick();
				final MidiMessage message = event.getMessage();
				if (message instanceof ShortMessage) {
					ShortMessage shortMessage = (ShortMessage) message;
					int channel = shortMessage.getChannel();
					if (channel == PERCUSSION_CHANNEL) {
						continue;
					}
					switch (shortMessage.getCommand()) {
					case ShortMessage.NOTE_ON: {
						if (channel > maxChannelNumber) {
							maxChannelNumber = channel;
						}
						Integer pitch = shortMessage.getData1();
						int velocity = shortMessage.getData2();
						if (velocity > 0) {
							totalNotesInstances++;
							setOfPitchesPerTrack.get(trackIndex).add(pitch);
							countOfNoteInstancesPerTrack[trackIndex]++;
							TreeMap<Long, Integer> mapFromTickToInstument = mapFromChannelToTickToInstrument
									.get(channel);
							if (mapFromTickToInstument == null) {
								mapFromTickToInstument = new TreeMap<>();
								mapFromChannelToTickToInstrument.put(channel, mapFromTickToInstument);
							}
							Map.Entry<Long, Integer> entry = mapFromTickToInstument.floorEntry(tick);
							int instrument = entry == null ? 0 : entry.getValue();
							if (combineSamePitchNotes) {
								if (doesAnyExistingActiveNoteHaveTheSameChannelAndPitch(channel, pitch, activeNotes,
										tick)) {
									continue;
								}
							}
							Note note = new Note(pitch, event.getTick(), instrument, channel, velocity);
							// In case the velocity or other properties of a
							// note changes.
							// One alternative would be to combine the notes
							// into one longer note that ignores velocity and
							// other changes.

							totalNotesInstancesTreatedAsDifferent++;
							deactivateAnyNotesWithSamePitchAndChannel(note, activeNotes, tick);
							notes.add(note);
							activeNotes.add(note);
						} else {
							long duration = endNote(notes, pitch, channel, tick, trackIndex, activeNotes);
							if (duration > 0) {
								totalDurationOfEndedNotes += duration;
								countOfEndedNotes++;
							}
						}
					}
						break;
					case ShortMessage.NOTE_OFF:
						int pitch = shortMessage.getData1();
						long duration = endNote(notes, pitch, channel, tick, trackIndex, activeNotes);
						if (duration > 0) {
							totalDurationOfEndedNotes += duration;
							countOfEndedNotes++;
						}
						break;
					case ShortMessage.PROGRAM_CHANGE:
						Integer instrument = shortMessage.getData1();
						// System.out.println("Program change on channel " +
						// channel + ", data1 = " + shortMessage.getData1() + ",
						// data2 = " +
						// shortMessage.getData2());
						TreeMap<Long, Integer> map2 = mapFromChannelToTickToInstrument.get(channel);
						if (map2 == null) {
							map2 = new TreeMap<>();
							mapFromChannelToTickToInstrument.put(channel, map2);
						}
						Integer oldValue = map2.put(event.getTick(), instrument);
						if (oldValue != null && !oldValue.equals(instrument)) {
							System.err.println("WARNING: in " + path + ", for track " + trackIndex + " and channel "
									+ channel + ", old value = " + oldValue + ", new value = " + instrument
									+ " at tick " + event.getTick());
							countWarnings++;
						}
						break;
					}
				}
			} // for(int i=0;i<trackSize;...)
			long avgDurationOfEndedNotes = countOfEndedNotes == 0 ? 10 : totalDurationOfEndedNotes / countOfEndedNotes;
			for (Note note : notes) {
				if (note.getEndTick() <= note.getStartTick()) {
					note.setEndTick(note.getStartTick() + avgDurationOfEndedNotes);
					totalCountOfUnEndedNotes++;
				}
			}
			notes.sort(new Comparator<Note>() {
				@Override
				public int compare(Note n1, Note n2) {
					long diff = n1.getStartTick() - n2.getStartTick();
					if (diff == 0) {
						return 0;
					} else if (diff < 0) {
						return -1;
					} else {
						return 1;
					}
				}
			});
			countPolyphonyPerTrack[trackIndex] = computePolyphony(notes);

			final TreeMap<Integer, TreeMap<Integer, List<Note>>> mapFromChannelToInstrumentToListOfNotes = makeMapFromChannelToInstrumentToListOfNotes(
					notes);
			perTrackMapFromChannelToInstrumentToListOfNotes.add(mapFromChannelToInstrumentToListOfNotes);
		} // for(int trackIndex = 0;....)

	} // void processTracks

	private static void deactivateAnyNotesWithSamePitchAndChannel(Note note, Set<Note> activeNotes, Long tick) {
		Iterator<Note> iterator = activeNotes.iterator();
		while (iterator.hasNext()) {
			Note active=iterator.next();
			if (active.getChannel()==note.getChannel() && active.getPitch()==note.getPitch()) {
				if (active.getEndTick()!=0) {
					throw new IllegalStateException();
				}
				couldFindStartNoteCount++;
				active.setEndTick(tick);
				iterator.remove();
			}
		}
	}
	private static boolean doesAnyExistingActiveNoteHaveTheSameChannelAndPitch(int channel, int pitch, Set<Note> activeNotes, Long tick) {
		Iterator<Note> iterator = activeNotes.iterator();
		while (iterator.hasNext()) {
			Note active=iterator.next();
			if (active.getChannel()==channel && active.getPitch()==pitch) {
				return true;
			}
		}
		return false;
	}

	private TreeMap<Integer, TreeMap<Integer, List<Note>>> makeMapFromChannelToInstrumentToListOfNotes(List<Note> notes) {
		final TreeMap<Integer,TreeMap<Integer,List<Note>>> mapFromChannelToInstrumentToListOfNotes= new TreeMap<>();
		List<List<Note>> noteSequences=new ArrayList<>();
		for(Note note:notes) {
			Integer channel=note.getChannel();
			TreeMap<Integer, List<Note>> mapFromInstrumentToListOfNotes = mapFromChannelToInstrumentToListOfNotes.get(channel);
			if (mapFromInstrumentToListOfNotes==null) {
				mapFromInstrumentToListOfNotes=new TreeMap<>();
				mapFromChannelToInstrumentToListOfNotes.put(channel, mapFromInstrumentToListOfNotes);
			}
			Integer instrument = note.getInstrument();
			List<Note> channelInstrumentNotes = mapFromInstrumentToListOfNotes.get(instrument);
			if (channelInstrumentNotes==null) {
				channelInstrumentNotes=new ArrayList<>();
				countOfTrackChannelInstrumentNoteSequences++;
				noteSequences.add(channelInstrumentNotes);
				mapFromInstrumentToListOfNotes.put(instrument, channelInstrumentNotes);
			}
			channelInstrumentNotes.add(note);
		}
		return mapFromChannelToInstrumentToListOfNotes;
	}

	private static boolean isSortedByStartTick(List<Note> notes) {
		long tick= Long.MIN_VALUE;
		for(Note note:notes) {
			long current=note.getStartTick();
			if (current<tick) {
				return false;
			}
			tick=current;
		}
		return true;
	}
	/**
	 * Precondition:  notes is sorted by startTick
	 * 
	 * @param notes
	 * @return the number of notes which overlap in time with another note (of higher index)
	 */
	private int computePolyphony(List<Note> notes) {
		assert(isSortedByStartTick(notes));
		int polyphony=0;
		for(int noteIndex=0;noteIndex<notes.size();noteIndex++) {
			Note note = notes.get(noteIndex);
			if (overlapsANoteWithAHigherNoteIndex(note,notes,noteIndex)) {
				polyphony++;
			}
		}
		return polyphony;
	}
	
	// Precondition: notes are sorted by startTick and note's startTick is greater than all the startTicks of notes
	public static boolean overlaps(Note note, List<Note> existingNotes) {
		long startTick = note.getStartTick();
		for(Note existsing:existingNotes) {
			if (existsing.getEndTick()>startTick) {
				return true;
			}
		}
		return false;
	}
	// Precondition: notes are sorted by startTick
	public static boolean overlapsANoteWithAHigherNoteIndex(Note note, List<Note> notes, int notesIndexOfNote) {
		assert(notes.get(notesIndexOfNote)==note);
		long endTick = note.getEndTick();
		for(int noteIndex=notesIndexOfNote+1;noteIndex<notes.size();noteIndex++) {
			Note other=notes.get(noteIndex);
			if (other.getStartTick()<endTick) {
				return true;
			}
			if (other.getStartTick()>endTick) {
				return false;
			}
		}
		return false;
	}
	private long endNote(List<Note> notes, int pitch, int channel, Long tick, int track, Set<Note> activeNotes) {
		for(int i=notes.size()-1;i>=0;i--) {
			Note other = notes.get(i);
			if (other.getChannel()==channel && other.getPitch()==pitch && other.getEndTick()==0) {
				other.setEndTick(tick);
				if (!activeNotes.remove(other)) {
					System.err.println("WARNING: in " + path + ", couldn't find activeNote for " + other);
				}
				couldFindStartNoteCount++;
				return tick-other.getStartTick();
			}
		}
		couldntFindStartNoteCount++;
		//System.err.println("WARNING: for " + path + " couldn't find start note to end pitch " + pitch + " in channel " + channel + " at tick " + tick + " in track " + track);
		return 0;
	}
	
	public static void processRecursively(File file, PrintStream analysisPrintStream, PrintStream melodiesPrintStream) {
		if (file.isDirectory()) {
			for(File child:file.listFiles()) {
				processRecursively(child,analysisPrintStream, melodiesPrintStream);
			}
		} else {
			String nameLowerCase=file.getName().toLowerCase();
			if (nameLowerCase.endsWith(".mid") || nameLowerCase.endsWith(".midi")) {
				try {				
					MidiMelodyExtractor midiFeatures = new MidiMelodyExtractor(file);
					melodiesPrintStream.println(MelodyStrings.COMMENT_STRING + file.getAbsolutePath());
					midiFeatures.printAnalysis(analysisPrintStream);
					midiFeatures.printMelodies(melodiesPrintStream);
					countMidiFiles++;
				} catch (Throwable thr) {
					System.err.println("For "+ file.getAbsolutePath() + ": " + thr);
					thr.printStackTrace();
				}
			}
		}
	}
	
	public static void processDirectoryAndWriteMelodyAndAnalysisFiles(String inputDirectoryPathLocal, String outputAnalysisFilePath, String outputMelodiesFilePath) {
		System.out.println("Processing " + inputDirectoryPathLocal);
		long startTime=System.currentTimeMillis();
		try {
			PrintStream analysisPrintStream = new PrintStream(new FileOutputStream(outputAnalysisFilePath));
			PrintStream melodiesPrintStream = new PrintStream(new FileOutputStream(outputMelodiesFilePath));
			processRecursively(new File(inputDirectoryPathLocal), analysisPrintStream, melodiesPrintStream);
			analysisPrintStream.close();
			melodiesPrintStream.close();
		
		} catch (IOException exc) {
			exc.printStackTrace();
		}
		double percent=(100.0*couldntFindStartNoteCount)/(couldntFindStartNoteCount+couldFindStartNoteCount);
		double seconds = 0.001*(System.currentTimeMillis() - startTime);
		System.out.println("couldFindStartNoteCount = " + couldFindStartNoteCount 
				+ ", couldntFindStartNoteCount " + couldntFindStartNoteCount + " (" + numberFormat.format(percent) + " %)"
				+ ", totalCountOfUnEndedNotes = " + totalCountOfUnEndedNotes
				);
		System.out.println(countMidiFiles + " midi files succeeded, " + countWarnings + " warnings in " + numberFormat.format(seconds) + " seconds.");
		System.out.println(totalCountPolyphonicNoteLists + " polyphonic note lists, " + totalCountMonophonicNoteLists + " monophonic note lists");
		System.out.println(totalCountOfMelodies + " melodies written to " + outputMelodiesFilePath);
		System.exit(0);
	}
	public static void main(String [] args) {
		processDirectoryAndWriteMelodyAndAnalysisFiles(DEFAULT_INPUT_DIRECTORY_PATH,DEFAULT_OUTPUT_DIRECTORY_PATH + "/analysis.txt", 
				DEFAULT_OUTPUT_DIRECTORY_PATH + "/melodies.txt");
	//	processDirectoryAndWriteMelodyAndAnalysisFiles("d:/music/MIDI/POP","d:/tmp/analysis-pop.txt","d:/tmp/pop-melodies.txt");
	}
}

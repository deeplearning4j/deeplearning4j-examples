/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.recurrent.character.harmonies;

/**
 * @author Don Smith (ThinkerFeeler@gmail.com)
 *
 * Extracts harmonies as strings from MIDI files, for use in Deep Learning of music.
 */

import org.deeplearning4j.examples.recurrent.character.melodl4j.MelodyStrings;
import org.deeplearning4j.examples.recurrent.character.melodl4j.Note;

import javax.sound.midi.*;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.io.*;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Pattern;

/*
 *  MidiMusicExtractor processes MIDI files and outputs two-part harmonies as strings, for processing by GravesLSTMForTwoPartHarmonies.
 *
 *  The main method of MidiMusicExtractor.java will convert a directory of MIDI files to a file containing symbolic melody strings
 *  and a file containing symbolic two-part harmony strings.
 *
 *  The program may print out many warning messages. These are due to bad data in MIDI files, such as an END_NOTE message with no corresponding
 *  START_NOTE message.
 *
 */
public class MidiMusicExtractor  {
	public static boolean combineSamePitchNotes=true;
	public static boolean skipBassesForMelody = true;
	public static int minSizeInNotesOfMelody=8;
	public static int minDistinctPitches=7;
	public static boolean useStrictOverlap=true; // When false, notes overlap if they have the same startTick. When true, they overlap if they share ticks.
	//  The loose interpretation seems to extract more melodies.

    // Whether to skip files like aaaaa.1.mid, aaaaa.2.mid (for clean_midi)
    public static boolean skipOtherVersions = false;
    public static Pattern patternForVersions = Pattern.compile("^.*\\.\\d+\\.mid");

	//In many MIDI files, a sequence of notes is played with "legato" (smoothly), meaning that there is some temporal overlap between the notes.
	//When extracting melodies we need to decide whether two successive notes are part of a monophonic melody or are in harmony (polyphony).
	//If we are strict about that decision then we will be unable to extract many melodies correctly, or we will extract every other note in the sequence.
	// The following variable lets you adjust the amount of overlap allowed before successive notes are considered in harmony.  A value of 0 would be strict.
	public static double maximumProportionOfOverlapBeforeNotesAreConsideredInHarmony=0.25;
	// After reading in the notes from the MIDI file, if the above variable has a value greater than 0, we remove make a pass through the
	// notes (sorted by start time) and a note overlaps the subsequent note by less than maximumProportionOfOverlapBeforeNotesAreConsideredInHarmony,
	// we modify the end time of the first note to coincide with the start time of the subsequent note.

	public static double minimumDurationInSecondsOfNoteToIncludeInOutput =0.02; // 1/50th of a second
	public static double maxProportionOfRepeatsOfPreviousNote=0.333;

	//............
    public final static String TMP_DIR_PATH = System.getProperty("java.io.tmpdir");
	private static boolean extractMelodyFromPolyphonicNoteList=true; // If false, polyphonic note lists are skipped and not turned into melodies
	//...
    protected static final int PERCUSSION_CHANNEL = 9; // 10 in normal MIDI
    private static final NumberFormat numberFormat = NumberFormat.getInstance();
	private final String path;
	private Sequence sequence;
	private int [] countOfNoteInstancesPerTrack;
	private int [] countPolyphonyPerTrack;
	private int maxChannelNumber=-1;
	private Track[] tracks;
	private int totalNotesInstances=0;
	private int totalNotesInstancesTreatedAsDifferent=0;
	private int countOfTrackChannelInstrumentNoteSequences=0;
	private int countMelodies=0;
	protected static int countOfNotesSkippedToTheirBeingShort=0;
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
	private static int countMelodiesRejectedDueTooFewDistinctPitches=0;
	private static int countMelodiesRejectedDueTooManyRepeatsOfPreviousNote=0;
	private static int countOfNotesTruncatedDueToLooseOverlapping=0;
	private static int countOfShortenedNotes=0;
	private MidiHarmonyUtility midiHarmonyUtility;
	static {
		numberFormat.setMaximumFractionDigits(2);
	}
	//-----------------------------
	public MidiMusicExtractor(File midiFile) throws InvalidMidiDataException, IOException {
		sequence= MidiSystem.getSequence(midiFile);
		this.path=midiFile.getAbsolutePath();
		// microsecondsPerTick=(0.0+sequence.getMicrosecondLength())/sequence.getTickLength();
		init();
	}
	public MidiMusicExtractor(Sequence sequence, String dummiePath) throws InvalidMidiDataException, IOException {
		this.path=dummiePath;
		this.sequence=sequence;
		init();
	}

	private void init()  throws InvalidMidiDataException, IOException {
		tracks=sequence.getTracks();
		countOfNoteInstancesPerTrack = new int[tracks.length];
		countPolyphonyPerTrack = new int[tracks.length];
		processTracks();
		midiHarmonyUtility = new MidiHarmonyUtility(perTrackMapFromChannelToInstrumentToListOfNotes);
		shortenNotesThatOverlapTheSubsequentNoteLessThanMaximumProportionOfOverlapBeforeNotesAreConsideredInHarmony();
	}

	public void printTwoPartHarmonies(String midiFilename, PrintWriter harmoniesWriter) throws IOException {
		final double microsecondsPerTick= sequence.getMicrosecondLength()/sequence.getTickLength();
		midiHarmonyUtility.printTwoPartHarmonies(midiFilename, microsecondsPerTick,harmoniesWriter);
	}
	private void shortenNotesThatOverlapTheSubsequentNoteLessThanMaximumProportionOfOverlapBeforeNotesAreConsideredInHarmony() {
		if (maximumProportionOfOverlapBeforeNotesAreConsideredInHarmony>0) {
			for(TreeMap<Integer,TreeMap<Integer,List<Note>>> map1: perTrackMapFromChannelToInstrumentToListOfNotes) {
				for(TreeMap<Integer,List<Note>> map2:   map1.values()) {
					for(List<Note> list:map2.values()) {
						for(int i=0;i<list.size()-1;i++) {
							Note note1=list.get(i);
							Note note2=list.get(i+1);
							double overlap=overlapProportion(note1,note2);
							if (overlap>=0 && overlap<=maximumProportionOfOverlapBeforeNotesAreConsideredInHarmony) {
								countOfShortenedNotes++;
								note1.setEndTick(note2.getStartTick());
							}
						}
					}
				}
			}
		}
	}

	//private final List<TreeMap<Integer,TreeMap<Integer,List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes = new ArrayList<>();
	public void printMelodies(PrintStream melodiesPrintStream) {
		for(int trackIndex=0;trackIndex<perTrackMapFromChannelToInstrumentToListOfNotes.size();trackIndex++) {
			TreeMap<Integer,TreeMap<Integer,List<Note>>> mapFromChannelToMapFromInstrumentToListOfNotes = perTrackMapFromChannelToInstrumentToListOfNotes.get(trackIndex);
			if (mapFromChannelToMapFromInstrumentToListOfNotes!=null && mapFromChannelToMapFromInstrumentToListOfNotes.size()>0) {
				for(Map.Entry<Integer,TreeMap<Integer,List<Note>>> channelAndMapFromInstrumentAndListOfNotes: mapFromChannelToMapFromInstrumentToListOfNotes.entrySet()) {
					int channel=channelAndMapFromInstrumentAndListOfNotes.getKey();
					for(Map.Entry<Integer,List<Note>> instrumentAndListOfNotes: channelAndMapFromInstrumentAndListOfNotes.getValue().entrySet()) {
						Integer instrument=instrumentAndListOfNotes.getKey();
						String program=PlayMusic.programs[instrument];
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
						String melodyString= MelodyStrings.convertToMelodyString(notes);
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
	private static double proportionOfNotesThatAreRepeatsOfPreviousNote(final List<Note> notes) {
		if (notes.isEmpty()) {
			return 0;
		}
		int sum=0;
		for(int i=1;i<notes.size();i++) {
			Note previousNote = notes.get(i-1);
			Note thisNote = notes.get(i);
			if (previousNote.getPitch()==thisNote.getPitch()) {
				sum++;
			}
		}
		return (0.0+sum)/notes.size();
	}
	private static List<Note> extractMonophonicMelodyFromPolyphonicNoteList(List<Note> notes) {
		final List<Note> result = new ArrayList<>();
		final Set<Note> notesToSkip = new HashSet<>(); // This contains the set of notes that we should skip because they overlap an earlier note
		Note previousNote=null;
		for(int i=0;i<notes.size();i++) {
			Note note=notes.get(i);
			if (notesToSkip.contains(note)) {
				continue;
			}
			if (previousNote!=null && previousNote.getEndTick()>note.getStartTick()) {
				countOfNotesTruncatedDueToLooseOverlapping++;
				previousNote.setEndTick(note.getStartTick());
			}
			result.add(note);
			addOverlappedNotesToNotesToSkip(note, notes, i,notesToSkip);
			previousNote=note;
		}
		return result;
	}


	private static void addOverlappedNotesToNotesToSkipNonStrict(Note note, List<Note> notes, int notesIndexOfNote, Set<Note> notesToSkip) {
		assert(notes.get(notesIndexOfNote)==note);
		long startTick=note.getStartTick();
		for(int noteIndex=notesIndexOfNote+1;noteIndex<notes.size();noteIndex++) {
			Note other=notes.get(noteIndex);
			if (other.getStartTick()==startTick) {
				notesToSkip.add(other);
			} else {
				break; // We can do this because the notes are sorted by startTick.
			}
		}
	}

	 private static void addOverlappedNotesToNotesToSkipStrict(Note note, List<Note> notes, int notesIndexOfNote, Set<Note> notesToSkip) {
         assert(notes.get(notesIndexOfNote)==note);
         long endTick = note.getEndTick();
         for(int noteIndex=notesIndexOfNote+1;noteIndex<notes.size();noteIndex++) {
                 Note other=notes.get(noteIndex);
                 if (other.getStartTick()<endTick) {
                         notesToSkip.add(other);
                 }
                 if (other.getStartTick()>endTick) { // We can do this because the notes are sorted by startTick
                         return;
                 }
         }
	 }
	 /**
		 * Pre condition: notes are sorted by startTick
		 *
		 *  We could implement one of the following semantics:
		 *     1. (strict) If note overlaps some other note at all (they play at the same time), we skip the second (other) note.
		 *     2. (loose) If two notes start at the same time, we skip the second one.
		 *
		 *     If we choose the first semantics, we omit too many melody strings.
		 * @param note
		 * @param notes
		 * @param notesIndexOfNote
		 * @param notesToSkip
		 */
	 public static void addOverlappedNotesToNotesToSkip(Note note, List<Note> notes, int notesIndexOfNote, Set<Note> notesToSkip) {
		 if (useStrictOverlap) {
			 addOverlappedNotesToNotesToSkipStrict(note, notes, notesIndexOfNote, notesToSkip);
		 } else {
			 addOverlappedNotesToNotesToSkipNonStrict(note, notes, notesIndexOfNote, notesToSkip);
		 }
	 }
	private boolean noteListIsTooBoring(List<Note> notes) {
		if (notes.size()<minSizeInNotesOfMelody) {
			return true;
		}
		if (countDistinctPitches(notes)< minDistinctPitches) {
			countMelodiesRejectedDueTooFewDistinctPitches++;
			return true;
		}
		if (proportionOfNotesThatAreRepeatsOfPreviousNote(notes)>maxProportionOfRepeatsOfPreviousNote) {
			countMelodiesRejectedDueTooManyRepeatsOfPreviousNote++;
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

	// Populates perTrackMapFromChannelToTickToInstrument, perTrackMapFromChannelToInstrumentToListOfNotes, countPolyphonyPerTrack, and countOfNoteInstancesPerTrack.
	// Note lists are sorted by startTick.
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
			if (overlapsANoteWithAHigherNoteIndexStrict(note,notes,noteIndex)) {
				polyphony++;
			}
		}
		return polyphony;
	}

	public static double overlapProportion(Note note1, Note note2) {
		long startNumerator= Math.max(note1.getStartTick(), note2.getStartTick());
		long endNumerator= Math.min(note1.getEndTick(), note2.getEndTick());
		if (startNumerator>=endNumerator) {
			return 0.0;
		}
		long startDenominator= Math.min(note1.getStartTick(), note2.getStartTick());
		long endDenominator= Math.max(note1.getEndTick(), note2.getEndTick());
		return (1.0*(endNumerator-startNumerator))/(endDenominator-startDenominator);
	}

	// Precondition: notes are sorted by startTick and note's startTick is >= all the startTicks of notes
	public static boolean overlapsStrict(Note note, List<Note> existingNotes) {
		long startTick = note.getStartTick();
		for(Note existing:existingNotes) { // TODO: this is a linear search
			if (existing.getEndTick()>startTick) {
				return true;
			}
		}
		return false;
	}
	// Precondition: notes are sorted by startTick and note's startTick is >= all the startTicks of notes
	// According to this definition, two notes overlap ONLY if they have the same startTick
		public static boolean overlapsLoose(Note note, List<Note> existingNotes) {
			long startTick = note.getStartTick();
			for(Note existing:existingNotes) { // TODO: this is a linear search
				if (existing.getStartTick()==startTick) {
					return true;
				} else {
					return false;
				}
			}
			return false;
		}
	// Precondition: notes are sorted by startTick
	public static boolean overlapsANoteWithAHigherNoteIndexStrict(Note note, List<Note> notes, int notesIndexOfNote) {
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

	public static void processRecursively(File file, PrintStream analysisPrintStream,
			PrintStream melodiesPrintStream, PrintWriter harmoniesWriter) {
		if (file.isDirectory()) {
			for(File child:file.listFiles()) {
				processRecursively(child,analysisPrintStream, melodiesPrintStream, harmoniesWriter);
			}
		} else {
			String nameLowerCase=file.getName().toLowerCase();
			if (skipOtherVersions && patternForVersions.matcher(nameLowerCase).matches()) {
			        return;
            }
			if (nameLowerCase.endsWith(".mid") || nameLowerCase.endsWith(".midi")) {
				try {
					MidiMusicExtractor midiFeatures = new MidiMusicExtractor(file);
					melodiesPrintStream.println(MelodyStrings.COMMENT_STRING + file.getAbsolutePath());
					midiFeatures.printAnalysis(analysisPrintStream);
					midiFeatures.printMelodies(melodiesPrintStream);
					midiFeatures.printTwoPartHarmonies(file.getName(),harmoniesWriter);
					countMidiFiles++;
				} catch (Throwable thr) {
					System.err.println("For "+ file.getAbsolutePath() + ": " + thr);
					thr.printStackTrace();
				}
			}
		}
	}

	public static void processDirectoryAndWriteMelodyAndAnalysisFiles(
			String inputDirectoryPathLocal,
			String outputAnalysisFilePath,
			String outputMelodiesFilePath,
			PrintWriter harmoniesWriter
			) {
		System.out.println("Processing " + inputDirectoryPathLocal);
		long startTime=System.currentTimeMillis();
		try {
			PrintStream analysisPrintStream = new PrintStream(new FileOutputStream(outputAnalysisFilePath));
			PrintStream melodiesPrintStream = new PrintStream(new FileOutputStream(outputMelodiesFilePath));
			processRecursively(new File(inputDirectoryPathLocal), analysisPrintStream, melodiesPrintStream, harmoniesWriter);
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
		System.out.println(countMelodiesRejectedDueTooFewDistinctPitches + " melodies rejected due to too few distinct pitches");
		System.out.println(countMelodiesRejectedDueTooManyRepeatsOfPreviousNote + " melodies rejected due to too many repeated pitches");
		System.out.println(totalCountPolyphonicNoteLists + " polyphonic note lists, " + totalCountMonophonicNoteLists + " monophonic note lists");
		System.out.println(totalCountOfMelodies + " melodies written to " + outputMelodiesFilePath);
		System.out.println(countOfNotesSkippedToTheirBeingShort + " notes skipped from being too short");
		System.out.println(countOfNotesTruncatedDueToLooseOverlapping + " notes truncated due to loose overlapping");
		System.out.println(countOfShortenedNotes + " notes shortened due to overlapping less than maximumProportionOfOverlapBeforeNotesAreConsideredInHarmony");
	}


	public static boolean directoryIsEmpty(File directory) {
		File[] children=directory.listFiles();
		for(File child:children) {
			if (child.getName().equals(".") || child.getName().equals("..")) { // linux, but this doesn't actually happen on linux
				continue;
			}
			if (child.isDirectory()) {
				return false;
			}
			if (child.length()>0) {
				return false;
			}
		}
		return true;
	}
	private static File chooseInputMidiDirectoryFile(String defaultDirectoryPath) {
		JFileChooser chooser = new JFileChooser(defaultDirectoryPath);
		chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		chooser.setFileFilter(new FileFilter(){
			@Override
			public boolean accept(File file) {
				return file.isDirectory();
			}

			@Override
			public String getDescription() {
				return "Directory containing midi files";
			}});
		 if (chooser.showDialog(null, "Choose midi dir")!=JFileChooser.APPROVE_OPTION) {
			 return null;
		 } else {
			 return chooser.getSelectedFile();
		 }
	}
	private static void restrictToBassAndGuitar() {
		for(int i=0;i<PlayMusic.programs.length;i++) {
			if (PlayMusic.programs[i].contains(" Bass")) {
				MidiHarmonyUtility.instrument1RestrictionSet.add(i);
			}
		}
		for(int i=0;i<PlayMusic.programs.length;i++) {
			if (PlayMusic.programs[i].contains("Electric Guitar") || PlayMusic.programs[i].contains("Acoustic Guitar")) {
				MidiHarmonyUtility.instrument2RestrictionSet.add(i);
			}
		}
	}
	public static void main(String[] args) {
		//restrictToBassAndGuitar();
		try {
			File inputDirectoryOfMidiFiles = chooseInputMidiDirectoryFile("d:/music/Midi"); // TODO: change path to your liking!!!!!!
			if (inputDirectoryOfMidiFiles == null) {
				return;
			}
			long startTime = System.currentTimeMillis();
			String directoryName = inputDirectoryOfMidiFiles.getName();
			//
			String outputRootDirectoryPath = PlayTwoPartHarmonies.ROOT_DIR_PATH + File.separator + directoryName
					+ File.separator;
			File outputRootDirectoryFile = new File(outputRootDirectoryPath);
			if (!outputRootDirectoryFile.exists()) {
			    if (!outputRootDirectoryFile.mkdirs()) {
			        System.err.println("Could not create " + outputRootDirectoryPath);
			        System.exit(1);
                }
            }
            String harmoniesOutputDirectoryPath = outputRootDirectoryPath + "harmonies-" + directoryName + ".txt";
			PrintWriter harmoniesWriter = new PrintWriter(harmoniesOutputDirectoryPath);
			processDirectoryAndWriteMelodyAndAnalysisFiles(inputDirectoryOfMidiFiles.getAbsolutePath(),
					outputRootDirectoryPath + "/analysis.txt", outputRootDirectoryPath + "/melodies.txt",
					 harmoniesWriter);
			// processDirectoryAndWriteMelodyAndAnalysisFiles("D:/Music/MIDI/pop/Beatles","d:/tmp/analysis-beatles.txt","d:/tmp/beatles-melodies-input.txt");
			double seconds = 0.001 * (System.currentTimeMillis() - startTime);
			System.out.println(seconds + " seconds");
		} catch (Throwable thr) {
			thr.printStackTrace();
			System.exit(1);
		}
	}
}

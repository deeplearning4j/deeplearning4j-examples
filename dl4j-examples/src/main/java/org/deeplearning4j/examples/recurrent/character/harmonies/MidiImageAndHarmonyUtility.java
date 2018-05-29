package org.deeplearning4j.examples.recurrent.character.harmonies;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.imageio.ImageIO;
import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import org.deeplearning4j.examples.recurrent.character.melodl4j.Note;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import lombok.val;

/**
 * This class converts Midi to Images and Images back to Midi.   The public static method playImageFile plays an image as MIDI, using your computer's MIDI software.
 * (If you try playing an arbitary image as MIDI, you'll hear wild music!)
 *
 * The class also converts MIDI to symbolic two-party harmony strings, saving them in a file (See printTwoPartHarmonies).
 *
 * If you set the variable writeInstrumentsToHarmonyStrings to true, the harmony strings will containn instrument numbers and be twice as long.
 * You should then modify GravesLSTMForTwoPartHarmonies by setting useInstruments to true.
 *
 * @author Don Smith (ThinkerFeeler@gmail.com)
 *
 */
public class MidiImageAndHarmonyUtility {
	public static boolean writeInstrumentsToHarmonyStrings=false; // Puts an instrument characters after each pitch character. But this doesn't work well.
	public static Set<Integer> instrument1RestrictionSet = new TreeSet<>();; // If you add numbers to this, the first voice must have one of the said instruments
	public static Set<Integer> instrument2RestrictionSet = new TreeSet<>(); // If you add numbers to this, the second voice must have one the said instrument s
	public static final char REST_CHAR = ' ';
	public static final String PITCH_CHARACTERS_FOR_HARMONY = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw";  // We only use up to w (49 chars)
	public static final int MIN_ALLOWED_PITCH = 36;
	public static final int MAX_ALLOWED_PITCH = 84;

	private static final int MIN_ALLOWED_NUMBER_OF_NOTES = 100;
	private static final long SMALLEST_TIME_INTERVAL_IN_MICROSECONDS = 1_000_000/16;
	private static final double MIN_PROPORTION_NON_SILENCE = 0.25;
	private static final double MAX_GAP_PROPORTION = 0.10;
	// Space represents a rest
	private static final Note REST_NOTE = new Note(0,-1,-1,-1,0);
	public static boolean isValidPitchCharacter(char ch) {
		return ch==0 || ch>='A' && ch<='Z' || ch>='a' && ch<='w';
	}
	public static boolean isValidInstrumentCharacter(char ch) {
		return ch> ' ' &&  ((int) ch) <= 108;
	}
	public static enum Encoding {
		RGB("Red=pitch, Green=instrument, Blue=Volume"),
		RGBA("Red=note:0-11, Green=pitch, Blue=instrument, Alpha=volume"),
		RGBA2("Red=note of scale:0-11, Green=scale (5=scale of middle C), Blue=instrument, Alpha=volume"),
		HSB("Hue=instrument, Saturation=volume, Brightness=pitch"),
		HSB2("Hue=pitch, Saturation=instrument, Brightness=volume"),
		HSB3("Hue=instrument, Saturation=pitch, Brightness=volume. (very lossy)"),
		;
		private String description;
		private Encoding(String descr) {
			this.description=descr;
		}
		public String getDescription() {
			return description;
		}
	}
	private static class GetNoteOrSilenceByTickFromNoteList {
		private final List<Note> notes;
		private long currentTick=0;
		private int currentIndex=0;
		public GetNoteOrSilenceByTickFromNoteList(final List<Note> notes) {
			this.notes= notes;
		}
		/**
		 *
		 * @return null if we're at the end, SILENCE if there's silence, otherwise the Note
		 */
		public Note getCurrentNote() {
			if (currentIndex==notes.size()) {
				return null;
			}
			Note note= notes.get(currentIndex);
			if (note.getStartTick()> currentTick) {
				return REST_NOTE;
			}
			return note;
		}

		public void advanceTicks(long ticks) {
			if (currentIndex == notes.size()) {
				return;
			}
			long newCurrentTick = ticks+currentTick;
			while (currentIndex<notes.size()) {
				Note note = notes.get(currentIndex);
				long tickAtEndOfCurrentNote = note.getStartTick() + note.getDurationInTicks();
				if (newCurrentTick >= tickAtEndOfCurrentNote) {
					currentIndex++;
				} else {
					break;
				}
			}
			currentTick = newCurrentTick;
		}
	}
	private static final int MAX_PITCH = 127;


	private static int defaultShortestDuration=30;
	static boolean forceNonZeroVolumeToBeAtLeastHalf=false;
	private final List<TreeMap<Integer,TreeMap<Integer,List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes;

	public MidiImageAndHarmonyUtility(final List<TreeMap<Integer,TreeMap<Integer,List<Note>>>> perTrackMapFromChannelToInstrumentToListOfNotes) {
		this.perTrackMapFromChannelToInstrumentToListOfNotes=perTrackMapFromChannelToInstrumentToListOfNotes;
	}
	/**
	 * Converts the MIDI notes into an image file.
	 *  The X coordinates represent time.
	 *  Each Y row represents a separate voice
	 *  So each row of the image is a monophonic melody.
	 *
	 * There are two formats you can choose for the pixels:
	 *
	 * 1. RGBA:
	 *     R is note value in scale (0=C, 1=C#, 2=D, .... 11=B)
	 *     G is pitch (60 is middle, 61 is C3, etc.   [Given G, you can derive R, but for Deep Learning, it's useful to have both features.]
	 *     B is instrument (0 is piano, etc)
	 *     A is volume (0 - 128)
	 *
	 *  2. HSBA (Hue, Saturation, Brightness, and Alpha)
	 *
	 *     Hue is note value in scale
	 *     Saturation is scale number
	 *     Brightness is instrument
	 *     Alpha is volume
	 *  Don't use HSBA encoding for grey-scale images, because the resulting music will be one pitch and won't play.
	 *
	 *  3. HSB
	 *     H is pitch value (60 is middle C, etc)
	 *     S is instrument
	 *     B is volume
	 *
	 *  4. HSB2:
	 *     H is instrument
	 *     S is pitch value (60 is middle C, etc)
	 *     B is volume
	 *
	 *  Saves image in outputDirectory with
	 *     1. name of  midiFileName (minus the ".mid" suffix)
	 *     2. followed by "-",
	 *     3. followed by shortestDurationOfANoteInTicks,
	 *     4. followed by ".png"
	 *  For example for "BRAND3.mid" the image will have a name like "BRAND3-14.png".
	 * @param midiFileName -- used to build output file name
	 * @param outputDirectory -- destination directory
	 * @param microsecondsPerTick
	 * @return image png file we created
	 * @throws IOException
	 */
	public File saveAsImage(final String midiFileName, final File outputDirectory, final double microsecondsPerTick, Encoding encoding) throws IOException {
		// We want to numerically encode the fact that C4 and C5 have C in common, as well as the fact that E is close to F.
		// Representing notes by pitch alone (frequency or midi pitch) encodes the latter but not the former.

		// Seems like a circle is the correct model for the pitch.    Angle encodes the note of scale (C, C# D, etc) and radius encodes scale number (C4, C5, etc).
		//Similarly, a point in a sphere encodes a note:   latitude encodes note of a scale, longitude encodes scale number, and distance from center encodes volume.

		// But the fact that 360 degrees is the same as 0 degrees, and both are close to 10 degrees, is not encoded numerically in a single number. You need a
		// function (sine or cosine).

		final List<List<Note>> voicesEachMonophonicAndOneInstrument = generateMonophonicVoicesOfOneInstrument();
		long shortestDurationOfANoteInTicks=Long.MAX_VALUE;
		for(List<Note>list: voicesEachMonophonicAndOneInstrument) {
			for(Note note:list) {
				if (note.getDurationInSeconds(microsecondsPerTick)<MidiMusicExtractor.minimumDurationInSecondsOfNoteToIncludeInImageOutput) {
					continue;
				}
				shortestDurationOfANoteInTicks=Math.min(shortestDurationOfANoteInTicks, note.getDurationInTicks());
			}
		}
		int endXOfAllVoices=0;
		System.out.println("voicesEachMonophonicAndOneInstrument.size() = " + voicesEachMonophonicAndOneInstrument.size());
		for(List<Note>list: voicesEachMonophonicAndOneInstrument) {
			endXOfAllVoices=Math.max(endXOfAllVoices, (int)((list.get(list.size()-1).getEndTick())/shortestDurationOfANoteInTicks));
		}
		int numberOfVoices=voicesEachMonophonicAndOneInstrument.size();
		System.out.println(endXOfAllVoices + ", " + MidiMusicExtractor.numberOfImageRowsPerVoice*numberOfVoices);
		final BufferedImage bufferedImage = new BufferedImage(endXOfAllVoices, MidiMusicExtractor.numberOfImageRowsPerVoice*numberOfVoices, BufferedImage.TYPE_INT_ARGB);
//		final WritableImage writableImage = new WritableImage(endXOfAllVoices, MidiMusicExtractor.numberOfImageRowsPerVoice*numberOfVoices);
//		final PixelWriter pixelWriter=writableImage.getPixelWriter();
		int count=0;
		int countXOutOfRange=0;
		int countYOutOfRange=0;
		for(int voice=0;voice<numberOfVoices;voice++) {
			int y=voice*MidiMusicExtractor.numberOfImageRowsPerVoice;
			List<Note> notes=voicesEachMonophonicAndOneInstrument.get(voice);
			for(Note note:notes) {
				if (note.getDurationInSeconds(microsecondsPerTick)<MidiMusicExtractor.minimumDurationInSecondsOfNoteToIncludeInImageOutput) {
					MidiMusicExtractor.countOfNotesSkippedToTheirBeingShort++;
					continue;
				}
				int pitch=note.getPitch(); // 21 - 108 according to spec. But I see pitches below 21
				int noteValueInScale=note.getNoteValueInScale(); // 0 - 11
				int noteValueInScale255=23*noteValueInScale;
				int volume=note.getVelocity(); // 0-127
				int instrument=note.getInstrument(); // 0 - 127
				int startX=(int)(note.getStartTick()/shortestDurationOfANoteInTicks);
				int endX=(int)(note.getEndTick()/shortestDurationOfANoteInTicks);
				if (startX>=endX) {
					System.err.println("startX=" + startX + ">=" + endX + "=endX for " + note + " in voice " + voice);
					continue;
				}

				int rgba;
				try {
					int volume255=2*volume;
					int instrument255= 2*instrument;
					int pitch255= 2*pitch;
					switch (encoding) {
					case RGB: {
						// Each parameter to Color below should be between 0 and 255 inclusive.
							rgba = new java.awt.Color(pitch255,instrument255,volume255).getRGB();
						}
						break;
					case RGBA: {
						// Each parameter to Color below should be between 0 and 255 inclusive.
							rgba = new java.awt.Color(noteValueInScale255,pitch255,instrument255,volume255).getRGB();
						}
						break;
					case RGBA2: {
						// Each parameter to Color below should be between 0 and 255 inclusive.
							int scale=pitch/12;
							// max scale is 9
							int scale255 = Math.min(255, scale*28);
							rgba = new java.awt.Color(noteValueInScale255,scale255,instrument255,volume255).getRGB();
						}
						break;
// Beware: Instrument cannot be saturation or brightness, unless we offset by 0.5, because 0 is piano, and 0 brightness is silence,
// and 0 saturation makes the hue 0.
					case HSB: {
						float hue = instrument/127.0f;
						double proportionOfMaxPitch= (0.0+Math.min(pitch,MAX_PITCH))/MAX_PITCH;
						float saturation = volume/127.0f;
						float brightness =(float) proportionOfMaxPitch;
						if (brightness>1.0f) {
							brightness=1.0f;
						}
						rgba=java.awt.Color.HSBtoRGB(hue, saturation, brightness);
						break;
					}
					case HSB2: {
						float hue = (0.0f+pitch)/MAX_PITCH;
						float saturation = (float) (0.5 + 0.5*instrument/127.0);
						float brightness = (float) (volume/127.0);
						rgba=java.awt.Color.HSBtoRGB(hue, saturation, brightness);
					}
						break;

					case HSB3: {
						float hue = (float) (instrument/127.0);
						float saturation = 0.5f+ (0.5f*pitch)/MAX_PITCH;
						float brightness = (float) (volume/127.0);
						rgba=java.awt.Color.HSBtoRGB(hue, saturation, brightness);
						if (brightness<=0) {
							System.out.println("brightness == " + brightness);
						}
					}
						break;

					default: throw new IllegalStateException();
				} // switch
					//if (count<20) {System.out.println("Writing rgba = " + String.format("%x",rgba) + " to image for " + note);}
				} catch (IllegalArgumentException exc) {
					System.err.println("pitch = " + pitch + ", noteValueInScale = " + noteValueInScale + ", instrument = " + instrument + ", volume = " + volume
							+ " got " + exc.getMessage());
					continue;
				}
				count++;
				for(int x=startX;x<endX;x++) {
					if (x>= bufferedImage.getWidth()) {
						countXOutOfRange++;
						break;
					}
					for(int yDelta=0;yDelta<MidiMusicExtractor.numberOfImageRowsPerVoice;yDelta++) {
						int yy=y+yDelta;
						if (yy>= bufferedImage.getHeight()) {
							countYOutOfRange++;
							break;
						}
						//pixelWriter.setArgb(x,y+yDelta, rgba);
						bufferedImage.setRGB(x, yy, rgba);
					} // for (int yDelta=0....)
				} // for(int x=...)
			} // for(Note note...)
			if (countXOutOfRange>0 || countYOutOfRange>0) {
				System.err.println("Warning: " + countXOutOfRange + " out of range for X, " + countYOutOfRange + " out of range for Y for " + midiFileName);
			}
		} // for (int voice=0...)
		//RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);
		String childFileName = getFilenameWithoutSuffix(midiFileName) + "-" + shortestDurationOfANoteInTicks + ".png";
		File outputPngFile = new File(outputDirectory,childFileName);
		if (outputPngFile.exists()) {
			outputPngFile.delete();
		}
		//[JPG, jpg, bmp, BMP, gif, GIF, WBMP, png, PNG, wbmp, jpeg, JPEG]
		System.out.println("Writing " + outputPngFile.getAbsolutePath() + " with " + bufferedImage.getHeight() + " rows and " + bufferedImage.getWidth() + " columns");
		if (!ImageIO.write(bufferedImage, "png", outputPngFile)) {// bmp and png are lossless
			System.err.println("Warning: unable to write " + outputPngFile.getAbsolutePath());
		}
		return outputPngFile;
	}
	private static int adjustPitch(int pitch) {
		if (pitch==0) {
			return pitch;
		}
		while (pitch>MAX_ALLOWED_PITCH) {
			pitch-=12;
		}
		while (pitch<MIN_ALLOWED_PITCH) {
			pitch+= 12;
		}
		return pitch;
	}
	private static long ticksOfNonSilence(List<Note> notes) {
		long ticks=0;
		for(Note note:notes) {
			if (note.getPitch()>0 && note.getVelocity()>0) {
				ticks += (note.getEndTick()- note.getStartTick());
			}
		}
		return ticks;
	}
	private static long longestGapInTicks(List<Note> notes) {
		long longestGap=0;
		long lastEndTick=0;
		for(Note note:notes) {
			if (note.getPitch()>0 && note.getVelocity()>0) {
				longestGap = Math.max(longestGap, note.getStartTick() - lastEndTick);
				lastEndTick=note.getEndTick();
			}
		}
		return longestGap;
	}
	// Returns characters up to ASCII  33+75 = 108
	private static char getInstrumentChar(Note note) {
		if (note==REST_NOTE || note.getVelocity()==0)  {
			return REST_CHAR;
		}
		int instr = note.getInstrument();
		if (instr>75) { // For instruments above Pan Flute, treat them as Electric Guitar (clean)
			instr=26; // Electric Guitar (clean)
		}
		instr+= 33; // So that we omit control characters. Space is REST_CHAR.
		return (char) instr;
	}
	public static int getInstrumentFrom(final char ch) {
		return (ch-33);
	}

	public void printTwoPartHarmonies(final String midiFileName, final double microsecondsPerTick, PrintWriter writer) throws IOException {
		// We want to numerically encode the fact that C4 and C5 have C in common, as well as the fact that E is close to F.
		// Representing notes by pitch alone (frequency or midi pitch) encodes the latter but not the former.

		// Seems like a circle is the correct model for the pitch.    Angle encodes the note of scale (C, C# D, etc) and radius encodes scale number (C4, C5, etc).
		//Similarly, a point in a sphere encodes a note:   latitude encodes note of a scale, longitude encodes scale number, and distance from center encodes volume.

		// But the fact that 360 degrees is the same as 0 degrees, and both are close to 10 degrees, is not encoded numerically in a single number. You need a
		// function (sine or cosine).

		final List<List<Note>> voicesEachMonophonicAndOneInstrument = generateMonophonicVoicesOfOneInstrument();
		sortVoicesByAveragePitchIgnoringSpaces(voicesEachMonophonicAndOneInstrument);
		long shortestDurationOfANoteInTicks=Long.MAX_VALUE;
		for(List<Note>list: voicesEachMonophonicAndOneInstrument) {
			for(Note note:list) {
				if (note.getDurationInSeconds(microsecondsPerTick)<MidiMusicExtractor.minimumDurationInSecondsOfNoteToIncludeInImageOutput) {
					continue;
				}
				shortestDurationOfANoteInTicks=Math.min(shortestDurationOfANoteInTicks, note.getDurationInTicks());
			}
		}
		int endXOfAllVoices=0;
		System.out.println("voicesEachMonophonicAndOneInstrument.size() = " + voicesEachMonophonicAndOneInstrument.size());
		for(List<Note>list: voicesEachMonophonicAndOneInstrument) {
			endXOfAllVoices=Math.max(endXOfAllVoices, (int)((list.get(list.size()-1).getEndTick())/shortestDurationOfANoteInTicks));
		}
		double ticksPerMicrosecond = 1.0/microsecondsPerTick;
		long tickDelta = Math.round(ticksPerMicrosecond * SMALLEST_TIME_INTERVAL_IN_MICROSECONDS);
		int numberOfVoices=voicesEachMonophonicAndOneInstrument.size();
	//	System.out.println(endXOfAllVoices + ", " + MidiMusicExtractor.numberOfImageRowsPerVoice*numberOfVoices);
		int countRejectedDueToSilence=0;
		int countRejectedDueToTooFewNotes=0;
		int countRejectedDueToLongGap=0;
		for(int voice1=0;voice1<numberOfVoices;voice1++) {
			List<Note> notes1=voicesEachMonophonicAndOneInstrument.get(voice1);
			if (notes1.size()<MIN_ALLOWED_NUMBER_OF_NOTES) {
				countRejectedDueToTooFewNotes++;
				continue;
			}
			if (instrument1RestrictionSet.size()>0 && !instrument1RestrictionSet.contains(notes1.get(0).getInstrument())) {
				continue;
			}
			final long endTick1 = notes1.get(notes1.size()-1).getEndTick();
			final long ticksOfNonSilence1= ticksOfNonSilence(notes1);
			if ((0.0+ticksOfNonSilence1)/endTick1 < MIN_PROPORTION_NON_SILENCE) {
				countRejectedDueToSilence++;
				continue;
			}
			if ((longestGapInTicks(notes1)+0.0)/endTick1 > MAX_GAP_PROPORTION) {
				countRejectedDueToLongGap++;
				continue;
			}
			for(int voice2 = voice1+1; voice2< numberOfVoices; voice2++) {
				List<Note> notes2=voicesEachMonophonicAndOneInstrument.get(voice2);
				if (notes2.size()<MIN_ALLOWED_NUMBER_OF_NOTES) {
					countRejectedDueToTooFewNotes++;
					continue;
				}
				if (instrument2RestrictionSet.size()>0 && !instrument2RestrictionSet.contains(notes2.get(0).getInstrument())) {
					continue;
				}
				final long endTick2 = notes2.get(notes2.size()-1).getEndTick();
				final long ticksOfNonSilence2= ticksOfNonSilence(notes2);
				if ((0.0+ticksOfNonSilence2)/Math.max(endTick1,endTick2) < MIN_PROPORTION_NON_SILENCE) {
					countRejectedDueToSilence++;
					continue;
				}
				if ((0.0+ticksOfNonSilence1)/Math.max(endTick1,endTick2) < MIN_PROPORTION_NON_SILENCE) {
					countRejectedDueToSilence++;
					continue;
				}
				if ((longestGapInTicks(notes2)+0.0)/endTick2 > MAX_GAP_PROPORTION) {
					countRejectedDueToLongGap++;
					continue;
				}

				GetNoteOrSilenceByTickFromNoteList selector1=new GetNoteOrSilenceByTickFromNoteList(notes1);
				GetNoteOrSilenceByTickFromNoteList selector2=new GetNoteOrSilenceByTickFromNoteList(notes2);
				StringBuilder sb = new StringBuilder();
				while (true) {
					Note note1=selector1.getCurrentNote();
					Note note2=selector2.getCurrentNote();
					if (note1==null || note2==null) {
						break;
					}
					int pitch1=adjustPitch(note1.getPitch());
					int pitch2=adjustPitch(note2.getPitch());
					char ch1= pitch1==0? REST_CHAR : PITCH_CHARACTERS_FOR_HARMONY.charAt(pitch1-MIN_ALLOWED_PITCH);
					char ch2= pitch2==0? REST_CHAR : PITCH_CHARACTERS_FOR_HARMONY.charAt(pitch2-MIN_ALLOWED_PITCH);
					if (writeInstrumentsToHarmonyStrings) {
						char instr1=  getInstrumentChar(note1);
						char instr2=  getInstrumentChar(note2);
						sb.append(ch1);
						sb.append(instr1);
						sb.append(ch2);
						sb.append(instr2);
					} else {
						sb.append(ch1);
						sb.append(ch2);
					}
					selector1.advanceTicks(tickDelta);
					selector2.advanceTicks(tickDelta);
				}
				String line=sb.toString();
				// 	Remove spaces/silence
				String line2 = PlayTwoPartHarmonies.removeSilences(line,writeInstrumentsToHarmonyStrings? 32: 16);
				if ((line.length()-line2.length())%2!=0) {
					throw new IllegalStateException();
				}
				line=line2;
				if (line.length()>= (writeInstrumentsToHarmonyStrings ? MIN_ALLOWED_NUMBER_OF_NOTES*2: MIN_ALLOWED_NUMBER_OF_NOTES)) {
					writer.println(line);
				}
			}
		}
		System.out.println("countRejectedDueToSilence = " + countRejectedDueToSilence + ", countRejectedDueToTooFewNotes = " + countRejectedDueToTooFewNotes
				 + ", countRejectedDueToLongGap = " + countRejectedDueToLongGap);
	}
	public static double getAveragePitchIgnoringSpaces(List<Note> notes) {
		int sum=0;
		int count=0;
		for(Note note:notes) {
			if (note!= REST_NOTE && note.getVelocity()>0 && note.getPitch()>0) {
				count++;
				sum+= note.getPitch();
			}
		}
		return count==0? 0: (0.0+sum)/count;
	}
	private static int getIndex(final Object obj,final Object [] objects) {
		for(int i=0;i<objects.length;i++) {
			if (obj == objects[i]) {
				return i;
			}
		}
		return -1;
	}
	private static boolean validateOrder(List<List<Note>> voices) {
		double lastAvg=0.0;
		for(List<Note> notes: voices) {
			double avg=getAveragePitchIgnoringSpaces(notes);
			if (avg<lastAvg) {
				return false;
			}
			lastAvg=avg;
		}
		return true;
	}
	// tom-adsfund suggested ordering voices like this (for Bach, where the lower voice and upper voice differ).
	private static void sortVoicesByAveragePitchIgnoringSpaces(List<List<Note>> voicesEachMonophonicAndOneInstrument) {
		final double averagePitches[] = new double[voicesEachMonophonicAndOneInstrument.size()];
		final Object[] voicesList = new Object[voicesEachMonophonicAndOneInstrument.size()];
		for(int i=0;i<voicesEachMonophonicAndOneInstrument.size();i++) {
			voicesList[i] = voicesEachMonophonicAndOneInstrument.get(i);
			averagePitches[i] = getAveragePitchIgnoringSpaces(voicesEachMonophonicAndOneInstrument.get(i));
		}
		voicesEachMonophonicAndOneInstrument.sort(new Comparator<List<Note>>(){
			@Override
			public int compare(List<Note> notes1, List<Note> notes2) {
				int index1=getIndex(notes1,voicesList);
				int index2=getIndex(notes2, voicesList);
				return Double.compare(averagePitches[index1], averagePitches[index2]);
			}});

		assert(validateOrder(voicesEachMonophonicAndOneInstrument));
	}
	private static int adjustVolume127(int volume127) {
		if (forceNonZeroVolumeToBeAtLeastHalf) {
			if (volume127>0) {
				return Math.max(63, volume127);
			}
		}
		return volume127;
	}

	private static  String getFilenameWithoutSuffix(String name) {
		int index=name.lastIndexOf('.');
		if (index>0) {
			return name.substring(0,index);
		}
		return name;
	}

	// Generates a list of lists of notes such that each list of notes is monophonic (no harmony) and consists of one instrument.
	// There may be multiple lists of notes for the same instrument.
	private List<List<Note>> generateMonophonicVoicesOfOneInstrument() {
		List<List<Note>> result = new ArrayList<>();
		int count=0;
		for(Note note:getAllNotesFromPerTrackMapFromChannelToInstrumentToListOfNotes()) {
			//if (count<20) {System.out.println(note);} count++;
			addNoteToVoiceOfSameInstrumentForWhichItOverlapsNoExistingNote(note,result);
		}
		return result;
	}

	private void addNoteToVoiceOfSameInstrumentForWhichItOverlapsNoExistingNote(Note note, List<List<Note>> result) {
		int instrument=note.getInstrument();
		for(List<Note> existingList:result) {
			Note firstNote = existingList.get(0);
			if (firstNote.getInstrument()==instrument && !MidiMusicExtractor.overlapsStrict(note,existingList)) {
				//System.out.println(note);
				existingList.add(note);
				return;
			}
		}
		List<Note>newList = new ArrayList<>();
		result.add(newList);
		newList.add(note);
	}

	private List<Note> getAllNotesFromPerTrackMapFromChannelToInstrumentToListOfNotes() {
		List<Note> allNotes = new ArrayList<>();
		for(TreeMap<Integer,TreeMap<Integer,List<Note>>> map1: perTrackMapFromChannelToInstrumentToListOfNotes) {
			for(TreeMap<Integer,List<Note>> map2:   map1.values()) {
				for(List<Note> list:map2.values()) {
					allNotes.addAll(list);
				}
			}
		}
		allNotes.sort(new Comparator<Note>(){
			@Override
			public int compare(Note n1, Note n2) {
				return n1.compareTo(n2);
			}});
		return allNotes;
	}
	  private static int getColFactorFromImagePath(String imageFileName) {
		  String imageFileNameWithoutSuffix=getFilenameWithoutSuffix(imageFileName);
		  int indexOfDash=imageFileNameWithoutSuffix.lastIndexOf('-');
		  String shortestDurationOfANoteInTicksString = imageFileNameWithoutSuffix.substring(1+indexOfDash, imageFileNameWithoutSuffix.length());
		  try {return Integer.parseInt(shortestDurationOfANoteInTicksString);}
		  catch (NumberFormatException exc) {
			  System.err.println("No shortestDurationOfANoteInTicks for " + imageFileName + ", using default: " + defaultShortestDuration);
			  return defaultShortestDuration;
		  }
	}
	  // Each row of the image represents monophonic notes of one instrument.
	  private static List<List<Note>> readVoiceNotesFromImageFile(String imagePath, Encoding encoding) throws IOException {
		  File imageFile = new File(imagePath);
		  BufferedImage bufferedImage=ImageIO.read(imageFile);
		  return readVoiceNotesFromImage(bufferedImage,imageFile.getName(),encoding,15);
	  }

	  private static List<List<Note>> readVoiceNotesFromImage(BufferedImage bufferedImage, String name, Encoding encoding,int maxChannels) throws IOException {
//		  WritableImage writableImage = SwingFXUtils.toFXImage(bufferedImage, null);
//		  PixelReader pixelReader = writableImage.getPixelReader();
		  final int width=bufferedImage.getWidth();
		  final int height=bufferedImage.getHeight();
		  final int heightDelta=height<= maxChannels? 1: height/maxChannels; // for random images, choose every 15th row
		  System.out.println(height + " voices for " +  name);
		  final List<List<Note>> result = new ArrayList<>();
		  Note note=null;
		  int colFactor=getColFactorFromImagePath(name);
		  int count=0;
		  final float[]hsbvals= new float[3];
		  long totalVolume=0;
		  final int startRow=height<60?0: 1;
		  for (int row = startRow; row < height && result.size()<maxChannels; row+=heightDelta) {
			List<Note> voice = new ArrayList<>();
			result.add(voice);
			int previousColor=Integer.MIN_VALUE;
			for (int col = 0; col < width; col++) {
				//int argb=pixelReader.getArgb(col, row);
				int argb=bufferedImage.getRGB(col,row);
				if (argb!=previousColor) {
					if (note!=null) {
						note.setEndTick(col*colFactor);
						voice.add(note);
						note=null;
						//if (count<20) {System.out.println("     Adding " + note); count++;}
					}
					int pitch;
					int instrument;
					int volume;
					switch (encoding) {
					case RGB: {
						Color color = new Color(argb, false);
						int pitch255 = color.getRed();
						int volume255 = color.getBlue();
						if (pitch255==0|| volume255==0) {
						//	System.err.println("Pitch255==0 for " + imagePath);
							continue;
						}
						int instrument255 = color.getGreen();
						pitch = pitch255 / 2;
						instrument = instrument255 / 2;
						volume = volume255 / 2;
					}
					break;
					case RGBA: {
						Color color = new Color(argb, true);
						int pitch255 = color.getGreen();
						int volume255 = color.getAlpha();
						//	rgba = new java.awt.Color(noteValueInScale255,pitch255,instrument255,volume255).getRGB();
						if (pitch255==0|| volume255==0) {
							continue;
						}
						int instrument255 = color.getBlue();
						pitch = pitch255 / 2;
						instrument = instrument255 / 2;
						volume = volume255 / 2;
					}
					break;
					case RGBA2: {
						Color color = new Color(argb, true);
						// 	rgba = new java.awt.Color(noteValueInScale255,scale255,instrument255,volume255).getRGB();
						int noteValueInScale255 = color.getRed();
						int scale255=color.getGreen();
						int instrument255 = color.getBlue();
						int volume255 = color.getAlpha();
						pitch = Math.min(MAX_PITCH, 12*(scale255/28)+(noteValueInScale255/23));
						instrument = instrument255 / 2;
						volume = volume255 / 2;
					}
					break;
					case HSB: {
						/*
						float hue = 360.0f*(instrument/127.0f);
						double proportionOfMaxPitch= (0.0+Math.min(pitch,MAX_PITCH))/MAX_PITCH;
						float saturation = (float) proportionOfMaxPitch;
						float brightness = volume/127.0f;
						 */
						Color color = new Color(argb, false);
						Color.RGBtoHSB(color.getRed(),color.getGreen(), color.getBlue(), hsbvals);
						float hue0To1 = hsbvals[0]; // /360.0f;
						float saturation0To1 = hsbvals[1];
						float brightness0To1 = hsbvals[2];
						pitch=Math.round(brightness0To1*MAX_PITCH);
						instrument=Math.round(127.0f* hue0To1);
						volume=Math.round(127.0f* saturation0To1);
						break;
					}
					case HSB2: {
						/*
						float hue = (0.0f+pitch)/MAX_PITCH;
						float saturation = (float) (0.5 + 0.5*instrument/127.0);
						float brightness = (float) (volume/127.0);
						rgba=java.awt.Color.HSBtoRGB(hue, saturation, brightness);
						 */
						Color color = new Color(argb, false);
						Color.RGBtoHSB(color.getRed(),color.getGreen(), color.getBlue(), hsbvals);
						float hue0To1 = hsbvals[0];
						float saturation0To1 = hsbvals[1];
						float brightness0To1 = hsbvals[2];
						pitch=Math.round(hue0To1*MAX_PITCH);
						instrument=2*(int)Math.round((saturation0To1-0.5)*127.0);
						volume= (int)Math.round(127.0*brightness0To1);
						if (instrument<0) {
							instrument=0;
						} else if (instrument>127) {
							instrument=127;
						}
						break;
					}

					case HSB3: {
						/*
						float hue = (float) (instrument/127.0);
						float saturation = 0.5f+ 0.5f*	pitch/MAX_PITCH;
						float brightness = (float) (volume/127.0);
						int rgb=java.awt.Color.HSBtoRGB(hue, saturation, brightness);
						 */
						Color color = new Color(argb, false);
						Color.RGBtoHSB(color.getRed(),color.getGreen(), color.getBlue(), hsbvals);
						float hue0To1 = hsbvals[0];
						float saturation0To1 = hsbvals[1];
						float brightness0To1 = hsbvals[2];
						pitch=2*(int)Math.round((saturation0To1-0.5)*127.0);
						instrument=Math.round(hue0To1*MAX_PITCH);
						volume= (int)Math.round(127.0*brightness0To1);
						if (volume<=0 || pitch<0) {
							// Why does this happen?
							continue;
						}
						if (pitch>=128) {
							pitch=127;
						}
						//System.out.println("pitch = " + pitch + ", instrument = " + instrument + ", volume = " + volume);
						break;
					}

					default:
						throw new IllegalStateException();
					}
					int startTick=col*colFactor;
					// We change the channel later
					volume=adjustVolume127(volume);
					note = new Note(pitch,startTick,instrument,0,volume);
					//if (col%17==0) System.out.println(pitch + ", " + PlayMelodyStrings.programs[instrument] + ", " + " volume = " + volume);
					//System.out.print(pitch + " "); if (col%20==0) {System.out.println();}
					previousColor=argb;
					totalVolume+=volume;
					count++;
				}
			} // for(int col = ...)
			if (note!=null) { // add last note
				note.setEndTick(width*colFactor);
				voice.add(note);
				note=null;
			}
		} // for(int row=0...)
		  System.out.println("Average volume = " + (totalVolume+0.0)/(width*result.size()));
		  return result;
	  }

	private static List<Note> findOrCreatListForInstrument(int instrument, List<List<Note>> existingInstrumentNotes) {
		  for(List<Note> list:existingInstrumentNotes) {
			if (list.get(0).getInstrument()==instrument) {
				return list;
			}
		  }
		  List<Note> notes= new ArrayList<>();
		  existingInstrumentNotes.add(notes);
		  return notes;
		}
	  // Let I be the number of distinct instruments appearing in the notes of the image file
	  // This method returns I lists of notes, one for each instrument. That is, all notes for the same instrument are put into a single track.
	  private static List<List<Note>> convertVoiceNotesToInstrumentNotes(List<List<Note>> voiceNotes) {
		  List<List<Note>> result=new ArrayList<>();
		  for(List<Note> voices:voiceNotes) {
			  if (voices.isEmpty()) {
				  continue;
			  }
			  int instrument = voices.get(0).getInstrument();
			  List<Note> notes = findOrCreatListForInstrument(instrument,result);
			  for(Note note:voices) {
				 notes.add(note);
			  }
		  }
		  for(List<Note> notes:result) {
			  notes.sort(null);
		  }
		  return result;
		}
	  private static void setInstrument(Track track, int channel, int instrument) throws InvalidMidiDataException {
		  if (instrument<0 || instrument>128) {
			  System.err.println("instrument = " + instrument);
		  }
		  MidiMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE,channel,instrument,0);
    	  track.add(new MidiEvent(midiMessage, 0));
	  }
	/**
	 * @param imagePath
	 * @return sequence corresponding to image
	 * @throws InvalidMidiDataException
	 * @throws IOException
	 */
	public static Sequence convertImageToSequence(String imagePath, Encoding encoding) throws InvalidMidiDataException, IOException {
		List<List<Note>> voiceNotes1=readVoiceNotesFromImageFile(imagePath,encoding);
		return convertVoicesToSequence(imagePath, voiceNotes1);
	}
	public static Sequence convertImageToSequence(BufferedImage image, String name, Encoding encoding) throws IOException, InvalidMidiDataException {
			return convertImageToSequence(image,name,encoding,15, new TreeSet<>());
	}
	public static Sequence convertImageToSequence(BufferedImage image, String name, Encoding encoding, int maxChannels,Set<Integer> mutedVoices) throws IOException, InvalidMidiDataException {
		List<List<Note>> voiceNotes1=readVoiceNotesFromImage(image,name,encoding,maxChannels);
		List<List<Note>> unmutedVoices = new ArrayList<>();
		for(int i=0;i<voiceNotes1.size();i++) {
			if (!mutedVoices.contains(new Integer(i))) {
				unmutedVoices.add(voiceNotes1.get(i));
			}
		}
		return convertVoicesToSequence(name, unmutedVoices);
	}
	/**
	 *
	 * @param image
	 * @param name
	 * @throws InvalidMidiDataException
	 * @throws IOException
	 * @throws MidiUnavailableException
	 */
	public static void convertImageToWavFile(BufferedImage image, String outputPath, Encoding encoding) throws IOException, InvalidMidiDataException, MidiUnavailableException {
		File waveFile=new File(outputPath);
		Sequence sequence=convertImageToSequence(image, outputPath,encoding);
		Midi2WavRenderer renderer = new Midi2WavRenderer();
		renderer.createWavFile(sequence,waveFile);
	}
	public static void sendImageAsWavFileToOutputStream(BufferedImage image, OutputStream outputStream, Encoding encoding) throws IOException, InvalidMidiDataException, MidiUnavailableException {
		Sequence sequence=convertImageToSequence(image, "(no name)",encoding);
		Midi2WavRenderer renderer = new Midi2WavRenderer();
		renderer.createWavAndSendToOutputStream(sequence, outputStream);
 	}
	private static Sequence convertVoicesToSequence(String name, List<List<Note>> voiceNotes1)
			throws FileNotFoundException, InvalidMidiDataException {
//		PrintWriter printWriter = new PrintWriter("d:/tmp/notes.txt");
//		for(List<Note> notes:voiceNotes1) {
//			for(Note note:notes) {
//				printWriter.println(note);
//			}
//			printWriter.println();
//		}
//		printWriter.close();
		List<List<Note>> instrumentNotesList = convertVoiceNotesToInstrumentNotes(voiceNotes1);
		Sequence sequence = new Sequence(Sequence.PPQ, 120 /*resolution*/);
		int countNotes=0;
		int channel=0;
		for(List<Note> instrumentNotes:instrumentNotesList) {
			Track track = sequence.createTrack();
			Note firstNote = instrumentNotes.get(0);
			setInstrument(track,channel,firstNote.getInstrument());
			for(Note note:instrumentNotes) {
				//System.out.println(note);
				note.setChannel(channel);
				note.addMidiEvents(track);
				countNotes++;
			}
			channel++;
			if (channel==9) {
				channel++; // percussion channel
			}
			if (channel==16) {
				System.err.println("Warning: more than 16 channels for " + name +  ", ignoring extra channels");
				break;
			}
		}
		// Convert each list of instrument notes to a track
		System.out.println("Made sequence with " + sequence.getTracks().length + " tracks and " + countNotes + " notes from " + name);
		return sequence;
	}

	private static void playImageFile(String imagePath, double speed, String name, boolean saveAsWav, Encoding encoding) throws InvalidMidiDataException, IOException, MidiUnavailableException {
		Sequence sequence = convertImageToSequence(imagePath,encoding);
		if (saveAsWav) {
			Midi2WavRenderer renderer = new Midi2WavRenderer();
			renderer.createWavFile(sequence,new File(MidiMusicExtractor.TMP_DIR_PATH + "/" + name + ".wav"));
		}
		PlayMusic.playSequence(sequence, speed);
	}
	// name should be like "Little_F-15.png".  It should end with a string  "-NN.png" where NN is a number representing the density of notes.
	public static void playImage(BufferedImage image,double speed, String name, Encoding encoding) throws Exception {
		playImage(image,speed,name,encoding,15);
	}
	public static void playImage(BufferedImage image,double speed, String name, Encoding encoding, int maxChannels) throws Exception {
		List<List<Note>> voiceNotes = readVoiceNotesFromImage(image, name,encoding,maxChannels);
		Sequence sequence = convertVoicesToSequence(name, voiceNotes);
		PlayMusic.playSequence(sequence, speed);
	}

	public static void main(String [] args) {
		JFileChooser fileChooser = new JFileChooser(new File("D:/Pictures"));
		fileChooser.setFileFilter(new FileFilter(){
			@Override
			public boolean accept(File file) {
				String name=file.getName().toLowerCase();
				return file.isDirectory() || name.endsWith("jpg") || name.endsWith("png");
			}
			@Override
			public String getDescription() {
				return "Image files";
			}});
		int result=fileChooser.showDialog(null,"Select");
		if (result!=JFileChooser.APPROVE_OPTION) {
			return;
		}
		String imagePath=fileChooser.getSelectedFile().getAbsolutePath();
		final File file = new File(imagePath);
		try {
			double speed=0.36;
			boolean saveAsWav=true;
			playImageFile(imagePath,speed,getFilenameWithoutSuffix(file.getName()),saveAsWav, Encoding.RGB);
		} catch (Exception exc) {
			exc.printStackTrace();
			System.exit(1);
		}
	}

}

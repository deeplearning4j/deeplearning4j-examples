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
 * This class prompts for and plays a file containing symbolic two-party harmony strings, in the syntax of MidiHarmonyUtility.java .
 * It also converts the music into a MP3 file.
 *
 * This class assumes the harmony strings do NOT contain instruments.
 *
 * By default this class plays only the first harmony line in the file.
 */

import org.deeplearning4j.examples.recurrent.character.melodl4j.Note;

import javax.sound.midi.*;
import javax.swing.*;
import java.io.*;
import java.util.Random;

public class PlayTwoPartHarmonies {
	public static int tickIncrement = 150;
	private static int transposeHalfSteps = 0;
	public static int instrument0=0; // PlayMusic.getInstrument("Flute"); // PlayMusic.getInstrument("Acoustic Bass");
	public static int instrument1=PlayMusic.getInstrument("Acoustic Guitar (nylon)"); // PlayMusic.getInstrument("Acoustic Guitar (steel)");
	public static final String ROOT_DIR_PATH = System.getProperty("user.home") +File.separator  + "midi-learning"; // Change this if you want!!!!!!!!!!!!!!!!!
	static {
		makeOrCheckDirectory(ROOT_DIR_PATH);
	}
	private static void makeOrCheckDirectory(String path) {
		File file = new File(path);
		if (file.exists() && file.isFile()) {
			throw new RuntimeException(path + " is a file, not a directory");
		}
		if (!file.exists()) {
			if (!file.mkdirs()) {
				System.err.println("ERROR: Couldn't create " + path);
				System.exit(1);
			}
		}
	}
	private static void addNoteForChar(char ch, Track track, int channel, long startTick, long endTick, int instrument) throws InvalidMidiDataException {
		int pitch= MidiHarmonyUtility.getPitchForChar(ch);
		if (pitch< MidiHarmonyUtility.MIN_ALLOWED_PITCH) {
			System.err.println("Got pitch " + pitch + " for " + ch + " at tick " + startTick);
			return;
		}
		Note note = new Note(pitch, startTick,instrument,channel,80);
		note.setEndTick(endTick);
		note.addMidiEvents(track);
	}
	public static final String makeSpaces(int count) {
		StringBuilder sb = new StringBuilder();
		for(int i=0;i<count;i++) {
			sb.append(' ');
		}
		return sb.toString();
	}
	private static void populateTrack(String part, Track track, int channel, int instrument) throws InvalidMidiDataException {
		char previousChar = 0;
		long startTick=0;
		long tick=0;
		for(int i=0;i<part.length();i++) {
			char ch = part.charAt(i);
			if (ch!=previousChar) {
				if (previousChar!= MidiHarmonyUtility.REST_CHAR && previousChar!= (char)0) {
						addNoteForChar(previousChar,track, channel,startTick,tick, instrument);
				}
				if (ch!= MidiHarmonyUtility.REST_CHAR) {
					startTick=tick;
				}
			}
			previousChar=ch;
			tick+= tickIncrement;
		}
		if (previousChar!= MidiHarmonyUtility.REST_CHAR) {
			addNoteForChar(previousChar,track,channel,startTick, tick, instrument);
		}
	}

	private static String evenPart(String harmonyString) {
		StringBuilder sb = new StringBuilder();
		for(int i=0;i<harmonyString.length();i+=2) {
			sb.append(harmonyString.charAt(i));
		}
		return sb.toString();
	}

	private static String oddPart(String harmonyString) {
		StringBuilder sb = new StringBuilder();
		for(int i=1;i<harmonyString.length();i+=2) {
			sb.append(harmonyString.charAt(i));
		}
		return sb.toString();
	}

	private static void addInstrument(Track track, int channel, int instrument) throws InvalidMidiDataException {
		ShortMessage midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE,channel,instrument,0);
    	track.add(new MidiEvent(midiMessage, 0));
	}
	public static Sequence playTwoPartHarmony(String harmonyString, String outMp3Path, String outMidiPath, int seconds, double tempoFactor) throws Exception {
		String evenPart = evenPart(harmonyString);
		String oddPart = oddPart(harmonyString);
		System.out.println(evenPart);
		System.out.println(oddPart);
		int resolution = (int)(460*tempoFactor);
		Sequence sequence = new Sequence(Sequence.PPQ, resolution); // higher resolution is faster
		if (outMidiPath!=null) {
			File outMidiFile = new File(outMidiPath);
			outMidiFile.delete();
			System.out.println("Writing to " + outMidiPath);
			FileOutputStream fos = new FileOutputStream(outMidiFile);
			int wrote=MidiSystem.write(sequence, 1, fos);
			fos.close();
			System.out.println("Wrote " + wrote + " bytes to " + outMidiPath);
		}
		Track evenTrack = sequence.createTrack();
		addInstrument(evenTrack, 0, instrument0);
		populateTrack(evenPart, evenTrack, 0, instrument0);
		Track oddTrack = sequence.createTrack();
		addInstrument(oddTrack, 1, instrument1);
		populateTrack(oddPart, oddTrack,1, instrument1);
		if (outMp3Path!=null) {
			Midi2WavRenderer.createMp3File(sequence, outMp3Path, transposeHalfSteps);
		}
		PlayMusic.playSequence(sequence, 1,seconds);  // We adjust tempo via the resolution above
		return sequence;
	}
	public final static String removeSilences(String line, int count) {
		StringBuilder sb = new StringBuilder();
		int len=line.length();
		int i=0;
		while(i < len) {
			boolean present=true;
			for(int j=i;j<i+count&& j<len;j++) {
				if (line.charAt(j)!= ' ') {
					present=false;
					break;
				}
			}
			if (present) {
				i+= count/2;
			} else {
				sb.append(line.charAt(i));
				if (i+1<len) {
					sb.append(line.charAt(i+1));
				}
				i+= 2;
			}
		}
		return sb.toString();
	}
	private static double getMeanPitchesMergingRepeats(String string) {
		int sum=0;
		int count=0;
		char lastChar = 0;
		for(int i=0;i<string.length();i++) {
			char ch= string.charAt(i);
			if (ch !=' ' && lastChar!=ch) {
				sum+= (int) ch;
				count++;
			}
			lastChar = ch;
		}
		return (sum+0.0)/count;
	}
	private static void validateOrdered(String line) {
		String even=evenPart(line);
		String odd=oddPart(line);
		double meanEven=getMeanPitchesMergingRepeats(even);
		double meanOdd=getMeanPitchesMergingRepeats(odd);
		if (meanEven > meanOdd) {
			System.err.println("Error sum of pitches is off: even = " + meanEven + ", odd = " + meanOdd);
		}
	}
	private static String removeLongRepeats(final String line, final int repeatLimit) {
		StringBuilder sb = new StringBuilder();
		int i=0;
		int countRepeats=0;
		char last0=0;
		char last1=0;
		int countRemoved=0;
		while (i+1<line.length()) {
			char ch0=line.charAt(i);
			char ch1=line.charAt(i+1);
			if (ch0==last0 && ch1==last1) {
				countRepeats++;
			} else {
				countRepeats=0;
				last0=ch0;
				last1=ch1;
			}
			if (countRepeats <= repeatLimit) {
				sb.append(ch0);
				sb.append(ch1);
			} else {
				countRemoved++;
			}
			i+=2;
		}
		if (countRemoved>0) {
			System.out.println("Removed " + countRemoved + " repeats");
		}
		return sb.toString();
	}
	public static void readFileAndPlayFirstHarmoniesLine(String path, String outMp3Path, String outMidiPath,
             int seconds, int repeatLimit, double tempoFactor) throws Exception {
		if (outMp3Path!=null && !outMp3Path.endsWith(".mp3")) {
			outMp3Path = outMp3Path + ".mp3";
		}
		System.out.println("Playing " + path);
		BufferedReader reader = new BufferedReader(new FileReader(path));
		while (true) {
			String line=reader.readLine();
			if (line==null) {
				break;
			}
			while (line.startsWith("  ")) {
				line=line.substring(2);
				if (line.isEmpty()) {
					continue;
				}
			}
			String someSpaces="          ";
			line=someSpaces+ line; // So there is a brief pause when you play it.
			String line2=removeSilences(line,8);
			System.out.println("Removed " + (line.length() - line2.length())/4 + " silences");

			line=line2;
			if (!line.isEmpty()) {
				//validateOrdered(line);
				line=removeLongRepeats(line, repeatLimit);
				playTwoPartHarmony(line,outMp3Path, outMidiPath, seconds, tempoFactor);
				break; // NOTE: This causes it to play just the first line of the file.
			}
		}
		reader.close();
	}
	private static void playRandomHarmonyFile(String directoryPath, int repeatLimit, double tempoFactor) throws Exception {
		File directory = new File(directoryPath);
		File [] files = directory.listFiles(new FileFilter(){
			@Override
			public boolean accept(File file) {
				return file.getName().endsWith("harmonies");
			}});
		Random random = new Random();
		File file = files[random.nextInt(files.length)];
		readFileAndPlayFirstHarmoniesLine(file.getAbsolutePath(), null,null, 60, repeatLimit,
            tempoFactor);
	}
	public static String removeSuffix(String name) {
		int index=name.lastIndexOf('.');
		if (index>0) {
			return name.substring(0, index);
		}
		return name;
	}
	private static File chooseInputHarmoniesFile() {
		JFileChooser chooser = new JFileChooser(ROOT_DIR_PATH);
		 if (chooser.showDialog(null, "Play harmonies file")!=JFileChooser.APPROVE_OPTION) {
			 return null;
		 } else {
			 return chooser.getSelectedFile();
		 }
	}
	public static void main(String[] args) {
		try {
			File inputHarmoniesFile = // new File(inputHarmoniesPath);
				chooseInputHarmoniesFile();
			if (inputHarmoniesFile==null) {
				return;
			}
			String nameWithoutSuffix = removeSuffix(inputHarmoniesFile.getName());
			String mp3OutPath= inputHarmoniesFile.getParent() + File.separator +nameWithoutSuffix + ".mp3";
			String midiOutPath=null; // Writing to MIDI files doesn't work yet.
			transposeHalfSteps = 0;
			int repeatLimit=16;
			int seconds=60;
			double tempoFactor=1.5;
			readFileAndPlayFirstHarmoniesLine(inputHarmoniesFile.getAbsolutePath(),mp3OutPath, midiOutPath,
                seconds,repeatLimit, tempoFactor);
			//playRandomHarmonyFile("d:/tmp/harmonies");
		} catch (Throwable thr) {
			thr.printStackTrace();
		}
		System.exit(1);
	}
}

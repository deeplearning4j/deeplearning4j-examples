/*******************************************************************************
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

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Sanity checks on the Midi-to-String and the String-to-Midi conversions.
 *
 * The method testConversion converts a midi file to melody strings then converts
 * those strings back into MIDI and plays the results on your computer speakers.
 */
public class TestMelodyConversion {
    private static List<String> convertFileToStrings(File file) throws IOException {
        final List<String> strings = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
               while (true) {
                   String line = reader.readLine();
                   if (line == null) {
                       break;
                   }
                   if (!line.startsWith("//")) {
                    strings.add(line);
                   }
               }
        }
        return strings;
    }

    /**
     * Converts midi file to melody strings and then plays back the melody strings.
     * This tests that the conversions from MIDI to strings and from strings back to MIDI
     * preserve melodies.  The sounds you hear should sound like recognizable melodies.
     * @param file
     * @throws Exception
     */
    private static void testConversion(File file) throws Exception {
        MidiMelodyExtractor extractor = new MidiMelodyExtractor(file);
        File outFile = new File(file.getParent(), file.getName() + ".txt");
        PrintStream printStream = new PrintStream(outFile);
        extractor.printMelodies(printStream);
        printStream.close();

        List<String> melodyStrings = convertFileToStrings(outFile);
        PlayMelodyStrings.playMelodies(melodyStrings, 100.0);
    }

    // This could be a unit test
    private static void sanityCheck1() {
        for(int pitchGap = -12;pitchGap<=12;pitchGap ++) {
            char ch = MelodyStrings.getCharForPitchGap(pitchGap);
            int delta = MelodyStrings.getPitchDelta(ch);
            if (pitchGap != delta){
                throw new IllegalStateException("pitchGap = " + pitchGap + ", delta = " + delta);
            }
        }
    }
    public static void main(String [] args) {
        String urlPath = args.length>0 ? args[0] : "http://waliberals.org/truthsite/music/988-aria.mid";
        //String urlPath = args.length>0 ? args[0] : "http://waliberals.org/truthsite/music/cavatina.mid";
        if (!urlPath.startsWith("http") && !urlPath.startsWith("file:")) {
            urlPath = "file:" + urlPath;
        }
        try {
            sanityCheck1();
            File midiFile = MelodyModelingExample.makeSureFileIsInTmpMidiDir(urlPath);
            testConversion(midiFile);
        } catch (Exception exc) {
            exc.printStackTrace();
        }
    }
}

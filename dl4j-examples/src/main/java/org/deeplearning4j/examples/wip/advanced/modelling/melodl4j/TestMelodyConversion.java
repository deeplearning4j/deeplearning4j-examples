package org.deeplearning4j.examples.wip.advanced.modelling.melodl4j;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

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
     * Plays converts midi file to melody strings and then plays back the melody strings.
     * This tests that the conversions from MIDI to strings and from strings back to MIDI
     * preserve melodies.
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
            File midiFile = MelodyModelingExample.makeSureFileIsInTmpDir(urlPath);
            testConversion(midiFile);
        } catch (Exception exc) {
            exc.printStackTrace();
        }
    }
}

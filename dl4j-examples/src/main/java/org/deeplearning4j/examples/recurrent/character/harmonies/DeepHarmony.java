
package org.deeplearning4j.examples.recurrent.character.harmonies;

import java.io.*;
import java.util.*;
import javax.sound.midi.MidiChannel;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Synthesizer;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import javafx.application.Application;
import javafx.scene.DepthTest;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.paint.Color;
import javafx.scene.transform.Rotate;
import javafx.stage.Stage;
import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.examples.recurrent.character.LSTMCharModellingExample;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Allows you to harmonize with a saved network, using a piano keyboard on the screen.
 * It prompts you for a saved network zip file and then displays two piano keyboards
 * on the screen.
 *
 *  You can download a saved network zip file, trained on Bach, from
 *
 *     http://deepmusic.info/BACH-layerCount_4-layerSize_100-tbpttLength_300-l2_0.0015-learningRate_0.05-updater_Adam-2018-06-16--10-44.zip
 *     (Also at http://truthsite.org/music/BACH-layerCount_4-layerSize_100-tbpttLength_300-l2_0.0015-learningRate_0.05-updater_Adam-2018-06-16--10-44.zip )
 *
 * You can play the upper piano keyboard, either by clicking with the mouse or by pressing keys on your keyboard:
 *    'q' is C2
 *    'i' is C3
 *    'c' is C4
 *    '/' is C5
 * The program waits for you to play your first note, then harmonizes with you.
 *
 * Hitting the Escape key resets the system, so that it waits again for you to play.
 *
 * There are ChoiceBoxes that let you choose which instrument plays per voice, or to mute a voice.
 *
 * This program works by sampling both voices every 1/20th of a second and feeding the corresponding
 * characters to the LSTM network by calling rnnTimeStep.
 *
 * Note: GravesLSTMForTwoPartHarmonies saves serialized network zip files at the end of epochs.
 *
 * @Author Donald A. Smith
 */
public class DeepHarmony extends Application {
    static {
        PlayMusic.loadSoundBank();
    }

    private static final int NOTE_OFF_VELOCITY_DECAY = 64;
    protected static final int WIDTH = 1200;
    protected static final int HEIGHT = 700;
    private Group root = new Group();
    private Piano humanPiano;
    private Piano neuralNetworkPiano;
    private static MultiLayerNetwork net = null;
    private static CharacterIterator characterIterator = null;
    private Synthesizer synthesizer;
    private MidiChannel midiChannelForHuman;
    private MidiChannel midiChannelForNeuralNetwork;
    private Map<Integer, Long> mapFromPitchToTimeOfPress = new HashMap<>();
    private static Map<KeyCode, Integer> mapFromKeyCodeToPitch = new HashMap<>();
    private static final Random random = new Random();
    private volatile int currentPitchHumanIsPlaying = 0;
    private volatile int currentPlayingPitchForNeuralNetwork=0;
    private int volumeForHuman=96; // default volume, out of 127
    private int volumeForNeuralNetwork= 96;
    static {
        //final int pitchForShift=
        mapFromKeyCodeToPitch.put(KeyCode.SHIFT, 55); // G
        mapFromKeyCodeToPitch.put(KeyCode.A, 56); // Ab
        mapFromKeyCodeToPitch.put(KeyCode.Z, 57); // A
        mapFromKeyCodeToPitch.put(KeyCode.S, 58); // Bb
        mapFromKeyCodeToPitch.put(KeyCode.X, 59); // B
        mapFromKeyCodeToPitch.put(KeyCode.D, 60); // C
        mapFromKeyCodeToPitch.put(KeyCode.C, 60); // C
        mapFromKeyCodeToPitch.put(KeyCode.F, 61); // C#
        mapFromKeyCodeToPitch.put(KeyCode.V, 62); // D
        mapFromKeyCodeToPitch.put(KeyCode.G, 63); // Eb
        mapFromKeyCodeToPitch.put(KeyCode.B, 64); // E
        mapFromKeyCodeToPitch.put(KeyCode.H, 65); // F
        mapFromKeyCodeToPitch.put(KeyCode.N, 65); // F
        mapFromKeyCodeToPitch.put(KeyCode.J, 66); // F#
        mapFromKeyCodeToPitch.put(KeyCode.M, 67); // G
        mapFromKeyCodeToPitch.put(KeyCode.K, 68); // A
        mapFromKeyCodeToPitch.put(KeyCode.COMMA, 69); // A
        mapFromKeyCodeToPitch.put(KeyCode.L, 70); // Bb
        mapFromKeyCodeToPitch.put(KeyCode.PERIOD, 71); // B
        mapFromKeyCodeToPitch.put(KeyCode.SEMICOLON, 72); // C
        mapFromKeyCodeToPitch.put(KeyCode.SLASH, 72); // C
        mapFromKeyCodeToPitch.put(KeyCode.QUOTE, 73); // C#


        mapFromKeyCodeToPitch.put(KeyCode.TAB, 35); // B
        mapFromKeyCodeToPitch.put(KeyCode.Q, 36); // C
        mapFromKeyCodeToPitch.put(KeyCode.W, 38); // D
        mapFromKeyCodeToPitch.put(KeyCode.E, 40); // E
        mapFromKeyCodeToPitch.put(KeyCode.R, 41); // F
        mapFromKeyCodeToPitch.put(KeyCode.T, 43); // G
        mapFromKeyCodeToPitch.put(KeyCode.Y, 45); // A
        mapFromKeyCodeToPitch.put(KeyCode.U, 47); // B
        mapFromKeyCodeToPitch.put(KeyCode.I, 48); // C
        mapFromKeyCodeToPitch.put(KeyCode.O, 50); // D
        mapFromKeyCodeToPitch.put(KeyCode.P, 52); // E
        mapFromKeyCodeToPitch.put(KeyCode.OPEN_BRACKET, 53); // F
        mapFromKeyCodeToPitch.put(KeyCode.CLOSE_BRACKET, 55); // G
        mapFromKeyCodeToPitch.put(KeyCode.BACK_SLASH, 57); // A
        mapFromKeyCodeToPitch.put(KeyCode.DELETE, 59); // B
        mapFromKeyCodeToPitch.put(KeyCode.END, 60); // C
        mapFromKeyCodeToPitch.put(KeyCode.PAGE_DOWN, 62); // D

        mapFromKeyCodeToPitch.put(KeyCode.DIGIT1, 35); // B
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT2, 37); // C#
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT3, 39); // D#
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT4, 41); // E#=F
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT5, 42); // F#
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT6, 44); // G#
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT7, 46); // A# = Bb
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT8, 48); // C
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT9, 49); // C#
        mapFromKeyCodeToPitch.put(KeyCode.DIGIT0, 51); // D#
        mapFromKeyCodeToPitch.put(KeyCode.MINUS, 53); // E
        mapFromKeyCodeToPitch.put(KeyCode.EQUALS, 54); // F#
        mapFromKeyCodeToPitch.put(KeyCode.BACK_SPACE, 56); // G#

        mapFromKeyCodeToPitch.put(KeyCode.INSERT, 58); // A#=Bb
        mapFromKeyCodeToPitch.put(KeyCode.HOME, 61); // C#
        mapFromKeyCodeToPitch.put(KeyCode.PAGE_UP, 63); // D#
    }

    //----------------------
    public static void main(String[] args) {
        try {
            String defaultDirectoryPath = System.getProperty("user.home") + "/midi-learning";
            File netFile = chooseNetworkModelFile(defaultDirectoryPath);
            if (netFile==null) {
                return; // User hit Cancel
            }
            net = ModelSerializer.restoreMultiLayerNetwork(netFile, false);
            characterIterator = GravesLSTMForTwoPartHarmonies.getCharacterIteratorForPlayBack();
            //GravesLSTMForTwoPartHarmonies.getHarmoniesIterator(64,2000);
            //loadCharacterIterator("D:/tmp/midi-character-iterator.jobj");
            System.out.println(characterIterator.inputColumns() + " input columns");
            System.out.println(characterIterator.totalOutcomes() + " total outcomes");
        } catch (Exception exc) {
            exc.printStackTrace();
            System.exit(1);
        }
        launch();
    }
    //---------------------------------------
    private static File chooseNetworkModelFile(String defaultPath) {
        JFileChooser chooser = new JFileChooser(defaultPath);
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
            "zip files", "zip", "ZIP");
        chooser.setFileFilter(filter);
        if (chooser.showDialog(null, "Choose network model file") != JFileChooser.APPROVE_OPTION) {
            return null;
        } else {
            return chooser.getSelectedFile();
        }
    }
    protected void setVolumeForHuman(int v) {
        this.volumeForHuman=v;
    }
    protected void setVolumeForNeuralNetwork(int v) {
        this.volumeForNeuralNetwork=v;
    }
    //----------
    public void setCurrentPitchHumanIsPlaying(int pitch) {
        this.currentPitchHumanIsPlaying=pitch;
    }
    //----------
    // If the user hits the ESCAPE key, it clears twoPartHarmonString. We don't compose via the neural
    // network until twoPartHarmonString is non-empty.
    // This starts with the human's character. Odd characters are composed by the neural network
    private volatile StringBuilder twoPartHarmonString = new StringBuilder();
    private void performFromNeuralNetwork() {
        while (true) {
            try {Thread.sleep(50);}
            catch (InterruptedException exc) {
                Thread.interrupted();
                System.err.println("Interrupted in sleep");
                break;
            }
            // TODO: initializing the network with the first char (in sampleFirstCharacterFromNetwork) is slow
            // (about 300 mls).
            if (currentPitchHumanIsPlaying >0 || twoPartHarmonString.length()>0) {
                char currentCharForHuman = MidiHarmonyUtility.getCharForPitch(currentPitchHumanIsPlaying);
//                long startTime=System.currentTimeMillis();
                twoPartHarmonString.append(currentCharForHuman);
                char neuralNetworkChar = twoPartHarmonString.length()==1?
                    sampleFirstCharacterFromNetwork(currentCharForHuman):
                    sampleSubsequentCharsFromNetwork(currentCharForHuman);
//                long mls = System.currentTimeMillis() -startTime;
//                System.out.println(mls + " milliseconds to get char from neural network");
                if (twoPartHarmonString.length()==0) {
                    continue; // The user must have hit ESCAPE to reset
                }
                twoPartHarmonString.append(neuralNetworkChar);
                int neuralNetworkPitch = MidiHarmonyUtility.getPitchForChar(neuralNetworkChar);
                if (neuralNetworkPitch != currentPlayingPitchForNeuralNetwork) {
                    if (currentPlayingPitchForNeuralNetwork>0) {
                        midiChannelForNeuralNetwork.noteOff(currentPlayingPitchForNeuralNetwork, NOTE_OFF_VELOCITY_DECAY);
                        neuralNetworkPiano.showPianoKeyAsNotPressed(currentPlayingPitchForNeuralNetwork);
                    }
                    if (neuralNetworkPitch>0) {
                        midiChannelForNeuralNetwork.noteOn(neuralNetworkPitch, volumeForNeuralNetwork);
                        neuralNetworkPiano.showPianoKeyAsPressed(neuralNetworkPitch);
                    }
                    currentPlayingPitchForNeuralNetwork=neuralNetworkPitch;
                }
            }
        }
    }

    private void startPlayingPitchForHuman(int pitch, long now) {
        midiChannelForHuman.noteOn(pitch, volumeForHuman);
        mapFromPitchToTimeOfPress.put(pitch, now);
        humanPiano.showPianoKeyAsPressed(pitch);
        currentPitchHumanIsPlaying = pitch;
    }

    private void stopPlayingPitchForHuman(int pitch) {
        midiChannelForHuman.noteOff(pitch, NOTE_OFF_VELOCITY_DECAY);
        mapFromPitchToTimeOfPress.remove(pitch);
        humanPiano.showPianoKeyAsNotPressed(pitch);
        if (pitch == currentPitchHumanIsPlaying) {
            currentPitchHumanIsPlaying = 0;
        }
    }

    private void handleKeyEvents(Scene scene) {
        scene.setOnKeyPressed(ke -> {
                KeyCode keyCode = ke.getCode();
                long now = System.currentTimeMillis();
                switch (keyCode) {
                    case ENTER:
                        break;
                    case ESCAPE:
                        twoPartHarmonString.setLength(0);
                        currentPitchHumanIsPlaying = 0;
                        currentPlayingPitchForNeuralNetwork=0;
                        midiChannelForHuman.allNotesOff();
                        midiChannelForNeuralNetwork.allNotesOff();
                        humanPiano.stopAllPitches();
                        neuralNetworkPiano.stopAllPitches();
                        break;
                    default: {
                        Integer pitch = mapFromKeyCodeToPitch.get(keyCode);
                        if (pitch == null) {
                            System.err.println("Warning no note for " + keyCode);
                        } else {
                            if (mapFromPitchToTimeOfPress.containsKey(pitch)) { // Still playing
                                break;
                            }
                            startPlayingPitchForHuman(pitch, now);
                        }
                    }
                }
                root.requestLayout();
        });
        scene.setOnKeyReleased(ke -> {
                KeyCode keyCode = ke.getCode();
                switch (keyCode) {
                    default:
                        Integer pitch = mapFromKeyCodeToPitch.get(keyCode);
                        if (pitch == null) {
                            break;
                        }
                        mapFromPitchToTimeOfPress.remove(pitch);
                        stopPlayingPitchForHuman(pitch);
                }
        });
    }



    //----------------
    private void initMusic() throws MidiUnavailableException {
        synthesizer = MidiSystem.getSynthesizer();
        synthesizer.open();
        MidiChannel chan[] = synthesizer.getChannels();
        // Check for null; maybe not all 16 channels exist.
        if (chan[0] != null) {
            midiChannelForHuman = chan[0];
            // chan[4].noteOn(60, 93);
        } else {
            System.err.println("Couldn't find channel 0");
            System.exit(1);
        }
        if (chan[1] != null) {
            midiChannelForNeuralNetwork = chan[1];
            // chan[4].noteOn(60, 93);
        } else {
            System.err.println("Couldn't find channel 1");
            System.exit(1);
        }
    }

    //-----------------------------------
    @Override
    public void start(Stage primaryStage) {
        try {
            startAux(primaryStage);
            Thread thread = new Thread() {
              @Override
              public void run() {
                  performFromNeuralNetwork();
              }
            };
            thread.start();
        } catch (Throwable thr) {
            thr.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Initializes the network with humanNoteChar, predicts a nn char, feeds that nn char into the network
     * and leaves state in nnOutput
     * @param humanNoteChar
     */
    public char sampleFirstCharacterFromNetwork(char humanNoteChar) {
        INDArray initializationInput = Nd4j.zeros(1, characterIterator.inputColumns(), 1);
        int idx = characterIterator.convertCharacterToIndex(humanNoteChar);
        initializationInput.putScalar(new int[]{0, idx, 0}, 1.0f);
        net.rnnClearPreviousState();
        INDArray nnOutput = net.rnnTimeStep(initializationInput);
        nnOutput = nnOutput.tensorAlongDimension(nnOutput.size(2) - 1, 1, 0);    //Gets the last time step nnOutput

        //Output is a probability distribution. Sample from this.
        double[] outputProbDistribution = new double[characterIterator.totalOutcomes()];
        for (int j = 0; j < outputProbDistribution.length; j++)
            outputProbDistribution[j] = nnOutput.getDouble(0, j);
        int sampledCharacterIdx = LSTMCharModellingExample.sampleFromDistribution(outputProbDistribution, random);
        char result= characterIterator.convertIndexToCharacter(sampledCharacterIdx);
        INDArray nextInput = Nd4j.zeros(1, characterIterator.inputColumns());
        nextInput.putScalar(new int[]{0, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
        System.out.println("First composed char = " + result);
        net.rnnTimeStep(nextInput);
        return result;
    }

    public char sampleSubsequentCharsFromNetwork(char humanNoteChar) {
        int humanCharIdx = characterIterator.convertCharacterToIndex(humanNoteChar);
        INDArray nextInput = Nd4j.zeros(1, characterIterator.inputColumns());
        nextInput.putScalar(new int[]{0, humanCharIdx}, 1.0f);
        INDArray nnOutput = net.rnnTimeStep(nextInput);
        //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
        double[] outputProbDistribution = new double[characterIterator.totalOutcomes()];
        for (int j = 0; j < outputProbDistribution.length; j++)
            outputProbDistribution[j] = nnOutput.getDouble(0, j);
        int sampledCharacterIdx = LSTMCharModellingExample.sampleFromDistribution(outputProbDistribution, random);
        char result = characterIterator.convertIndexToCharacter(sampledCharacterIdx);
        // Feed the sampled char back into the net
        nextInput.putScalar(new int[] {0,humanCharIdx}, 0.0f); // clear out old value
        nextInput.putScalar(new int[]{0, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
        net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        return result;
    }

    private void startAux(Stage primaryStage) throws Exception {
        initMusic();
        System.out.println("maxPolyphony = " + synthesizer.getMaxPolyphony()); // 64
        System.out.println("latency = " + synthesizer.getLatency()); // 200_000 microseconds = 0.200 mls
        root.setDepthTest(DepthTest.ENABLE);
        Scene scene = new Scene(root, WIDTH, HEIGHT, true);
        scene.setFill(Color.DIMGRAY);
        primaryStage.setScene(scene);
        double scale=1.9;
        double rotate= -5;
        double pianosX = 300;
        double pianosY = 150;
        double deltaY = 0.5*HEIGHT;
        double pianosZ = 0;
        humanPiano = new Piano(this,scale, pianosX,pianosY,pianosZ, true, midiChannelForHuman);
        humanPiano.setRotationAxis(Rotate.X_AXIS);
        humanPiano.setRotate(rotate);

        neuralNetworkPiano = new Piano(this,scale, pianosX, deltaY+pianosY, pianosZ, false, midiChannelForNeuralNetwork);
        neuralNetworkPiano.setRotationAxis(Rotate.X_AXIS);
        neuralNetworkPiano.setRotate(rotate);
        root.getChildren().addAll(humanPiano, neuralNetworkPiano);
        primaryStage.show();
        primaryStage.setTitle("Deep Harmony:  Play upper piano with mouse or keys: 'q'=C2, 'i'=C2, 'c'=C4, '/'=C5. ESCAPE key resets.");
        primaryStage.setOnCloseRequest(e -> {
            System.exit(0);
        });

        handleKeyEvents(scene);
    }
}

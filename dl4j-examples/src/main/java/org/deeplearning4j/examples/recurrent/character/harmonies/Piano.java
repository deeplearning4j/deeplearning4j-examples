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

import javafx.animation.AnimationTimer;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Group;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;

import javax.sound.midi.MidiChannel;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Piano keyboard for DeepHarmony
 * @Author Donald A. Smith
 */
public class Piano extends Group { // If I use VBox, the piano keys don't respond to clicks.
    public static final int BASE_PITCH = 36; // lowest C shown on piano
    private static final int NOTE_OFF_VELOCITY_DECAY = 64;
    private final static double keyWidth = 20;
    private final BlockingQueue<Runnable> uiTaskQueue = new ArrayBlockingQueue<>(20);
    private ChoiceBox<String> instrumentChoiceBox = new ChoiceBox<>();
    private static final Color WHITE_KEY_UNPRESSED_COLOR = Color.WHITE; // AZURE
    private static final Color WHITE_KEY_PRESSED_COLOR = Color.AZURE.darker();
    private static final PhongMaterial WHITE_KEY_UNPRESSED_MATERIAL = new PhongMaterial(WHITE_KEY_UNPRESSED_COLOR);
    private static final PhongMaterial WHITE_KEY_PRESSED_MATERIAL = new PhongMaterial(WHITE_KEY_PRESSED_COLOR);
    private static final PhongMaterial BLACK_KEY_UNPRESSED_MATERIAL = new PhongMaterial(Color.BLACK);
    private static final PhongMaterial BLACK_KEY_PRESSED_MATERIAL = new PhongMaterial(Color.DIMGREY);
    private final Map<Integer, Long> mapFromPitchToTimeOfPress = new HashMap<>();
    private final Map<Integer, Box> mapFromPitchToBox = new HashMap<>();
    private final boolean isHuman;
    private MidiChannel theMidiChannel;
    private final DeepHarmony deepHarmony;
    private final Slider volumeSlider = new Slider(0, 127, 80);
    private final Slider pitchBendSlider = new Slider(0,127, 64); // It actually takes 0=min,  8192=off, 16383= max
    private final Slider channelPressureSlider = new Slider(0, 127, 0);
    //private final Slider polyPressureSlider = new Slider();
    //theMidiChannel.setPitchBend(64); //  0=min,  8192=off, 16383= max
    //theMidiChannel.setChannelPressure(int) // 0=min  127=max
    int volume = 80;

    public Piano(DeepHarmony deepHarmony, double scale, double x, double y, double z, boolean isHuman, MidiChannel midiChannel) {
        this.deepHarmony=deepHarmony;
        this.isHuman = isHuman;
        this.theMidiChannel=midiChannel;
        this.setScaleX(scale);
        this.setScaleY(scale);
        this.setScaleZ(scale);
        this.setTranslateX(x);
        this.setTranslateY(y);
        this.setTranslateZ(z);
        Group piano = makePiano();
        piano.setTranslateX(10); // was -10

        configureInstrumentChoiceBox();
        HBox controlHBox = new HBox(10);
        instrumentChoiceBox.setTranslateX(0.16*DeepHarmony.WIDTH);
        instrumentChoiceBox.setTranslateY(0.065*DeepHarmony.HEIGHT);
        controlHBox.setTranslateX(5); // was 10
        controlHBox.setTranslateY(40+0.065*DeepHarmony.HEIGHT);
        //instrumentChoiceBox.setTranslateZ(-30); // If I don't do this, the choice box doesn't work.
        //piano.setTranslateZ(-50);
        controlHBox.getChildren().addAll(sliderAndRegion("Volume",volumeSlider),
            sliderAndRegion("Pressure",channelPressureSlider), sliderAndRegion("PitchBend",pitchBendSlider));
        this.getChildren().addAll(piano, instrumentChoiceBox, controlHBox);
        setUpSliders();
        animate();

    }
    private HBox sliderAndRegion(String name, Slider slider) {
        slider.setShowTickMarks(true);
        HBox hbox = new HBox(5);
        hbox.getChildren().addAll(new Label(name), slider);
        return hbox;
    }
    private void setUpSliders() {
        volumeSlider.valueProperty().addListener(new ChangeListener<Number>() {
            public void changed(ObservableValue<? extends Number> ov,
                                Number old_val, Number new_val) {
               // System.out.println("Setting volume to " + new_val.intValue());
                volume = new_val.intValue();
                if (isHuman) {
                    deepHarmony.setVolumeForHuman(volume);
                } else {
                    deepHarmony.setVolumeForNeuralNetwork(volume);
                }
            }
        });
        channelPressureSlider.valueProperty().addListener(new ChangeListener<Number>() {
            public void changed(ObservableValue<? extends Number> ov,
                                Number old_val, Number new_val) { // 0 to 127
               // System.out.println("Setting channel pressure to " + new_val.intValue());
                theMidiChannel.setChannelPressure(new_val.intValue());
            }
        });
        pitchBendSlider.valueProperty().addListener(new ChangeListener<Number>() {
            public void changed(ObservableValue<? extends Number> ov,
                                Number old_val, Number new_val) {
                // The slider's range is 0 - > 127
                //  0=min,  8192=off, 16383= max
                //System.out.println("Setting pitch bend to " + new_val.intValue());
                double pitchBend = (16383.0/127.0)* new_val.doubleValue();
                theMidiChannel.setPitchBend((int) Math.round(pitchBend));
            }
        });
    }
    //----------------------
    private void configureInstrumentChoiceBox() {
        final String MUTE = "MUTE";
        ObservableList<String> items = FXCollections.observableArrayList();
        items.add(MUTE);
        items.addAll(PlayMusic.programs);
        instrumentChoiceBox.setItems(items);
        instrumentChoiceBox.setValue(items.get(1)); // Acoustic Grand Piano
        instrumentChoiceBox.setOnAction(event -> {
            if (instrumentChoiceBox.getValue().equals(MUTE)) {
                theMidiChannel.setMute(true);
                return;
            } else {
                theMidiChannel.setMute(false);
            }
            int instrument = PlayMusic.getInstrument(instrumentChoiceBox.getValue());
            theMidiChannel.programChange(instrument);
        });
    }
    //.................
    private void makePianoKey(boolean isWhite, double x, double y, double z, Group group, final int pitch) {
        double widthRatio = isWhite ? 0.9 : 0.7;
        if (!isWhite) {
            x -= 2;
            y -= 13;
            z -= 12;
        }
        final Box box = new Box(widthRatio * keyWidth, isWhite ? keyWidth * 4 : keyWidth * 2.5, keyWidth);
        mapFromPitchToBox.put(pitch, box);
        PhongMaterial material = isWhite ? WHITE_KEY_UNPRESSED_MATERIAL : BLACK_KEY_UNPRESSED_MATERIAL;
        box.setMaterial(material);
        box.setTranslateX(x + 0.5 * (1 - widthRatio) * keyWidth);
        box.setTranslateY(y);
        box.setTranslateZ(z);
        box.setUserData(isWhite);
        group.getChildren().add(box);

        if (isHuman) {
            final long lastPressTime[] = {0};
            box.setOnMousePressed((MouseEvent me) -> {
                lastPressTime[0] = System.currentTimeMillis();
                startPlayingPitch(pitch, lastPressTime[0]);
                Runnable uiRunnable = new Runnable() {
                    @Override
                    public void run() {
                        showPianoKeyAsPressed(box, pitch);
                    }
                };
                uiTaskQueue.add(uiRunnable);
            });
            box.setOnMouseReleased(event -> {
                stopPlayingPitch(pitch);
                Runnable uiRunnable = new Runnable() {
                    @Override
                    public void run() {
                        showPianoKeyAsNotPressed(box, pitch);
                    }
                };
                uiTaskQueue.add(uiRunnable);
            });
        }
    }

    public void showPianoKeyAsPressed(int pitch) {
        Box box=mapFromPitchToBox.get(pitch);
        showPianoKeyAsPressed(box,pitch);
    }
    private void showPianoKeyAsPressed(Box box, int pitch) {
        if (box==null) { // The neural network can play keys not on our piano
            //System.err.println("showPianoKeyAsPressed: Box is null for " + pitch);
            return;
        }
        Boolean isWhite = (Boolean) box.getUserData();
        box.setMaterial(isWhite ? WHITE_KEY_PRESSED_MATERIAL : BLACK_KEY_PRESSED_MATERIAL);
    }

    public void showPianoKeyAsNotPressed(int pitch) {
        Box box=mapFromPitchToBox.get(pitch);
        showPianoKeyAsNotPressed(box, pitch);
    }
    private void showPianoKeyAsNotPressed(Box box, int pitch) {
        if (box==null) { // The neural network can play keys not on our piano
            //System.err.println("showPianoKeyAsNotPressed: Box is null for " + pitch);
            return;
        }
        Boolean isWhite = (Boolean) box.getUserData();
        box.setMaterial(isWhite ? WHITE_KEY_UNPRESSED_MATERIAL : BLACK_KEY_UNPRESSED_MATERIAL);
    }

    public void startPlayingPitch(int pitch, long now) {
        theMidiChannel.noteOn(pitch, volume);
        if (this.isHuman) {
            deepHarmony.setCurrentPitchHumanIsPlaying(pitch);
        }
        mapFromPitchToTimeOfPress.put(pitch, now);
        showPianoKeyAsPressed(pitch);
    }

    public void stopPlayingPitch(int pitch) {
        theMidiChannel.noteOff(pitch, NOTE_OFF_VELOCITY_DECAY);
        if (this.isHuman) {
            deepHarmony.setCurrentPitchHumanIsPlaying(0);
        }
        mapFromPitchToTimeOfPress.remove(pitch);
        showPianoKeyAsNotPressed(pitch);
    }
    public void stopAllPitches() {
        List<Integer> playingPitches = new ArrayList<>();
        playingPitches.addAll(mapFromPitchToTimeOfPress.keySet());
        for(Integer pitch: playingPitches) {
            stopPlayingPitch(pitch);
        }
        for(Box box: mapFromPitchToBox.values()) {
            showPianoKeyAsNotPressed(box,1);
        }
    }
    private Group makePianoKeys(int basePitch) {
        Group group = new Group();
        makePianoKey(true, 0, 0, 0, group, 0 + basePitch);
        makePianoKey(true, 1.0 * keyWidth, 0, 0, group, 2 + basePitch);
        makePianoKey(true, 2.0 * keyWidth, 0, 0, group, 4 + basePitch);
        makePianoKey(true, 3.0 * keyWidth, 0, 0, group, 5 + basePitch);
        makePianoKey(true, 4.0 * keyWidth, 0, 0, group, 7 + basePitch);
        makePianoKey(true, 5.0 * keyWidth, 0, 0, group, 9 + basePitch);
        makePianoKey(true, 6.0 * keyWidth, 0, 0, group, 11 + basePitch);

        makePianoKey(false, 0.5 * keyWidth, 0, 0, group, 1 + basePitch);
        makePianoKey(false, 1.5 * keyWidth, 0, 0, group, 3 + basePitch);
        makePianoKey(false, 3.5 * keyWidth, 0, 0, group, 6 + basePitch);
        makePianoKey(false, 4.5 * keyWidth, 0, 0, group, 8 + basePitch);
        makePianoKey(false, 5.5 * keyWidth, 0, 0, group, 10 + basePitch);
        return group;
    }

    private Group makePiano() {
        Group group = new Group();
        Group scale0 = makePianoKeys(BASE_PITCH);
        scale0.setTranslateX(0);

        Group scale1 = makePianoKeys(48);
        scale1.setTranslateX(keyWidth * 7);

        Group scale2 = makePianoKeys(60);
        scale2.setTranslateX(keyWidth * 14);

        Group scale3 = makePianoKeys(72);
        scale3.setTranslateX(keyWidth * 21);

        Group scale4 = new Group();
        makePianoKey(true, 0, 0, 0, scale4, 84);
        scale4.setTranslateX(keyWidth * 28);
        group.getChildren().addAll(scale0, scale1, scale2, scale3, scale4);
        return group;
    }


    //------------------------------
    private void animate() {
        final AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long nowInNanoSeconds) {
                final long now = System.currentTimeMillis();
                while (true) {
                    Runnable runnable = uiTaskQueue.poll();
                    if (runnable == null) {
                        break;
                    }
                    runnable.run();
                }
                Iterator<Map.Entry<Integer, Long>> iterator = mapFromPitchToTimeOfPress.entrySet().iterator();
                List<Integer> pitchesToStop = new ArrayList<>();
                while (iterator.hasNext()) {
                    Map.Entry<Integer, Long> entry = iterator.next();
                    if (entry.getKey() == 72 && now - entry.getValue() > 1000) {
                        pitchesToStop.add(entry.getKey());
                    }
                }
                for (Integer pitch : pitchesToStop) {
                    stopPlayingPitch(pitch);
                }
            } // void handle
        };
        timer.start();
    }
}

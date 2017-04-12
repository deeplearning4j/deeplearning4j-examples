
package org.deeplearning4j.examples.recurrent.character.melodl4j;

    import java.io.File;
    import java.io.FileInputStream;
    import java.io.IOException;
    import java.io.ObjectInputStream;
    import java.util.*;
    import java.util.concurrent.ArrayBlockingQueue;
    import java.util.concurrent.BlockingQueue;
    import javafx.animation.AnimationTimer;
    import javafx.application.Application;
    import javafx.event.EventHandler;
    import javafx.geometry.Insets;
    import javafx.scene.DepthTest;
    import javafx.scene.Group;
    import javafx.scene.Scene;
    import javafx.scene.control.Button;
    import javafx.scene.control.TextField;
    import javafx.scene.control.Tooltip;
    import javafx.scene.input.KeyCode;
    import javafx.scene.input.KeyEvent;
    import javafx.scene.input.MouseEvent;
    import javafx.scene.layout.*;
    import javafx.scene.paint.Color;
    import javafx.scene.paint.Paint;
    import javafx.scene.paint.PhongMaterial;
    import javafx.scene.shape.Box;
    import javafx.scene.transform.Rotate;
    import javafx.stage.Stage;
    import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
    import org.deeplearning4j.util.ModelSerializer;

public class Piano extends Application {
    private static final String defaultSavedNetworkPath="/tmp/MelodyModel-bach.zip";
    private static final String defaultSavedIteratorPath="/tmp/midi-character-iterator.jobj";
    private static final int WIDTH=1000;
    private static final int HEIGHT=400;
    private static final double KEY_PRESS_DISTANCE_Y=1;
    private static final double KEY_PRESS_DISTANCE_Z=8;
    private Group root = new Group();
    private Stage stage;
    private final BlockingQueue<Box> boxesToRestore = new ArrayBlockingQueue<>(10);
    private double keyWidth=20;
    private static MultiLayerNetwork net=null;
    private static CharacterIterator characterIterator=null;
    private static Random random = new Random(1234);
    private TextField notesEditor = new TextField();
    private int initialPitch=48;
    //----------------------
    public static void main(String[] args) {
		try {
            String netPathName=args.length>0? args[0]: defaultSavedNetworkPath; // You must first create this in M
            String iteratorPath=args.length>1? args[1]: defaultSavedIteratorPath;
            net = ModelSerializer.restoreMultiLayerNetwork(new File(netPathName), false);
            characterIterator= loadCharacterIterator(iteratorPath);
        }
        catch (Exception exc) {
           exc.printStackTrace();
           System.exit(1);
        }
        launch(new String[]{});
    }
    // Frere Jacques:
    // C5q D5q E5q C5q   C5q D5q E5q C5q   E5q F5q G5q Rq    E5q F5q G5q Rq   G5i A5i G5i F5i E5q C5q Rt   G5i A5i G5i F5i E5q C5q Rt   C5q G4q C5q Rq  C5q G4q C5q
    //  C5q B4q Eb5q C5q Rq D5q Eb5q G5q F5q Eb5q D5q F5q B4q  =  l!l4h#lRl2l1l4l@l@l!l3l^l
    //  C5q Eb5q G5q F5q Rq Eb5q D5q F5q Eb5q D5q Eb5q D5q C5q
    //----------
    private static CharacterIterator loadCharacterIterator(String inPath) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(inPath));
        CharacterIterator iter = (CharacterIterator) ois.readObject();
        ois.close();
        System.out.println("iter = " + iter);
        return iter;
    }
    //-----------
    private void composeAndPlayInSeparateThread() {
        Runnable runnable = new Runnable() {
          @Override
          public void run() {
            try {
                composeAndPlay();
            } catch (Exception exc) {
                exc.printStackTrace();
            }
          }
        };
        new Thread(runnable).start();
    }
    //-----------
    private void composeAndPlay() throws Exception {
        String initialization= convertNotesEditorNotesToMelodyString();
        System.out.println("Composing with initialization = "+  initialization);
        String[] melodies = MelodyModelingExample.sampleCharactersFromNetwork(initialization,net,characterIterator,random,100,1);
        for(String melody: melodies) {
            PlayMelodyStrings.playMelody(melody,20,initialPitch);
        }
    }
    //----------------------
    // Mutates the member variable initialPitch
    private String convertNotesEditorNotesToMelodyString() {
        String[] notes = notesEditor.getText().trim().replace("r","R").split("\\s+");
        for(String note:notes) {
            System.out.print(note + " ");
        }
        System.out.println();
        if (notes.length==0) {
            return "";
        }
        initialPitch = SymbolicNotes.getPitchFromJFuguePatternString(notes[0]);
        System.out.println("initial pitch = " + initialPitch  + " for " + notes[0]);
        if (initialPitch<=0) {
            System.exit(1);
        }

        char initialDurationChar = SymbolicNotes.getMelodyDurationCharacterFromJFuguePatternString(notes[0]);
        System.out.println("Initial duration char = " + initialDurationChar);
        StringBuilder sb = new StringBuilder();
        sb.append(initialDurationChar);
        int lastPitch=initialPitch;
        for(int i=1;i<notes.length;i++) {
            String note = notes[i];
            if (note.startsWith("R")) {
                sb.append("R");
                sb.append(SymbolicNotes.getMelodyDurationCharacterFromJFuguePatternString(note));
            } else {
                int pitch=SymbolicNotes.getPitchFromJFuguePatternString(note);

                int deltaPitch=pitch-lastPitch;
                char pitchChar= Midi2MelodyStrings.getCharForPitchGap(deltaPitch);
                sb.append(pitchChar);
                lastPitch=pitch;
                char durationChar = SymbolicNotes.getMelodyDurationCharacterFromJFuguePatternString(note);
                sb.append(durationChar);
            }
        }
        return sb.toString();
    }
    //----------
    private void playSavedNotes() {
        if (notesEditor.getText().trim().isEmpty()) {
            return;
        }

        String melody= convertNotesEditorNotesToMelodyString();
        System.out.println("Playing " + melody + " with initial pitch " + initialPitch);
        System.out.println(">>>>" + SymbolicNotes.melodyToJFugue(melody, initialPitch));
        try {
            PlayMelodyStrings.playMelody(melody, 100, initialPitch);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    //---------
    private void addNote(int rawPitch) {
        String note = SymbolicNotes.pitchToJFugueNoteStringWithDefaultDuration(rawPitch);
        if (notesEditor.getText().length()>0) {
            note = " " + note;
        }
        notesEditor.setText(notesEditor.getText() + note);
    }
    //.................
    private void makePianoKey(boolean isWhite, double x, double y, double z, Group group, final int pitch) {
        double widthRatio = isWhite? 0.9 : 0.7;
        if (!isWhite) {
            x-= 2;
            y-= 13;
            z -= 12;
        }
        final Box box = new Box(widthRatio*keyWidth, isWhite? keyWidth*4 : keyWidth*2.5,keyWidth);
        PhongMaterial material = new PhongMaterial(isWhite?  Color.AZURE: Color.BLACK);
        box.setMaterial(material);
        box.setTranslateX(x+0.5*(1-widthRatio)*keyWidth);
        box.setTranslateY(y);
        box.setTranslateZ(z);
        group.getChildren().add(box);
        box.setOnMousePressed((MouseEvent me) -> {
            Runnable runnable = new Runnable() {
                @Override
                public void run() {
                    try {
                        addNote(pitch);
                        PlayMelodyStrings.playMelody("s", 1, pitch);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            };
            new Thread(runnable).start();
            box.setTranslateZ(KEY_PRESS_DISTANCE_Z + box.getTranslateZ());
            box.setTranslateY(-KEY_PRESS_DISTANCE_Y + box.getTranslateY());
            box.setUserData(new Long(System.currentTimeMillis()));
            synchronized (boxesToRestore) {
                boxesToRestore.add(box);
            }
            root.requestLayout();
        });
    }
    private void handleKeyEvents(Scene scene) {
        scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
            public void handle(KeyEvent ke) {
                switch (ke.getCode()) {
                    case Q:
                        System.exit(0);
                        break;
                    case ENTER:
                        composeAndPlayInSeparateThread();
                        break;
                    case P:
                        playSavedNotes();
                        break;
//                    case A: case S: case D: case F: case G: case H: case J: case  K: case  L: case COLON: case QUOTE:
//                          case Z: case X: case C: case V: case B: case N: case M: case COMMA: case PERIOD: case BACK_SLASH:
//                          if (!keyCodeToTimePressedInMlsMap.containsKey(ke.getCode())) {
//                              System.out.println("Got key press " + ke.getCode());
//                              keyCodeToTimePressedInMlsMap.put(ke.getCode(), System.currentTimeMillis());
//                          }
//                        break;

                    case BACK_SPACE:
                        String text=notesEditor.getText().trim();
                        int index=text.lastIndexOf(' ');
                        if (index>0) {
                            text=text.substring(0,index);
                            notesEditor.setText(text);
                            notesEditor.requestLayout();
                        }
                        break;
                    default:
                        break;
                }
                root.requestLayout();
            }
        });
        scene.setOnKeyReleased(new EventHandler<KeyEvent>() {
            public void handle(KeyEvent ke) {
                switch (ke.getCode()) {
//                    case A: case S: case D: case F: case G: case H: case J: case  K: case  L: case COLON: case QUOTE:
//                    case Z: case X: case C: case V: case B: case N: case M: case COMMA: case PERIOD: case BACK_SLASH:
//                            Long startTime = keyCodeToTimePressedInMlsMap.get(ke.getCode());
//                            if (startTime==null) {
//                                System.err.println("Warning: no start time for key code " + ke.getCode());
//                            } else {
//                                keyCodeToTimePressedInMlsMap.remove(ke.getCode());
//                                lastNoteStopTimeInMls=System.currentTimeMillis();
//                                long mlsDurationOfNote = lastNoteStopTimeInMls-startTime.longValue();
//                                System.out.println(ke.getCode() + " release after " + mlsDurationOfNote + " mls");
//                                int timeUnits = (int) (mlsDurationOfNote/125);
//                                if (timeUnits>Midi2MelodyStrings.durationChars.length()) {
//                                    timeUnits = Midi2MelodyStrings.durationChars.length()-1;
//                                }
//                                char durationChar =Midi2MelodyStrings.durationChars.charAt(timeUnits);
//                                String symbolicNote = getSymbolForKeyEvent(ke) + durationChar;
//                                // TODO
//                            }
//                    break;
                    default:
                        break;
                }
            }
        });
    }
    private String getSymbolForKeyEvent(KeyEvent ke) {
        int scale=5;
        if (ke.isShiftDown()) {
            scale=6;
        } else if (ke.isControlDown()) {
            scale=4;
        }
        KeyCode code= ke.getCode();
        switch (code) {
                    case C:
                        return "C" + scale;
                    case V:
                        return "D" + scale;
                    case B:
                        return "E" + scale;
                    case N:
                        return "F" + scale;
                    case M:
                        return "G"+ scale;
                    case COMMA:
                        return "A" + scale;
                    case PERIOD:
                        return "B"+ scale;
                    case SLASH:
                        scale++;
                        return "C" + scale;
                    case R:
                        return "R" + scale;
            default:
                return null;
        }
    }
    private void animate() {
        final AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long nowInNanoSeconds) {
                synchronized (boxesToRestore) {
                    Box box = boxesToRestore.peek();
                    if (box != null) {
                        long now = System.currentTimeMillis();
                        long boxTime = (Long) box.getUserData();
                        if (now - boxTime > 600) {
                            boxesToRestore.poll(); // remove it
                            box.setTranslateZ(-KEY_PRESS_DISTANCE_Z + box.getTranslateZ());
                            box.setTranslateY(KEY_PRESS_DISTANCE_Y + box.getTranslateY());
                        }
                    }
                }
            }
        };
        timer.start();
    }
    private Group makePianoKeys(int basePitch) {
        Group group = new Group();
        makePianoKey(true,  0, 0, 0, group,0+basePitch);
        makePianoKey(true,  1.0*keyWidth, 0, 0, group,2+basePitch);
        makePianoKey(true, 2.0*keyWidth, 0, 0, group,4+basePitch);
        makePianoKey(true, 3.0*keyWidth, 0, 0, group,5+basePitch);
        makePianoKey(true, 4.0*keyWidth, 0, 0, group,7+basePitch);
        makePianoKey(true, 5.0*keyWidth, 0, 0, group,9+basePitch);
        makePianoKey(true, 6.0*keyWidth, 0, 0, group,11+basePitch);

        makePianoKey(false, 0.5*keyWidth, 0, 0, group,1+basePitch);
        makePianoKey(false, 1.5*keyWidth, 0, 0, group,3+basePitch);
        makePianoKey(false, 3.5*keyWidth, 0, 0, group,6+basePitch);
        makePianoKey(false, 4.5*keyWidth, 0, 0, group,8+basePitch);
        makePianoKey(false, 5.5*keyWidth, 0, 0, group,10+basePitch);
        return group;
    }
    private Group makePiano() {
        Group group = new Group();
        Group scale1 = makePianoKeys(48);
        scale1.setTranslateX(0);

        Group scale2 = makePianoKeys(60);
        scale2.setTranslateX(keyWidth*7);

        Group scale3 = makePianoKeys(72);
        scale3.setTranslateX(keyWidth*14);

        Group scale4 = new Group();
        makePianoKey(true,  0, 0, 0, scale4,84);
        scale4.setTranslateX(keyWidth*21);
        group.getChildren().addAll(scale1,scale2,scale3,scale4);
        return group;
    }
    //---------
    private void makePlayButton() {
        Button playButton = new Button("Play");
        playButton.setTranslateX(WIDTH/2-120);
        playButton.setTranslateY(HEIGHT-50);
        playButton.setTooltip(new Tooltip("To preview the notes you entered."));
        Insets insets= Insets.EMPTY;
        CornerRadii radii= new CornerRadii(3);
        Paint paint = Color.AQUAMARINE;
        BackgroundFill fill= new BackgroundFill(paint,radii,insets);
        Background background = new Background(fill);
        playButton.setBackground(background);
        playButton.setOnAction(e -> playSavedNotes());
        root.getChildren().add(playButton);
    }

    //---------
    private void makeComposeButton() {
        Button composeButton = new Button("Compose");
        composeButton.setTranslateX(WIDTH/2+80);
        composeButton.setTranslateY(HEIGHT-50);
        composeButton.setTooltip(new Tooltip("To compose a melody that starts with the notes you entered."));
        Insets insets= Insets.EMPTY;
        CornerRadii radii= new CornerRadii(3);
        Paint paint = Color.AQUAMARINE;
        BackgroundFill fill= new BackgroundFill(paint,radii,insets);
        Background background = new Background(fill);
        composeButton.setBackground(background);
        composeButton.setOnAction(e -> composeAndPlayInSeparateThread());
        root.getChildren().add(composeButton);
    }

    private void makeNotesEditor() {
        notesEditor.setPrefColumnCount(70);
        CornerRadii radii= new CornerRadii(2);
        BorderStroke borderStroke = new BorderStroke(Color.SADDLEBROWN, BorderStrokeStyle.SOLID, radii, new BorderWidths(2));
        Border border = new Border(borderStroke);
        notesEditor.setBorder(border);
        Paint paint = Color.LIGHTGRAY;
        Insets insets= Insets.EMPTY;
        BackgroundFill fill= new BackgroundFill(paint,radii,insets);
        Background background = new Background(fill);
        notesEditor.setBackground(background);
        notesEditor.setTranslateX(WIDTH/9);
        notesEditor.setTranslateY(HEIGHT*0.7);
        notesEditor.setText("");
        root.getChildren().add(notesEditor);

    }
    //-----------------------------------
	/* (non-Javadoc)
	 * @see javafx.application.Application#start(javafx.stage.Stage)
	 */
    @Override
    public void start(Stage primaryStage) throws Exception {
        root.setDepthTest(DepthTest.ENABLE);
        Scene scene = new Scene(root, WIDTH, HEIGHT, true);
        scene.setFill(Color.DIMGRAY);
        stage=primaryStage;
        primaryStage.setScene(scene);
        primaryStage.show();
        primaryStage.setTitle("Use piano to enter melody. Edit in text box: t=32nd note, s=16th, i=8th, q=quarter, h=half, w=whole; add '.' for dotted.");
        primaryStage.setOnCloseRequest( e -> {System.exit(0);});
        Group piano = makePiano();
        piano.setRotationAxis(Rotate.X_AXIS);
        piano.setRotate(-10);
        makePlayButton();
        makeComposeButton();
        makeNotesEditor();
        double scale=2.0;
        piano.setScaleX(scale);
        piano.setScaleY(scale);
        piano.setScaleZ(scale);
        piano.setTranslateX(300);
        piano.setTranslateY(125);
        root.getChildren().add(piano);
        handleKeyEvents(scene);
        animate();
    }
}

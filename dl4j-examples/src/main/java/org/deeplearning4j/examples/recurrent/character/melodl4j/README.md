# Learning to compose music from MIDI files using an LSTM neural network in deeplearning4j

#### by [Don Smith](mailto:ThinkerFeeler@gmail.com)

 MELODL4J is a java package in dl4j-examples for extracting melodies from MIDI, feeding them to a LSTM neural network, and composing music using deep learning.

It also has UI components, written in JavaFX, for visualizing gradients in real-time using 3D graphics.

You train the network on a set of sample melodies. The neural network learns to generate new melodies similar to the input melodies.

During network training, it plays melodies at the end of each training epoch. "Deep humming". At the end of training, it outputs all melodies to a file. There's a utility for listening to the generated melodies.

Listen: http://truthsite.org/music/melodies-like-pop.mp3, http://truthsite.org/music/melodies-like-bach.mp3, and http://truthsite.org/music/melodies-like-the-beatles.mp3 are examples of melodies created by the neural network (eight melodies each).

The neural network was trained on melodies extracted from midi files at http://truthsite.org/music/pop-midi.zip, http://truthsite.org/music/bach-midi.zip, and the Beatles MIDI files at http://colinraffel.com/projects/lmd/.

The LSTM neural network is based on the GravesLSTMCharModellingExample class by Alex Black in deeplearning4j.

The generated melodies are monophonic, meaning that only one note plays at a time: there are no chords or harmony.

No claim is made that this work is state of the art for computer music generation. But the generated melodies do sound pleasant and interesting. I pursued the project mostly to familiarize myself with deeplearning4j and because it's cool. I hope the example is educational and fun for others, as well as a foundation for doing more complex musical composition.

There were 6921 training melodies (about 5MB uncompressed) for the pop music training, for which were 541,859 parameters in the LSTM network. For the bach training, there were 363 input melodies. Perhaps the network memorized some snippets.

To help judge the extent to which generated melodies mimic existing melodies in the input training set, I wrote a utility to find the longest approximate string match between two melody strings. Using that utility, I found a melody closest to the first melody of http://truthsite.org/music/melodies-like-bach.mp3. Both that first melody (snippet) and the Bach melody closest to it are in http://truthsite.org/music/closest-matches1.mp3. Judge for yourself whether the neural network is mimicing Bach. I think it's similar but not a copy.

See http://www.asimovinstitute.org/analyzing-deep-learning-tools-music/, and https://github.com/tensorflow/magenta for some previous work in music generation via deep learning. I found out recently that MELODL4J is similar to magenta: both extract monophonic melodies from MIDI -- and both have a class named NoteSequence! -- but MELODL4J was developed independently and uses different techniques.
* * *

### Overview of methodology, simplifications, and tricks

1.  <tt>Midi2MelodyStrings.java</tt> parses MIDI files and outputs melodies in symbolic form.
2.  Given a Midi file, the program outputs one or more strings representing the melodies appearing in that midi file.
3.  If you point <tt>Midi2MelodyStrings.java</tt> at a directory, it processes all midi files under that directory. All extracted melodies are appended to a single output file.
4.  Each melody string is a sequence of characters representing either notes (e.g., middle C), rests, or durations. Each note is followed by a duration. Each rest is also followed by a duration.
5.  There is a utility class, <tt>PlayMelodyStrings.java</tt>, that converts melody strings into MIDI and plays the melody on your computer's builtin synthesizer (tested on Windows).
6.  To limit the number of possible symbols, the program restricts pitches to a two-octave interval centered on Middle C. Notes outside those two octaves are moved up or down an octave until they're inside the interval.
7.  The program ignores gradations of volume (aka "velocity") of MIDI notes. It also ignores other effects such as pitch bending.
8.  For each MIDI file that it processes, and for each MIDI track appearing in that file, Midi2MelodyStrings outputs zero or more melodies for that track.
9.  The parser skips percussion tracks.
10.  To exclude uninteresting melodies (e.g., certain bass accompaniments or tracks with long silences) the program skips melodies that have too many consecutive identical notes, too small a variety of pitches, too much silence, or too long rests between notes.
11.  For polyphonic tracks, the program outputs (up to) two monophonic melodies from the track: (i) the top notes of the harmony and (ii) the bottom notes of the harmony.
12.  A monophonic track results in a single output melody.
13.  The sequence of pitches in a melody is represented by a sequence of intervals (delta pitches). Effectively, all melodies are transposed to a standard key.
14.  To handle different tempos, the program normalize all durations relative to the average duration of notes in the melody.
15.  The program quanticizes tempos into 32 possible durations: 1*d, 2*d, 3*d, .... 32*d, where d is 1/8th the duration of the average note. Assuming the average note is a quarter note, the smallest possible duration for notes and rests is typically a 1/32nd note.
16.  The longest duration for a note or a rest is four times the duration of the average note. Typically, this means that notes longer than a whole note are truncated to be a whole note in length.
17.  No attempt is made to learn the characteristics of different instruments (e.g. pianos versus violins).
18.  <tt>MelodyModelingExample.java</tt> is the neural network code that composes melodies. It's closely based on GravesLSTMCharModellingExample.java by Alex Black.
19.  At the end of each learning epoch, <tt>MelodyModelingExample.java</tt> plays 15 seconds of the last melody generated. As learning progresses you can hear the compositions getting better.
20.  Before exiting, <tt>MelodyModelingExample.java</tt> writes the generated melodies to a specified file (in reverse order, so that the highest quality melodies tend to be at the start of the file).
21.  The melody strings composed by the neural network sometimes have invalid syntax (especially at the beginning of training). For example, in a valid melody string, each pitch character is followed by a duration character. PlayMelodyStrings.java will ignore invalid characters in melody strings.
22.  [http://truthsite.org/music/composed-in-the-style-of-pop.txt](http://truthsite.org/music/composed-in-the-style-of-pop.txt) and [http://truthsite.org/music/composed-in-the-style-of-bach.txt](http://truthsite.org/music/composed-in-the-style-of-bach.txt) are sample melody strings generated by the neural network. Each line in the files is one melody. You can play the melodies with <tt>PlayMelodyStrings.java</tt> .
23.  You can download some MIDI files from [http://truthsite.org/music/bach-midi.zip](http://truthsite.org/music/bach-midi.zip), [http://truthsite.org/music/pop-midi.zip](http://truthsite.org/music/pop-midi.zip), from [http://musedata.org](http://musedata.org), or from the huge Lakh MIDI Dataset at [http://colinraffel.com/projects/lmd/](http://colinraffel.com/projects/lmd/).
24.  You can download symbolic music files from [http://truthsite.org/music/midi-melodies-bach.txt.gz](http://truthsite.org/music/midi-melodies-bach.txt.gz) and [http://truthsite.org/music/midi-melodies-pop.txt.gz](http://truthsite.org/music/midi-melodies-pop.txt.gz) .

* * *

### Possible directions for improvement

1.  Represent durations numerically, not symbolically. Currently, both pitches and durations are represented symbolically (as characters), not numerically. This makes sense for pitches probably, since the qualitative feel of a C followed by a G is quite different from the qualitative feel of a C followed by a G#. Likewise, following a C, a G is more similar to a E than to a G#; both E and G are notes in a C major chord. But for tempos, a 32d note is more similar to a 16th note than to a quarter note, etc.
2.  Explore different activation functions, numbers of layers, and network hyper-parameters.
3.  Run larger experiments on more MIDI files (e.g., from the Lakh MIDI Dataset at [http://colinraffel.com/projects/lmd](http://colinraffel.com/projects/lmd)).
4.  Handle polyphonic music: chords and harmony.
5.  Handle gradations of volume, as well as other effects such as vibrato and pitch bend.
6.  Learn different melodies for different instruments.
7.  Make an interactive app that lets you "prime" the neural network with a melody -- via a real or virtual (piano) keyboard -- similar to <tt>generationInitialization</tt> in <tt>GravesLSTMCharModellingExample</tt>.
8.  Enhance the MIDI parser and make a DataVec reader.
9.  Add "Attention" as in [Generating Long-Term Structure in Songs and Stories](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/).

* * *

### Resources

*   [Melodies composed by the neural network in the style of Bach (MP3) -- melodies-like-bach.mp3](melodies-like-bach.mp3)
*   [Melodies composed by the neural network in the style of pop music (MP3) -- melodies-like-pop.mp3](melodies-like-pop.mp3)
*   [The symbolic melody strings extracted from the Bach MIDI files](midi-melodies-bach.txt) (output of <tt>Midi2MelodyStrings.java</tt>)
*   [The symbolic melody strings extracted from the pop MIDI files](midi-melodies-pop.txt) (output of <tt>Midi2MelodyStrings.java</tt>)
*   [The symbolic melody strings composed by the neural network to mimic Bach](composed-in-the-style-of-bach.txt) (<tt>PlayMelodyStrings.java</tt> converts the melodies to music.)
*   [The symbolic melody strings composed by the neural network to mimic pop](composed-in-the-style-of-pop.txt) (<tt>PlayMelodyStrings.java</tt> converts the melodies to music.)

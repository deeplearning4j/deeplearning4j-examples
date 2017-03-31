
MELODL4J is a small java package in dl4j-examples for extracting melodies from MIDI, feeding them to a LSTM neural network, and composing music using deep learning.

You train the network on a set of sample melodies. The neural network learns to generate new melodies similar to the input melodies.

During network training, it plays melodies at the end of each training epoch. "Deep humming". At the end of training, it outputs all melodies to an output file.

There's a utility for listening to the generated melodies.

http://truthsite.org/music/melodies-like-bach.mp3 and http://truthsite.org/music/melodies-like-pop.mp3 are examples of pieces created by the neural network (eight pieces each), based
on training melodies extracted from midi files from http://truthsite.org/music/bach-midi.zip and http://truthsite.org/music/pop-midi.zip.

The LSTM neural network is based on the GravesLSTMCharModellingExample class by Alex Black in deeplearning4j.

The generated melodies are monophonic, meaning that only one note plays at a time: there are no chords or harmony.

No claim is made that this work is state of the art for computer music generation.  But the generated melodies do sound pleasant and interesting.  I pursued the project mostly to familiarize myself with deep learning and and deeplearning4j.  I hope the example is educational and fun for others, as well as a foundation for doing more complex musical composition.

See http://www.asimovinstitute.org/analyzing-deep-learning-tools-music/,
https://cs224d.stanford.edu/reports/allenh.pdf and https://github.com/tensorflow/magenta for some previous work in music generation via deep learning.

I found out recently that MELODL4J is similar to magenta: both extract monophonic melodies from MIDI,
but MELODL4J was developed independently.

=========================

Overview of methodology, simplifications, and tricks:

- Midi2MelodyStrings.java parses MIDI files and outputs melodies in symbolic form.
You can feed the melodies as input to a neural network that learns to compose melodies similar to the input melodies.
- Given a Midi file, the program outputs one or more strings representing the melodies appearing in that midi file.
- If you point Midi2MelodyStrings.java at a directory, it processes all midi files under that directory. All extracted melodies are appended to a single output file.
- Each melody string is a sequence of characters representing either notes (e.g., middle C), rests, or durations. Each note is followed by a duration. Each rest is also followed by a duration.
- There is a utility class, PlayMelodyStrings.java, that converts melody strings into MIDI and plays the melody on your computer's builtin synthesizer (tested on Windows).
- To limit the number of possible symbols,  the program restricts pitches to a two-octave interval centered on Middle C. Notes outside that those two octaves are moved up or down an octave until they're inside the interval.
- The program ignores gradations of volume (aka "velocity") of MIDI notes. It also ignores other effects such as pitch bending.
- For each MIDI file that it processes, and for each MIDI track appearing in that file, Midi2MelodyStrings outputs zero or more melodies for that track.
- The parser skips percussion tracks.
- To exclude uninteresting melodies (e.g., bass accompaniments or melodies with long silences) the program skips melodies that have too many consecutive identical notes, too small a variety of pitches, too much silence, or too long rests between notes.
- For polyphonic tracks, the program outputs (up to) two monophonic melodies from the track:  (i) the top notes of the harmony and (ii) the bottom notes of the harmony.
- A monophonic track results in a single output melody.
- The sequence of pitches in a melody is represented by a sequence of intervals (delta pitches). Effectively, all melodies are transposed to a standard key.
- To handle different tempos, the program normalize all durations relative to the average duration of notes in the melody.
- The program quanticizes tempos into 32 possible durations: 1*d, 2*d, 3*d, .... 32*d, where d is 1/8th the duration of the average note. Assuming the average note is a quarter note, the smallest possible duration for notes and rests is typically a 1/32nd note.
- The longest duration for a note or a rest is four times the duration of the average note. Typically, this means that notes longer than a whole note are truncated to be a whole note in length.
- No attempt is made to learn the characteristics of different instruments (e.g. pianos versus violins).
- MelodyModelingExample.java  is the neural network code that composes melodies.
- It's closely based on GravesLSTMCharModellingExample.java by Alex Black.
- At the end of each learning epoch, MelodyModelingExample.java plays 15 seconds of the last melody generated. As learning progresses you can hear the compositions getting better.
- Before exiting, MelodyModelingExample.java writes the generated melodies to a specified file (in reverse order, so that the highest quality melodies tend to be at the start of the file).
- The melody strings composed by the neural network sometimes have invalid syntax (especially at the beginning of training). For example, in a valid melody string, each pitch character is followed by a duration character.  PlayMelodyStrings.java will ignore invalid characters  in melody strings.
- http://truthsite.org/music/composed-in-the-style-of-bach.txt and http://truthsite.org/music/composed-in-the-style-of-pop.txt are sample melodies generated by the neural network.
 Each line in the files is one melody. You can play the melodies with PlayMelodyStrings.java .
- You can download some MIDI files from http://truthsite.org/music/bach-midi.zip, http://truthsite.org/music/pop-midi.zip,
from http://musedata.org , or from the huge Lakh MIDI Dataset at http://colinraffel.com/projects/lmd/ .
- You can download symbolic music files from http://truthsite.org/music/midi-melodies-bach.txt.gz and http://truthsite.org/music/midi-melodies-pop.txt.gz .
- http://truthsite.org/music has more info.

Possible directions of improvement

 a. Represent durations numerically, not symbolically. Currently, both pitches and durations are represented symbolically (as characters), not numerically.  This makes sense for pitches probably, since the qualitative feel of a C followed by a G is quite different from the qualitative feel of a C followed by a G#. Likewise, following a C, a G is more similar to a E than to a G#; both E and G are notes in a C major chord.  But for tempos, a 32d note is more similar to a 16th note than to a quarter, etc.

 b. Explore different activation functions, numbers of layers, and network hyper-parameters.

 c. Run experiments on larger MIDI file inputs (e.g., from the Lakh MIDI Dataset at http://colinraffel.com/projects/lmd)

 d. Handle polyphonic music: chords and harmony.

 e. Handle gradations of volume, as well as other effects such as vibrato and pitch bend.

 f. Learn different melodies for different instruments.

 g. Make an interactive app that lets you "prime" the neural network with a melody -- via a real or virtual keyboard -- similar to generationInitialization in GravesLSTMCharModellingExample.

 h. Enhance the MIDI parser and make a DataVec reader.

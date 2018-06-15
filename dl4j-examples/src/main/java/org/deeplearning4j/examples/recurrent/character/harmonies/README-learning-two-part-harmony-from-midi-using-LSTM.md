
## Composing Two-Party Harmonies from MIDI using LSTM Deep Learning

by Donald A. Smith (ThinkerFeeler@gmail.com, @DonaldAlan on github)

Deep Learning can be used to compose interesting-sounding, musical pieces in the style of a given composer.

Listen to the motifs and variations in the first several pieces at the following site.

        http://deepmusic.info/

Here I explain how to represent two-part harmonies extracted from MIDI files as character string, for learning with LSTM networks.

In earlier work

    https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/melodl4j

I composed melodies (without harmony) by extracting monophonic sequences of notes from MIDI tracks.  I then used Graves LSTM
learning in deeplearning4j to produce symbolic melody strings that mimicked the strings found in the sample MIDI files.

In this new work, I follow a similar path, but I model two-party harmonies, which requires a change to the way music is encoded as strings.
Even more than before, extracting music is nontrivial; I needed to use several configurable heuristics to obtain useful harmony strings.

Options allow you to control whether to include instrument numbers in the harmony strings. But as of yet, training with instruments has
not resulted in reasonable results, so I recommend against configuring the system to use instruments.

## How it works

A MIDI file often has more than one voice toning at a time, distributed across multiple tracks, channels, and instruments. Even within
a single track and channel, there may be polyphony.   Often there is one channel per track; a channel plays one instrument at a time.
Each Note has a channel, but there is a PROGRAM_CHANGE command to change the instrument assigned to a channel.

I ignore the volume (all non-zero volumes are treated the same), and I omit percussion notes.

For each track, channel, and instrument I generate a voice: a list of Notes, ordered by start time. From a Midi file, I extract one or
more voices. For MIDI files with complex instrumentation, there may be a dozen or more voices.

If two adjacent notes in a track overlap significantly (more than 25%), I treat them as parts of two separate voices.   But when
played with legato, notes can overlap slightly. So, if two notes overlap slightly (less than 25%), I shorten the first note, so they
don't overlap. This degrades the quality of the music but greatly increases the number of monophonic voices I can extract.
Other approaches are possible.

I merge voices of the same instrument that don't overlap.  This covers some of the case where two different channels or tracks
play different parts of a melody. But it does not allow a melody to switch between different instruments. That is a future enhancement.

I skip voices that are too short or that have too little variety of notes.

I order the list of voices by the average pitch of their notes. (Thanks to tom-adfund for suggesting this heuristic,
which he said would help learn Bach harmonies. It does seem to help.)

For each pair of voices, where the first voice is lower in pitch than the second voice, I make a two-part harmony.  Note that in
a piece with many instruments there may be many pairs of voices.  Only some of them involve the melody. I use heuristics to
filter out harmonies that are likely to be boring.

A two-party harmony is represented by a pair of lists, where each list contains Note objects. Each Note object says the pitch,
the instrument number, the start time, the end time, and the volume. The volume and (by default) the instrument are ignored for learning.
Because each of the two voices is monophonic, there are no overlaps between the notes within one voice: every note ends before
(or equal to) the starting time of the subsequent note. However there may be overlaps between different voices at a given time.

I delete sections of long silence or sustained notes in both voices.  I skip two-part harmonies where one part is short in cumulative
note time relative to that of the other part.  Again the limits are configurable.

To convert a two-part harmony (represented as a pair of lists of Notes) into a symbolic two-part harmony string, I sample at
increments of 1/20th of a second.   I maintain a time variable t and repeatedly increment it by 0.20 repeatedly. For each value of t,
I sample from each of the two voices.  I convert the note in each sample to a character:  space indicates silence;
'A' = two octaves below middle C; 'B' = a half note above that; ....; 'w' = two octaves above middle C. So the available characters
when not modeling instruments are the following:

   "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvw"

 In all, this covers four octaves.  If a note is outside that range I adjust it by one octave until it lies inside the range.
 See MidiHarmonyUtility.java.

When not using instruments, in a symbolic two-part harmony string the even characters (at indexes 0, 2, 4, 6, ...) represent the
first voice. The odd characters (at indexes 1, 3, 5, 7, ...) represent the second voice.

This process results in two-party harmony strings such as the following (from Mozart training, without instruments):

    ttttototttttvtltttqqqqqqqqqqqqqqqqolololololoa a a e e e f f f f e e e f f e e V V c c a a j j e e c c e e c c e e clclVlVlclclclclclclcococo h h h h h hJa a a m m t t t q q q u u mq qqqqqorrrrrrrrrrrrrrrvqtqtqtqtqtqtqtqtqtqtmtmtltltltflflhthteqeqeqeqaqXqXqaqaqYqXqXqhqhqhqhqhqhqeqeqeqeqeqeqeqeqeqcococococo l lclclclZmZmcmcmcmcmclclclelelelele e e gjgjgjgjhhhh h h h h h h h  VhVhV U U Vh hZhZhZhZhZhZ ZoZ ZhZhZhZhjjjjjhhhhffgegecgcgUhZhZhZhZhZhLeLeLeSeVeVeLeLeXeXe eZlZlZlZlXjXjVhVhVhVhCeCeCeCeLoLoL LcLcLcLcLcOc cg gj g g mQoQoQmJqJsJgJgJhJfJfJfJaJaJZOaNZNZNZJcJcJcJaCaCaCaCaOaCaBZBZBZBZBZBZBaBaBaBaCaCaCaGeGeGeEcEcEcEcEcEcEcEclcccccccchchchcaaaaaaZZZZ ZcaeJeJcJcJZJcJcJZJXJXJcJcJhOeOeOeOaOaOaOaOaOfOfOeOeOcOcOeOeOeOeOeOeOcJcJcOcOcOeOeOeOeJfJfJfJfJfJcJeJeJeJeJeJeJeJcJcJcJcJZJZJZJZJXLXLXLXBXBXBZBZBaCaCaLvLvAmAmLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJVJVJVJVJVJVjVEVEVEVEhEhEoBlBlBlBoBoB BsBqBqBoJoJoZmZmclclcl lLlLlLlLlBlBlBhBhEoErGhGhJoEoD C B L L Q Q Q QrErErE E EqEqEoEoEmEmElJlJlJlJlJmJmJ CsCsCmCmCmCvCvCvCvClllmmJlJlJlJlMjNjNhNhNhNhNhNhNhNhNhLgLgLgLeJhJqJqJoJoJoJoJlJlJlJlJlJlJlJlJlJlJlZlhlhlhlOmOmOoOmOaararTrTrSqSqQrOqSqQsQsSvSvVlVlVlVhVhQhQhQhJVJVJVJVJVJVOJQJQJ JlJlJlJlJlJmJmJmJmJmJmJmJmJlJlJlJlJlJlJlJlOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOlOlOlOlOlOlOlOmOmOmOmOmOmOmOmOmOmOiOiOlOlOlOlOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmOmSmSmSmSmSmSmSmSmSmSmSmVmVmVmVm

This representation of harmonies is quite different from that used in my previous melody-only work.  In that representation, each
note had a pitch character and a duration character.  In the new representation, durations are implicit in the sampling.

If you include the instrument, four characters are output for every time step: a character representing the first voice's pitch,
a character representing the first voice's instrument, a character representing the second voice's pitch, and a character
representing the second voice's instrument, in that order.  To avoid needing characters above ASCII 127, to limit the number
possible characters, and to avoid infrequently used instruments, if the Note uses an instrument above MIDI instrument 75 (Pan Flute),
I treat it as instrument 26 (Electric Guitar (clean)). Instrument characters go from from Ascii 33 to Ascii 108.

I wrote utilities to convert two-part harmony strings back to MIDI sequences, and from MIDI sequences to wav and mp3 files.
When converting from a symbolic two-part harmony string back to a MIDI sequences, adjacent notes of the same pitch are merged
into a single Midi Note.

If you assign MIDI instrument numbers to the public static variables instrument1Restriction and instrument2Restriction
in MidiHarmonyUtility.java, you can limit notes to particular instruments (such as for bass guitar and guitar).
If you assign instrument numbers to the variables instrument0 and instrument1 in PlayTwoPartHarmonies.java, you can force
the generated music to use the respective instruments.

I collected thousands of MIDI music files and divided them into different directories: for Bach, Handel, Beethoven, jazz,
the Beatles, Abba, etc. Extracted harmony string files were of size between 15 MB (Bach) to 85 MB (Beatles).

Since adjacent melodies are often similar (coming from the same piece), I randomly permute the training lines in the data file.
This seems to prevent wild oscillations of learning loss.

I feed the symbolic two-party harmony strings to Graves LSTM, train it, and let it generate output strings, which I then
convert back to MIDI and MP3 files.  Training time varied from about an hour to eight hours, on a GTX 1070 gpu.  Loss generally
fell from  about 1000 to between 50 and 100.

I do minor post-processing of generated music: just removing long periods of silence and very long sustained notes.

I did not do a lot of tuning of the LSTM network, but I did find out that decreasing the learning rate and increasing the
mini-batch size did seem to help it learn: the loss went down pretty monotonically, instead of oscillating wildly.

I found that three LSTM layers usually worked better than four. Often if I tried to train with four layers, learning got
stuck or oscillated wildly.

The sample MP3s at http://deepmusic.info/ involve only a minor amount of curating, except for the top group.

------

## Ideas for future enhancement

If musical scores were digitized and converted into symbolic form, that would provide a higher quality source of training data.

JFugue has a language for representing music as strings, including instruments, chords and other features.  See
http://www.jfugue.org/examples.html

GUIDO is another computer music notation. See  https://en.wikipedia.org/wiki/GUIDO_music_notation
and http://wiki.ccarh.org/wiki/Guido_Music_Notation

Further tuning of LSTM is worthwhile: more layers, different hyper parameters, perhaps learned via arbiter and transfer learning.

An advantage of the symbolic harmony representation without instruments is that it's very simple: the even-indexed
characters are one voice, and the odd-indexed characters are the second voice. LSTM seemed to have no trouble learning it.
When I tried adding instruments (so that the pitch of the first voice was followed by the instrument of the first voice,
then the pitch of the second voice, and finally the instrument of the second voice), learning seemed unable to model the
conventions:  instrument characters were misaligned in the generated compositions.

Since instruments change rarely, I could intersperse the harmony strings with occasional instrument characters (which must
be distinct from pitch characters).  But each voice can have a different instrument, and there are 128 distinct MIDI instruments,
though most are rarely used.  Adding instruments would make the representation much less standardized and probably harder for
the LSTM to model.

Adding dynamics is an obvious enhancement.  To add volume, we could add a volume character after each voice character, as I
did with instruments, but that would suffer from the same problems as our representation for instruments.

There are 128 standard MIDI instruments, so we would have to go beyond the printable ASCII characters to represent them all.

Using a larger vocabulary of (non-ASCII) characters is another approach. In that way, a character could encode the pitch and
the instrument, and possibly the volume.  That would make the search space much larger.

GravesLSTM lets you prime text generation with a prefix string.  It would be interesting to give the program a beginning fragment
of a two-part harmony and let it complete the piece.

Can we train a network to know how to end a piece?  The pieces I've made it composed so far all end abruptly and arbitrarily.

Another challenge: given one voice, find an accompanying voice.

Other ideas: Specialize learning on percussion tracks and accompanying melodies. I already tried making it learn a bass and
lead guitar (seeh ttp://deepmusic.info/ ). Learn rhythm guitar to accompany lead guitar.  A challenge is that it's often
hard to figure out from MIDI files what track is what, although the instrument number is a strong hint.

-----
## Details about how to run the system:

In short, you

   * Download a zipfile of MIDI files (e.g., http://truthsite.org/music/bach-midi.zip  http://deepmusic.info/bach-midi.zip)
   * Unzip into a directory containing midi files
   * Run MidiMusicExtractor on that directory (it will prompt you and put output files in your home directory in midi-learning/)
   * Run GravesLSTMForTwoPartHarmonies on the harmonies file output by MidiMusicExtractor (it will prompt you)
   * Then play the resulting samples with PlayTwoPartHarmonies. (You'll need to copy-and-paste a sample to a file.)

You can also run DeepHarmony.java and harmonize with a serialized network zip file, which it prompts you for.

MidiMusicExtractor.java prompts you for a path to a midi directory. It then produces, inside midi-learning/ in your
home directory, a subdirectory containing a harmonies file of extracted two-party harmony strings, a melodies.txt file,
and an analysis.txt file. For example, if the MIDI directory you choose has the name "BEATLES", your midi-learning/
directory will contain something like the following:

  MINGW64 ~/midi-learning/BEATLES
  total 5188
  20 analysis.txt  3708 harmonies-BEATLES.mp3  1392 harmonies-BEATLES.txt    60 melodies.txt

If you want the files to be written to a different directory, you can change the default.

PlayTwoPartHarmony.java prompts you for a file containing two-party harmony strings (such as "harmonies-BEATLES.txt),
renders the first line as MIDI and plays it on your speakers. It also outputs an mp3 file ("harmonies-BEATLES.mp3" above),
provided you've installed lame http://lame.sourceforge.net/ (on linux) ffmpeg (on Windows) https://www.ffmpeg.org/,
to convert WAV files to mp3 files.

PlayTwoPartHarmony.java lets you can configure the tempo and a transpose (pitch offset to adjust each note). For the
samples at http://deepmusic.info/, I did not adjust the tempo, but in a few cases I adjusted the transpose.

MidiHarmonyUtility.java converts MIDI sequence files to symbolic two-party harmony string files.

Midi2WavRenderer.java (modified from JFugue, which has an apache license), converts MIDI files to wav files. PlayMusic.java has methods
for converting wav files to MP3 using lame or ffmpeg.

GravesLSTMForTwoPartHarmonies.java prompts you for a harmony file (such as "harmonies-BEATLES.txt" above), trains an LSTM network,
and outputs composed symbolic two-party harmony strings to a file such as

    samples-layerCount_4-layerSize_200-tbpttLength_300-l2_0.0015-learningRate_0.05-updater_Adam-2018-06-03--09-55.txt

in the same directory of the harmony file you chose.   During learning, after every 20 or so mini-batches (configurable), it
writes samples to the sample file. Generally, the samples near the end will sound better, and you should copy each one
you like to a separate file and render it with PlayTwoPartyHarmony.java.

 Note: You can download MIDI files from http://deepmusic.info//music/bach-midi.zip , http://deepmusic.info/pop-midi.zip,
 http://deepmusic.info/classical-midi.zip, and the large collection at http://colinraffel.com/projects/lmd/ .

 Contact me if you have questions: ThinkerFeeler@gmail.com .


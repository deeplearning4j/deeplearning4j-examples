/* *****************************************************************************
 *
 *
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

package org.deeplearning4j.examples.wip.advanced.modelling.encoderdecoder;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

/**
 * <p>
 * This is a seq2seq encoder-decoder LSTM model made according to Google's paper
 * <a href="https://arxiv.org/abs/1506.05869">A Neural Conversational Model</a>.
 * </p>
 * <p>
 * The model tries to predict the next dialog line using the provided one. It
 * learns on the <a href=
 * "https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html">Cornell
 * Movie Dialogs corpus</a>. Unlike simple char RNNs this model is more
 * sophisticated and theoretically, given enough time and data, can deduce facts
 * from raw text. Your mileage may vary. This particular network architecture is
 * based on AdditionRNN but changed to be used with a huge amount of possible
 * tokens (10-40k) instead of just digits.
 * </p>
 * <p>
 * Use the get_data.sh script to download, extract and optimize the train data.
 * It's been only tested on Linux, it could work on OS X or even on Windows 10
 * in the Ubuntu shell.
 * </p>
 * <p>
 * Special tokens used:
 * </p>
 * <ul>
 * <li><code>&lt;unk&gt;</code> - replaces any word or other token that's not in
 * the dictionary (too rare to be included or completely unknown)</li>
 * <li><code>&lt;eos&gt;</code> - end of sentence, used only in the output to
 * stop the processing; the model input and output length is limited by the
 * ROW_SIZE constant.</li>
 * <li><code>&lt;go&gt;</code> - used only in the decoder input as the first
 * token before the model produced anything
 * </ul>
 * <p>
 * The architecture is like this:
 * <p>
 * <pre>
 * Input =&gt; Embedding Layer =&gt; Encoder =&gt; Decoder =&gt; Output (softmax)
 * </pre>
 * <p>
 * The encoder layer produces a so called "thought vector" that contains a
 * neurally-compressed representation of the input. Depending on that vector the
 * model produces different sentences even if they start with the same token.
 * There's one more input, connected directly to the decoder layer, it's used to
 * provide the previous token of the output. For the very first output token we
 * send a special <code>&gt;go&lt;</code> token there, on the next iteration we
 * use the token that the model produced the last time. On the training stage
 * everything is simple, we apriori know the desired output so the decoder input
 * would be the same token set prepended with the <code>&gt;go&lt;</code> token
 * and without the last <code>&lt;eos&gt;</code> token. Example:
 * </p>
 * <p>
 * Input: "how" "do" "you" "do" "?"<br>
 * Output: "I'm" "fine" "," "thanks" "!" "<code>&lt;eos&gt;</code>"<br>
 * Decoder: "<code>&lt;go&gt;</code>" "I'm" "fine" "," "thanks" "!"
 * </p>
 * <p>
 * Actually, the input is reversed as per <a href=
 * "https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sequence
 * to Sequence Learning with Neural Networks</a>, the most important words are
 * usually in the beginning of the phrase and they would get more weight if
 * supplied last (the model "forgets" tokens that were supplied "long ago", i.e.
 * they have lesser weight than the recent ones). The output and decoder input
 * sequence lengths are always equal. The input and output could be of any
 * length (less than {@link #ROW_SIZE}) so for purpose of batching we mask the
 * unused part of the row. The encoder and decoder layers work sequentially.
 * First the encoder creates the thought vector, that is the last activations of
 * the layer. Those activations are then duplicated for as many time steps as
 * there are elements in the output so that every output element can have its
 * own copy of the thought vector. Then the decoder starts working. It receives
 * two inputs, the thought vector made by the encoder and the token that it
 * _should have produced_ (but usually it outputs something else so we have our
 * loss metric and can compute gradients for the backward pass) on the previous
 * step (or <code>&lt;go&gt;</code> for the very first step). These two vectors are simply
 * concatenated by the merge vertex. The decoder's output goes to the softmax
 * layer and that's it.
 * </p>
 * <p>
 * The test phase is much more tricky. We don't know the decoder input because
 * we don't know the output yet (unlike in the train phase), it could be
 * anything. So we can't use methods like outputSingle() and have to do some
 * manual work. Actually, we can but it would require full restarts of the
 * entire process, it's super slow and ineffective.
 * </p>
 * <p>
 * First, we do a single feed forward pass for the input with a single decoder
 * element, <code>&lt;go&gt;</code>. We don't need the actual activations except
 * the "thought vector". It resides in the second merge vertex input (named
 * "dup"). So we get it and store for the entire response generation time. Then
 * we put the decoder input (<code>&lt;go&gt;</code> for the first iteration) and the thought
 * vector to the merge vertex inputs and feed it forward. The result goes to the
 * decoder layer, now with rnnTimeStep() method so that the internal layer state
 * is updated for the next iteration. The result is fed to the output softmax
 * layer and then we sample it randomly (not with argMax(), it tends to give a
 * lot of same tokens in a row). The resulting token is looked up in the
 * dictionary, printed to the {@link System#out} and then it goes to the next
 * iteration as the decoder input and so on until we get
 * <code>&lt;eos&gt;</code>.
 * </p>
 * <p>
 * To continue the training process from a specific batch number, enter it when
 * prompted; batch numbers are printed after each processed macrobatch. If
 * you've changed the minibatch size after the last launch, recalculate the
 * number accordingly, i.e. if you doubled the minibatch size, specify half of
 * the value and so on.
 * </p>
 */
public class EncoderDecoderLSTM {

    /**
     * Dictionary that maps words into numbers.
     */
    private final Map<String, Double> dict = new HashMap<>();

    /**
     * Reverse map of {@link #dict}.
     */
    private final Map<Double, String> revDict = new HashMap<>();

    /**
     * The contents of the corpus. This is a list of sentences (each word of the
     * sentence is denoted by a {@link java.lang.Double}).
     */
    private final List<List<Double>> corpus = new ArrayList<>();

    private static final int HIDDEN_LAYER_WIDTH = 512; // this is purely empirical, affects performance and VRAM requirement
    private static final int EMBEDDING_WIDTH = 128; // one-hot vectors will be embedded to more dense vectors with this width
    private static final String CORPUS_FILENAME = "movie_lines.txt"; // filename of data corpus to learn
    private static final String MODEL_FILENAME = "rnn_train.zip"; // filename of the model
    private static final String BACKUP_MODEL_FILENAME = "rnn_train.bak.zip"; // filename of the previous version of the model (backup)
    private static final int MINIBATCH_SIZE = 32;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5); // save the model with this period
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1); // test the model with this period
    private static final int MAX_DICT = 20000; // this number of most frequent words will be used, unknown words (that are not in the
    // dictionary) are replaced with <unk> token
    private static final int TBPTT_SIZE = 25;
    private static final double LEARNING_RATE = 1e-1;
    private static final int ROW_SIZE = 40; // maximum line length in tokens

    /**
     * The delay between invocations of {@link java.lang.System#gc()} in
     * milliseconds. If VRAM is being exhausted, reduce this value. Increase
     * this value to yield better performance.
     */
    private static final int GC_WINDOW = 2000;

    private static final int MACROBATCH_SIZE = 20; // see CorpusIterator

    /**
     * The computation graph model.
     */
    private ComputationGraph net;

    public static void main(String[] args) throws IOException {
        new EncoderDecoderLSTM().run();
    }

    private void run() throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);

        createDictionary();

        File networkFile = new File(toTempPath(MODEL_FILENAME));
        int offset = 0;
        if (networkFile.exists()) {
            System.out.println("Loading the existing network...");
            net = ComputationGraph.load(networkFile, true);
            System.out.print("Enter d to start dialog or a number to continue training from that minibatch: ");
            String input;
            try (Scanner scanner = new Scanner(System.in)) {
                input = scanner.nextLine();
                if (input.toLowerCase().equals("d")) {
                    startDialog(scanner);
                } else {
                    offset = Integer.parseInt(input);
                    test();
                }
            }
        } else {
            System.out.println("Creating a new network...");
            createComputationGraph();
        }
        System.out.println("Number of parameters: " + net.numParams());
        net.setListeners(new ScoreIterationListener(1));
        train(networkFile, offset);
    }

    /**
     * Configure and initialize the computation graph. This is done once in the
     * beginning to prepare the {@link #net} for training.
     */
    private void createComputationGraph() {
        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final GraphBuilder graphBuilder = builder.graphBuilder()
                .backpropType(BackpropType.Standard)
                .tBPTTBackwardLength(TBPTT_SIZE)
                .tBPTTForwardLength(TBPTT_SIZE)
                .addInputs("inputLine", "decoderInput")
                .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("embeddingEncoder",
                        new EmbeddingLayer.Builder()
                                .nIn(dict.size())
                                .nOut(EMBEDDING_WIDTH)
                                .build(),
                        "inputLine")
                .addLayer("encoder",
                        new LSTM.Builder()
                                .nIn(EMBEDDING_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .build(),
                        "embeddingEncoder")
                .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
                .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
                .addVertex("merge", new MergeVertex(), "decoderInput", "dup")
                .addLayer("decoder",
                        new LSTM.Builder()
                                .nIn(dict.size() + HIDDEN_LAYER_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .build(),
                        "merge")
                .addLayer("output",
                        new RnnOutputLayer.Builder()
                                .nIn(HIDDEN_LAYER_WIDTH)
                                .nOut(dict.size())
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "decoder")
                .setOutputs("output");

        net = new ComputationGraph(graphBuilder.build());
        net.init();
    }

    private void train(File networkFile, int offset) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        CorpusIterator logsIterator = new CorpusIterator(corpus, MINIBATCH_SIZE, MACROBATCH_SIZE, dict.size(), ROW_SIZE);
        for (int epoch = 1; epoch < 10000; ++epoch) {
            System.out.println("Epoch " + epoch);
            if (epoch == 1) {
                logsIterator.setCurrentBatch(offset);
            } else {
                logsIterator.reset();
            }
            int lastPerc = 0;
            while (logsIterator.hasNextMacrobatch()) {
                net.fit(logsIterator);
                logsIterator.nextMacroBatch();
                System.out.println("Batch = " + logsIterator.batch());
                int newPerc = (logsIterator.batch() * 100 / logsIterator.totalBatches());
                if (newPerc != lastPerc) {
                    System.out.println("Epoch complete: " + newPerc + "%");
                    lastPerc = newPerc;
                }
                if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
                    saveModel(networkFile);
                    lastSaveTime = System.currentTimeMillis();
                }
                if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
                    test();
                    lastTestTime = System.currentTimeMillis();
                }
            }
        }
    }

    @SuppressWarnings("InfiniteLoopStatement")
    private void startDialog(Scanner scanner) throws IOException {
        System.out.println("Dialog started.");
        while (true) {
            System.out.print("In> ");
            // input line is appended to conform to the corpus format
            String line = "1 +++$+++ u11 +++$+++ m0 +++$+++ WALTER +++$+++ " + scanner.nextLine() + "\n";
            CorpusProcessor dialogProcessor = new CorpusProcessor(new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), ROW_SIZE,
                    false) {
                @Override
                protected void processLine(String lastLine) {
                    List<String> words = new ArrayList<>();
                    tokenizeLine(lastLine, words, true);
                    final List<Double> wordIdxs = wordsToIndexes(words);
                    if (!wordIdxs.isEmpty()) {
                        System.out.print("Got words: ");
                        for (Double idx : wordIdxs) {
                            System.out.print(revDict.get(idx) + " ");
                        }
                        System.out.println();
                        System.out.print("Out> ");
                        output(wordIdxs, true);
                    }
                }
            };
            dialogProcessor.setDict(dict);
            dialogProcessor.start();
        }
    }

    private void saveModel(File networkFile) throws IOException {
        System.out.println("Saving the model...");
        File backup = new File(toTempPath(BACKUP_MODEL_FILENAME));
        if (networkFile.exists()) {
            if (backup.exists()) {
                backup.delete();
            }
            networkFile.renameTo(backup);
        }
        ModelSerializer.writeModel(net, networkFile, true);
        System.out.println("Done.");
    }

    private void test() {
        System.out.println("======================== TEST ========================");
        int selected = rnd.nextInt(corpus.size());
        List<Double> rowIn = new ArrayList<>(corpus.get(selected));
        System.out.print("In: ");
        for (Double idx : rowIn) {
            System.out.print(revDict.get(idx) + " ");
        }
        System.out.println();
        System.out.print("Out: ");
        output(rowIn, true);
        System.out.println("====================== TEST END ======================");
    }

    private void output(List<Double> rowIn, boolean printUnknowns) {
        net.rnnClearPreviousState();
        Collections.reverse(rowIn);
        INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), 1, 1, rowIn.size());
        double[] decodeArr = new double[dict.size()];
        decodeArr[2] = 1;
        INDArray decode = Nd4j.create(decodeArr, 1, dict.size(), 1);
        net.feedForward(new INDArray[] { in, decode }, false, false);
        //INDArray[] netOuts = net.output(false, false, new INDArray[]{in, decode});
        org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) net
                .getLayer("decoder");
        Layer output = net.getLayer("output");
        GraphVertex mergeVertex = net.getVertex("merge");
        INDArray thoughtVector = mergeVertex.getInputs()[1];
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
        for (int row = 0; row < ROW_SIZE; ++row) {
            mergeVertex.setInputs(decode, thoughtVector);
            INDArray merged = mergeVertex.doForward(false, mgr);
            INDArray activateDec = decoder.rnnTimeStep(merged, mgr);
            INDArray out = output.activate(activateDec, false, mgr);
            double d = rnd.nextDouble();
            double sum = 0.0;
            int idx = -1;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, 0);
                if (d <= sum) {
                    idx = s;
                    if (printUnknowns || s != 0) {
                        System.out.print(revDict.get((double) s) + " ");
                    }
                    break;
                }
            }
            if (idx == 1) {
                break;
            }
            double[] newDecodeArr = new double[dict.size()];
            newDecodeArr[idx] = 1;
            decode = Nd4j.create(newDecodeArr, 1, dict.size(), 1);
        }
        System.out.println();
    }

    private void createDictionary() throws IOException {
        double idx = 3.0;
        dict.put("<unk>", 0.0);
        revDict.put(0.0, "<unk>");
        dict.put("<eos>", 1.0);
        revDict.put(1.0, "<eos>");
        dict.put("<go>", 2.0);
        revDict.put(2.0, "<go>");
        String CHARS = "-\\/_&" + CorpusProcessor.SPECIALS;
        for (char c : CHARS.toCharArray()) {
            if (!dict.containsKey(String.valueOf(c))) {
                dict.put(String.valueOf(c), idx);
                revDict.put(idx, String.valueOf(c));
                ++idx;
            }
        }
        System.out.println("Building the dictionary...");
        CorpusProcessor corpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, true);
        corpusProcessor.start();
        Map<String, Double> freqs = corpusProcessor.getFreq();
        Map<Double, Set<String>> freqMap = new TreeMap<>(new Comparator<Double>() {

            @Override
            public int compare(Double o1, Double o2) {
                return (int) (o2 - o1);
            }
        }); // tokens of the same frequency fall under the same key, the order is reversed so the most frequent tokens go first
        for (Entry<String, Double> entry : freqs.entrySet()) {
            Set<String> set = freqMap.computeIfAbsent(entry.getValue(), k -> new TreeSet<>());
            // tokens of the same frequency would be sorted alphabetically
            set.add(entry.getKey());
        }
        int cnt = 0;
        // the tokens order is preserved for TreeSet
        Set<String> dictSet = new TreeSet<>(dict.keySet());
        // get most frequent tokens and put them to dictSet
        for (Entry<Double, Set<String>> entry : freqMap.entrySet()) {
            for (String val : entry.getValue()) {
                if (dictSet.add(val) && ++cnt >= MAX_DICT) {
                    break;
                }
            }
            if (cnt >= MAX_DICT) {
                break;
            }
        }
        // all of the above means that the dictionary with the same MAX_DICT constraint and made from the same source file will always be
        // the same, the tokens always correspond to the same number so we don't need to save/restore the dictionary
        System.out.println("Dictionary is ready, size is " + dictSet.size());
        // index the dictionary and build the reverse dictionary for lookups
        for (String word : dictSet) {
            if (!dict.containsKey(word)) {
                dict.put(word, idx);
                revDict.put(idx, word);
                ++idx;
            }
        }
        System.out.println("Total dictionary size is " + dict.size() + ". Processing the dataset...");
        corpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, false) {
            @Override
            protected void processLine(String lastLine) {
                List<String> words = new ArrayList<>();
                tokenizeLine(lastLine, words, true);
                corpus.add(wordsToIndexes(words));
            }
        };
        corpusProcessor.setDict(dict);
        corpusProcessor.start();
        System.out.println("Done. Corpus size is " + corpus.size());
    }

    private String toTempPath(String path) {
        return  "/tmp/" + path;
    }

}

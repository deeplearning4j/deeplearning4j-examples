/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
package org.deeplearning4j.modelimportexamples.tf.advanced.bert;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.provider.CollectionLabeledPairSentenceProvider;
import org.deeplearning4j.modelimportexamples.utilities.DownloaderUtility;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.transform.*;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.resources.Downloader;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;

import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.*;


/**
 * This example demonstrates how to run inference on a fine tuned BERT model in SameDiff, where the fine tuning happens outside in Tensorflow
 * <p>
 * Details on the fine tuning:
 * The pretrained BERT model is fine tuned on the Microsoft Research Paraphrase Corpus (MRPC) corpus to
 * classify sentence pairs as like or unlike. This fine tuned model is then frozen to give a protobuf(pb) file.
 * More details about how this model and frozen pb are generated can be found here:
 * https://github.com/KonduitAI/dl4j-dev-tools/tree/master/import-tests/model_zoo/bert
 * <p>
 * The model is evaluated on the test dataset.
 * As expected it has about 84% accuracy and 0.89 F1 score reflecting the training metrics during the fine tuning.
 * Example also demonstrates how to do inference on a single minibatch.
 * <p>
 * Similar to all tensorflow models the frozen pb is imported into a SameDiff graph. In the case of BERT the graph has to be
 * preprocessed (removing unneeded nodes etc) before inference can be carried out on it. More details with code below.
 * @deprecated "Note this example uses an older TF import API that will be going away. Please consider using the newer import api."
 */
@Deprecated()
public class BertInferenceExample {

    public static String bertModelPath;
    //This BERT model uses a FIXED (hardcoded) minibatch size, not dynamic as most models use
    public static final int MINI_BATCH_SIZE = 4;
    public static final int MAX_LENGTH = 128;

    public static void main(String[] args) throws Exception {

        File frozenBertPB = downloadBERTFineTunedMSPR();

        //replace iterator with placeholder for inputs
        Map<String, TFImportOverride> iterToPlaceholderOverride = overrideIteratorsToPlaceholders();
        //Don't need the "IteratorV2" node from the graph, hence filtering when importing
        TFOpImportFilter filterNodeIterV2 = filterNodeByName("IteratorV2");
        SameDiff sd = TFGraphMapper.importGraph(frozenBertPB, iterToPlaceholderOverride, filterNodeIterV2);

        //rename replaced placeholders with more appropriate names
        sd.renameVariable("IteratorGetNext", "tokenIdxs");
        sd.renameVariable("IteratorGetNext:1", "mask");
        sd.renameVariable("IteratorGetNext:4", "sentenceIdx");

        //remove hard coded dropouts for inference
        sd = removeHardCodedDropOutOps(sd);
        sd.setTrainingConfig(new TrainingConfig.Builder()
            .updater(new Sgd())
            .dataSetFeatureMapping("tokenIdxs", "sentenceIdx")
            .dataSetFeatureMaskMapping("mask")
            .dataSetLabelMapping("loss/Softmax").build());

        //Downloads test data and sets up the bert iterator correctly
        MultiDataSetIterator iterTest = getMSPRTestIterator();

        System.out.println("\nRunning inference on the test dataset. This might take a while ... depending on your hardware");
        //Evaluates model on the entire test dataset and prints evaluation stats
        EvaluationRecord evaluationRecord = sd
            .evaluate()
            .data(iterTest)
            .evaluate("loss/Softmax", 0, new Evaluation()) //0 specifies the label index - needed since this is a multidataset iterator, "loss/Softmax" is the output node of interest
            .exec();
        System.out.println(evaluationRecord.evaluation("loss/Softmax").stats());

        //Four sentence pairs to run inference on
        List<Pair<String, String>> sentencePairs = new ArrayList<>();
        sentencePairs.add(new Pair<>("The broader Standard & Poor's 500 Index <.SPX> was 0.46 points lower, or 0.05 percent, at 997.02.", "The technology-laced Nasdaq Composite Index .IXIC was up 7.42 points, or 0.45 percent, at 1,653.44."));
        sentencePairs.add(new Pair<>("Shares in BA were down 1.5 percent at 168 pence by 1420 GMT, off a low of 164p, in a slightly stronger overall London market.", "Shares in BA were down three percent at 165-1/4 pence by 0933 GMT, off a low of 164 pence, in a stronger market."));
        sentencePairs.add(new Pair<>("Last year, Comcast signed 1.5 million new digital cable subscribers.", "Comcast has about 21.3 million cable subscribers, many in the largest U.S. cities."));
        sentencePairs.add(new Pair<>("Revenue rose 3.9 percent, to $1.63 billion from $1.57 billion.", "The McLean, Virginia-based company said newspaper revenue increased 5 percent to $1.46 billion."));
        //Featurizes them
        BertIterator bertIter = (BertIterator) iterTest;
        Pair<INDArray[], INDArray[]> featurizedWithMasks = bertIter.featurizeSentencePairs(sentencePairs);
        INDArray[] features = featurizedWithMasks.getFirst();
        INDArray[] masks = featurizedWithMasks.getSecond();

        System.out.println("\nRunning inference on a single minibatch with sentence pairs as follows:");
        for (Pair<String,String> sentencePair: sentencePairs) {
            System.out.println("\t" + sentencePair.getFirst() + "\t" + sentencePair.getSecond());
        }
        //Runs inference
        INDArray output = sd.batchOutput()
            .input("tokenIdxs", features[0])
            .input("sentenceIdx", features[1])
            .input("mask", masks[0])
            .output("loss/Softmax")
            .outputSingle();
        System.out.println("\n" + output);

    }


    private static MultiDataSetIterator getMSPRTestIterator() throws Exception {
        List<String> sentencesL = new ArrayList<>();
        List<String> sentencesR = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        URL testDataURL = new URL("https://raw.githubusercontent.com/lanwuwei/SPM_toolkit/master/PWIM/data/msrp/test/msr_paraphrase_test.txt");
        String testFileName = "msr_paraphrase_test.txt";
        String fileMD5 = "b7e1ed816b22c76d51e0f4bd87768056";
        //retry download five times
        Downloader.download(testFileName, testDataURL, new File(bertModelPath, testFileName), fileMD5, 5);

        List<String> lines = FileUtils.readLines(new File(bertModelPath, testFileName), "utf-8");
        for (int i = 0; i < lines.size(); i++) {
            if (i == 0) continue; //skip header
            String line = lines.get(i);
            String[] columns = line.split("\t");
            //Quality #1 ID   #2 ID   #1 String   #2 String
            labels.add(columns[0]);
            sentencesL.add(columns[3]);
            sentencesR.add(columns[4]);
        }

        CollectionLabeledPairSentenceProvider labeledPairSentenceProvider = new CollectionLabeledPairSentenceProvider(sentencesL, sentencesR, labels, null);
        URL vocabURL = new URL("https://dl4jdata.blob.core.windows.net/testresources/uncased/uncased_L-12_H-768_A-12/vocab.txt");
        String vocabFileName = "vocab.txt";
        fileMD5 = "64800d5d8528ce344256daf115d4965e";
        Downloader.download(vocabFileName,vocabURL, new File(bertModelPath,vocabFileName),fileMD5,5);

        File wordPieceTokens = new File(bertModelPath, vocabFileName);

        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(wordPieceTokens, true, true, StandardCharsets.UTF_8);
        BertIterator b = BertIterator.builder()
            .tokenizer(t)
            .padMinibatches(true)
            .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, MAX_LENGTH)
            .minibatchSize(MINI_BATCH_SIZE)
            .sentencePairProvider(labeledPairSentenceProvider)
            .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
            .vocabMap(t.getVocab())
            .task(BertIterator.Task.SEQ_CLASSIFICATION)
            .prependToken("[CLS]")
            .appendToken("[SEP]")
            .build();

        return b;
    }

    private static File downloadBERTFineTunedMSPR() throws Exception {
        bertModelPath = DownloaderUtility.BERTEXAMPLE.Download(false);
        return new File(bertModelPath, "bert_mrpc_frozen.pb");
    }

    /**
     * These are op import overrides. We skip the IteratorGetNext node and instead create placeholders.
     */
    private static Map<String, TFImportOverride> overrideIteratorsToPlaceholders() {
        Map<String, TFImportOverride> m = new HashMap<>();
        m.put("IteratorGetNext", (inputs, controlDepInputs, nodeDef, initWith, attributesForNode, graph) -> {
            return Arrays.asList(
                initWith.placeHolder("IteratorGetNext", DataType.INT, MINI_BATCH_SIZE, MAX_LENGTH),
                initWith.placeHolder("IteratorGetNext:1", DataType.INT, MINI_BATCH_SIZE, MAX_LENGTH),
                initWith.placeHolder("IteratorGetNext:4", DataType.INT, MINI_BATCH_SIZE, MAX_LENGTH)
            );
        });
        return m;
    }

    private static TFOpImportFilter filterNodeByName(String nodeName) {
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> {
            return nodeName.equals(nodeDef.getName());
        };
        return filter;
    }

    /**
     * Modify the network to remove hard-coded dropout operations for inference.
     * Tensorflow/BERT's dropout is implemented as a set of discrete operations - random, mul, div, floor, etc.
     * We need to select all instances of this subgraph, and then remove them from the graph entirely.
     * The subgraph to select are defined with predicates and a sub graph processor that passes input to output is used to replace it
     */
    private static SameDiff removeHardCodedDropOutOps(SameDiff sd) {

        /* Note that in general there are two ways to define subgraphs (larger than 1 operation) for use in GraphTransformUtil
            (a) withInputSubgraph - the input must match this predicate, AND it is added to the subgraph (i.e., matched and is selected to be part of the subgraph)
            (b) withInputMatching - the input must match this predicate, BUT it is NOT added to the subgraph (i.e., must match only)
           In effect, this predicate will match the set of directly connected operations with the following structure:
              (.../dropout/div, .../dropout/Floor) -> (.../dropout/mul)
              (.../dropout/add) -> (.../dropout/Floor)
              (.../dropout/random_uniform) -> (.../dropout/add)
              (.../dropout/random_uniform/mul) -> (.../dropout/random_uniform)
              (.../dropout/random_uniform/RandomUniform, .../dropout/random_uniform/sub) -> (.../dropout/random_uniform/mul)

          Then, for all subgraphs that match this predicate, we will process them (in this case, simply replace the entire subgraph by passing the input to the output)
          NOTE: How do you work out the appropriate subgraph to replace? The simplest approach is to visualize the graph - either in TensorBoard or using SameDiff UI.
        */
        SubGraphPredicate p = SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/mul"))
            .withInputCount(2)
            .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/div")))
            .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/Floor"))
                .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/add"))
                    .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform"))
                        .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/mul"))
                            .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/RandomUniform")))
                            .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/sub")))
                        )
                    )
                )
            );
        /*
        Create the subgraph processor.
        The subgraph processor is applied to each subgraph - i.e., it defines what we should replace it with.
        It's a 2-step process:
        (1) The SubGraphProcessor is applied to define the replacement subgraph (add any new operations, and define the new outputs, etc).
            In this case, we aren't adding any new ops - so we'll just pass the "real" input (pre dropout activations) to the output.
            Note that the number of returned outputs must match the existing number of outputs (1 in this case).
            Immediately after SubgraphProcessor.processSubgraph returns, both the existing subgraph (to be replaced) and new subgraph (just added)
            exist in parallel.
        (2) The existing subgraph is then removed from the graph, leaving only the new subgraph (as defined in processSubgraph method)
            in its place.
         Note that the order of the outputs you return matters!
         If the original outputs are [A,B,C] and you return output variables [X,Y,Z], then anywhere "A" was used as input
         will now use "X"; similarly Y replaces B, and Z replaces C.
         */
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd, p, new SubGraphProcessor() {
            @Override
            public List<SDVariable> processSubgraph(SameDiff sd, SubGraph subGraph) {
                List<SDVariable> inputs = subGraph.inputs();
                SDVariable newOut = null;
                for (SDVariable v : inputs) {
                    if (v.getVarName().endsWith("/BiasAdd") || v.getVarName().endsWith("/Softmax") || v.getVarName().endsWith("/add_1") || v.getVarName().endsWith("/Tanh")) {
                        newOut = v;
                        break;
                    }
                }

                if (newOut != null) {
                    return Collections.singletonList(newOut);
                }

                throw new RuntimeException("No pre-dropout input variable found");
            }
        });

        return sd;
    }
}

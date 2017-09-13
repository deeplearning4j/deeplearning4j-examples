package org.deeplearning4j.examples.nlp.paragraphvectors.tools;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * This is primitive seeker for nearest labels.
 * It's used instead of basic wordsNearest method because for ParagraphVectors
 * only labels should be taken into account, not individual words
 *
 * @author raver119@gmail.com
 */
public class LabelSeeker {
    private List<String> labelsUsed;
    private InMemoryLookupTable<VocabWord> lookupTable;

    public LabelSeeker(List<String> labelsUsed, InMemoryLookupTable<VocabWord> lookupTable) {
        if (labelsUsed.isEmpty()) throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors");
        this.lookupTable = lookupTable;
        this.labelsUsed = labelsUsed;
    }

    /**
     * This method accepts vector, that represents any document,
     * and returns distances between this document, and previously trained categories
     * @return
     */
    public List<Pair<String, Double>> getScores(INDArray vector) {
        List<Pair<String, Double>> result = new ArrayList<>();
        for (String label: labelsUsed) {
            INDArray vecLabel = lookupTable.vector(label);
            if (vecLabel == null) throw new IllegalStateException("Label '"+ label+"' has no known vector!");

            double sim = Transforms.cosineSim(vector, vecLabel);
            result.add(new Pair<String, Double>(label, sim));
        }
        return result;
    }
}

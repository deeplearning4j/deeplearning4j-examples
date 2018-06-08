package org.datavec.inputsplit;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.Iterator;
import java.util.Random;

/**
 * {@link org.datavec.api.split.BaseInputSplit} and its implementation provides a
 * {@link org.datavec.api.split.BaseInputSplit#sample(PathFilter, double...)} method that is very useful for generating
 * several {@link org.datavec.api.split.InputSplit}s from the main split.
 *
 * This can be used for dividing your dataset into several subsets. For example, into training, validation and testing.
 *
 * The {@link PathFilter} is useful for filtering the main split before generating the input splits array.
 * The second argument is a list of weights, which indicate a percentage of each input split.
 *
 * The samples are divided in the following way -> totalSamples * (weight1/totalWeightSum, weight2/totalWeightSum, ...,
 * weightN/totalWeightSum)
 *
 * {@link PathFilter} has two default implementations,
 * {@link org.datavec.api.io.filters.RandomPathFilter} that simple randomizes the order of paths in an array.
 * and
 * {@link org.datavec.api.io.filters.BalancedPathFilter} that randomizes the order of paths in an array and removes
 * paths randomly to have the same number of paths for each label. Further interlaces the paths on output based on
 * their labels, to obtain easily optimal batches for training.
 *
 * Their usages are shown here.
 */
public class _4_SamplingBaseInputSplitExample {
    public static void main(String[] args) throws Exception{
        FileSplit fileSplit = new FileSplit(new ClassPathResource("inputsplit/files").getFile());

        //Sampling with a RandomPathFilter
        InputSplit[] inputSplits1 = fileSplit.sample(
            new RandomPathFilter(new Random(123), null),
            10, 10, 10, 10, 10);

        System.out.println(String.format(("Random filtered splits -> Total(%d) = Splits of (%s)"), fileSplit.length(),
            String.join(" + ", () -> new InputSplitLengthIterator(inputSplits1))));

        //Sampling with a BalancedPathFilter
        InputSplit[] inputSplits2 = fileSplit.sample(
            new BalancedPathFilter(new Random(123), null, new ParentPathLabelGenerator()),
            10, 10, 10, 10, 10);

        System.out.println(String.format(("Balanced Splits are: %s"),
            String.join(" + ", () -> new InputSplitLengthIterator(inputSplits2))));
    }

    private static class InputSplitLengthIterator implements Iterator<CharSequence> {

        InputSplit[] inputSplits;
        int i;

        public InputSplitLengthIterator(InputSplit[] inputSplits) {
            this.inputSplits = inputSplits;
            this.i = 0;
        }

        @Override
        public boolean hasNext() {
            return i < inputSplits.length;
        }

        @Override
        public CharSequence next() {
            return String.valueOf(inputSplits[i++].length());
        }
    }
}

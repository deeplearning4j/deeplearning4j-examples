package org.deeplearning4j.examples.recurrent.seq2seq;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Created by susaneraly on 3/27/16.
 */

public class CustomSequenceIterator implements MultiDataSetIterator {

    private Random randnumG;
    private int currentBatch;
    private final int seed;
    private final int batchSize;
    private final int totalBatches;
    private final int numdigits;
    private final int encoderSeqLength;
    private final int decoderSeqLength;
    private final int outputSeqLength;
    private final int timestep;

    private static final int SEQ_VECTOR_DIM = 12;

    public CustomSequenceIterator (int seed, int batchSize, int totalBatches, int numdigits, int timestep) {

        this.seed = seed;
        this.randnumG = new Random(seed);

        this.batchSize = batchSize;
        this.totalBatches = totalBatches;

        this.numdigits = numdigits;
        this.timestep = timestep;

        this.encoderSeqLength = numdigits * 2 + 1;
        this.decoderSeqLength = numdigits + 2; // (numdigits + 1)max the sum can be + 1 (GO/END marker of seq)
        this.outputSeqLength = numdigits + 2; // (numdigits + 1)max the sum can be + 1 (GO/END marker of seq)

        this.currentBatch = 0;
    }
    public MultiDataSet generateTest(int testSize) {
        MultiDataSet testData = next(testSize);
        INDArray seq1 = testData.getFeatures(0);
        INDArray seq2 = testData.getFeatures(1);
        INDArray seq3 = testData.getLabels(0);
        System.out.println("Printing TEST........");
        System.out.println("Seq1........");
        System.out.println(seq1);
        System.out.println("Seq2........");
        System.out.println(seq2);
        System.out.println("Seq3........");
        System.out.println(seq3);
        //Handle processing of test results
        return testData;
    }
    @Override
    public MultiDataSet next(int sampleSize) {
        /* PLEASE NOTE:
            I don't check for repeats from pair to pair with the generator
            This is to be fixed later. This however should not affect score and nans.
         */
        //Initialize everything with zeros - will eventually fill with one hot vectors
        INDArray encoderSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, encoderSeqLength );
        INDArray decoderSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, decoderSeqLength );
        INDArray outputSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, outputSeqLength );

        //Since these are fixed length sequences of timestep
        //Masks are not required at the input sequence which is padded to length timestep
        //Masks are not required at the output for <GO> and <EOS>
        INDArray encoderMask = Nd4j.ones(sampleSize, encoderSeqLength);
        INDArray decoderMask = Nd4j.ones(sampleSize, decoderSeqLength);
        INDArray outputMask = Nd4j.ones(sampleSize, outputSeqLength);

        /* ========================================================================== */
        for (int iSample = 0; iSample < sampleSize; iSample++) {
            //Generate two random numbers with numdigits
            int num1 = randnumG.nextInt((int)Math.pow(10,numdigits));
            int num2 = randnumG.nextInt((int)Math.pow(10,numdigits));
            int sum = num1 + num2;
            /*
            Encoder sequence:
            Eg. with numdigits=4, num1=123, num2=90
                123 + 90 is encoded as "   09+321"
                Converted to a string to a fixed size given by 2*numdigits + 1 (for operator)
                then reversed and then masked
                Reversing input gives significant gain: <HTTP link>
                Each character is transformed to a 12 dimensional one hot vector
                    (index 0-9 for corresponding digits, 10 for "+", 11 for " ")
            */
            int spaceFill = (encoderSeqLength) - (num1 + "+" + num2).length();
            int iPos = 0;
            //Fill in spaces, as necessary
            while (spaceFill > 0) {
                //spaces encoded at index 12
                encoderSeq.putScalar(new int[] {iSample,11,iPos},1);
                iPos++;
                spaceFill--;
            }

            //Fill in the digits in num2 backwards
            String num2Str = String.valueOf(num2);
            for(int i = num2Str.length()-1; i >= 0; i--){
                int onehot = Character.getNumericValue(num2Str.charAt(i));
                encoderSeq.putScalar(new int[] {iSample,onehot,iPos},1);
                iPos++;
            }
            //Fill in operator in this case "+", encoded at index 11
            encoderSeq.putScalar(new int [] {iSample,10,iPos},1);
            iPos++;
            //Fill in the digits in num1 backwards
            String num1Str = String.valueOf(num1);
            for(int i = num1Str.length()-1; i >= 0; i--){
                int onehot = Character.getNumericValue(num1Str.charAt(i));
                encoderSeq.putScalar(new int[] {iSample,onehot,iPos},1);
                iPos++;
            }
            //Mask input for rest of the time series
            //while (iPos < timestep) {
            //    encoderMask.putScalar(new []{iSample,iPos},1);
            //    iPos++;
           // }
            /*
            Decoder and Output sequences:
            Following example above -
            Decoder sequence, 213 becomes "<GO>213  "
            Output sequence, 213 becomes "213  <END>"
            The starting char represented by <GO> will be masked
            The ending char represented by <END> will also be masked
            */
            //Fill in the digits from the sum
            iPos = 0;
            char [] sumCharArr = String.valueOf(num1+num2).toCharArray();
            for(char c : sumCharArr) {
                int digit = Character.getNumericValue(c);
                outputSeq.putScalar(new int [] {iSample,digit,iPos},1);
                decoderSeq.putScalar(new int [] {iSample,digit,iPos+1},1);
                iPos++;
            }
            //Fill in spaces, as necessary
            while (iPos < numdigits + 1) {
                //spaces encoded at index 12
                outputSeq.putScalar(new int [] {iSample,11,iPos}, 1);
                decoderSeq.putScalar(new int [] {iSample,11,iPos+1}, 1);
                iPos++;
            }
            //Fill in mask for <END> in output sequence
            //outputMask.putScalar(new int [] {iSample,outputSeqLength-1}, 1);
            //Fill in mask for <GO> in decode sequence
            //decoderMask.putScalar(new int[] {iSample,0}, 1);
        }
        /* ========================================================================== */
        INDArray[] inputs = new INDArray[]{encoderSeq, decoderSeq};
        INDArray[] inputMasks = new INDArray[]{encoderMask, decoderMask};
        INDArray[] labels = new INDArray[]{outputSeq};
        INDArray[] labelMasks = new INDArray[]{outputMask};
        currentBatch++;
        return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
    }

    @Override
    public void reset() {
        currentBatch = 0;
        randnumG = new Random(seed);
    }

    @Override
    public boolean hasNext() {
        //This generates numbers on the fly
        return currentBatch < totalBatches;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

    }
 }


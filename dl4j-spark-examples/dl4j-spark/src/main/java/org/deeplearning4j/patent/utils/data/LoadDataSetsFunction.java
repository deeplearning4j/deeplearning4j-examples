package org.deeplearning4j.patent.utils.data;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.hadoop.conf.Configuration;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.StaticWord2Vec;
import org.deeplearning4j.patent.utils.WordVectorProvider;
import org.nd4j.api.loader.Source;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.AbstractStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.DataInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * This is a DataSetLoader that loads the integer format of data written in DownloadParsePatents.PatentsToIndexFilesFunction
 *
 * @author Alex Black
 */
public class LoadDataSetsFunction implements DataSetLoader {
    private static Configuration conf = new Configuration();
    private String wordVectorsPath;
    private int numClasses;
    private int vectorSize;

    public LoadDataSetsFunction(String wordVectorsPath, int numClasses, int vectorSize) {
        this.wordVectorsPath = wordVectorsPath;
        this.numClasses = numClasses;
        this.vectorSize = vectorSize;
    }

    @Override
    public DataSet load(Source source) throws IOException {
        WordVectors wv = WordVectorProvider.getWordVectors(conf, wordVectorsPath);

        IntArrayList content = new IntArrayList();
        List<int[]> examples = new ArrayList<>();
        IntArrayList labels = new IntArrayList();
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE;
        try(DataInputStream dis = new DataInputStream(source.getInputStream())){
            while(dis.available() > 0){
                int i = dis.readInt();
                if(i < 0){
                    //Label
                    int[] example = content.toIntArray();
                    examples.add(example);
                    labels.add(i);
                    content.clear();
                    maxLength = Math.max(maxLength, example.length);
                    minLength = Math.min(minLength, example.length);
                } else {
                    content.add(i);
                }
            }
        }

        boolean needsFM = maxLength != minLength;
        INDArray f = Nd4j.create(new int[]{examples.size(), vectorSize, maxLength},'f');
        INDArray l = Nd4j.create(examples.size(), numClasses);
        INDArray fm = (needsFM ? Nd4j.create(examples.size(), maxLength) : null);

        AbstractStorage<Integer> storage;
        try {
            StaticWord2Vec sw2v = (StaticWord2Vec) wv;
            Field field = StaticWord2Vec.class.getDeclaredField("storage");
            field.setAccessible(true);
            storage = (AbstractStorage<Integer>)field.get(sw2v);
        } catch (Throwable t){
            throw new RuntimeException(t);
        }

        for( int i=0; i<examples.size(); i++ ){
            int[] idxs = examples.get(i);

            for( int j=0; j<idxs.length; j++ ){
                INDArray w = Transforms.unitVec(storage.get(idxs[j]).dup());
                f.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)).assign(w);
            }

            l.putScalar(i, Math.abs(labels.getInt(i)), 1.0);

            if(needsFM){
                fm.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, idxs.length)).assign(1.0);
            }
        }

        return new DataSet(f,l,fm,null);
    }
}

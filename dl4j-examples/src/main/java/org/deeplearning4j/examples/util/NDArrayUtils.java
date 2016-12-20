package org.deeplearning4j.examples.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.string.NDArrayStrings;

import java.io.IOException;
import java.util.List;


/**
 * So the intention of this utility is to make it easy to get structured data out of
 * the result buffers and data sets so that its easier to do things like build data
 * Java like structures, which would be important for enterprise use cases.
 *
 * Created by claytantor on 9/10/16.
 */
public class NDArrayUtils {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * Weekly typed N dimensional List. The user needs to be willing to implicity know
     * the depth of the list.
     *
     * This implementation uses Jackson, which I dont like because its a cheat. What would be thinner and
     * faster would be to re-implement the format method here:
     * https://github.com/deeplearning4j/nd4j/blob/2b3e8c60e42e3ed7f3ef72d45a599b74da31354c/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/string/NDArrayStrings.java#L66
     *
     *
     * @param source
     * @param precision
     * @return rows
     * @throws IOException
     */
    public static List makeRowsFromNDArray(INDArray source, int precision) throws IOException {
        String serializedData = new NDArrayStrings(", ",precision).format(source);
        List rows = (List)OBJECT_MAPPER.readValue(
                serializedData.getBytes(),List.class);
        return rows;
    }

    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }


}

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
package org.deeplearning4j.modelimportexamples.onnx;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.DrawBoundingBoxes;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.*;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.guava.primitives.Floats;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A port of the post processing for yolov4 to java from:
 * https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4
 *
 * @author Adam Gibson
 */
public class ImageProcessUtils {


    private static INDArray STRIDES = Nd4j.create(Nd4j.createBuffer(new long[]{8,16,32})).castTo(DataType.INT64);
    private static INDArray XYSCALE = Nd4j.create(Nd4j.createBuffer(new float[]{1.2f,1.1f,1.05f})).castTo(DataType.FLOAT);
    private static INDArray ANCHORS = Nd4j.create(Nd4j.createBuffer(new long[]{ 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401}));

    private static NDMath ndMath = new NDMath();
    private static NDNN ndNn = new NDNN();
    private static NDBitwise ndBitwise = new NDBitwise();
    private static NDBase ndBase = new NDBase();
    private static NDImage ndImage = new NDImage();

    /**
     * Return a pre processed image for yolov4
     * @param inputImage the input image file
     * @param targetHeight the target height of the image
     * @param targetWidth the target width of the image
     * @param inputHeight the input height of the image
     * @param inputWidth the input width of the image
     * @return the pre processed image
     * @throws IOException
     */
    public static INDArray yolov4PreProcess(File inputImage, int targetHeight, int targetWidth,int inputHeight,int inputWidth) throws IOException {
        long scale = Math.min(targetHeight / inputHeight,targetWidth / inputWidth);
        long newHeight = scale * targetHeight;
        long newWidth = scale * targetWidth;
        NativeImageLoader nativeImageLoader = new NativeImageLoader(newHeight,newWidth,3);
        INDArray array = nativeImageLoader.asMatrix(inputImage).castTo(DataType.FLOAT);
        INDArray padded = Nd4j.valueArrayOf(new long[]{inputHeight,inputWidth,3},128.0f);
        long dw = (inputWidth - newWidth) / 2;
        long dh = (inputHeight - newHeight) / 2;
        padded.put(new INDArrayIndex[]{NDArrayIndex.interval(dh,newHeight + dh),NDArrayIndex.interval(dw,newWidth + dw),NDArrayIndex.all()},array);
        padded.divi(255.0);
        return padded;
    }



    public static INDArray getAnchors(String anchorsPath,boolean tiny) throws IOException {
        List<String> anchorLines = FileUtils.readLines(new File(anchorsPath), Charset.defaultCharset());
        String[] anchors = anchorLines.get(0).split(",");
        return Nd4j.create(Floats.toArray(Arrays.stream(anchors).map(input -> Float.parseFloat(input))
                        .collect(Collectors.toList())))
                .reshape(3,3,2);
    }


    public static INDArray postProcessBoundingBox(List<INDArray> preBbox,INDArray anchors,INDArray strides,INDArray xyScale) {
        for(int i = 0; i < preBbox.size(); i++) {
            INDArray arr = preBbox.get(i);
            long[] shape = arr.shape();
            long outputSize = shape[1];
            INDArray convrawDxDy = arr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,2));
            INDArray convRawDwDh = arr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(2,4));
            INDArray[] xyGridOutput = ndMath.meshgrid(new INDArray[]{Nd4j.arange(outputSize),Nd4j.arange(outputSize)},true);
            INDArray xyGrid = Nd4j.expandDims(Nd4j.stack(-1,xyGridOutput),2);
            xyGrid = Nd4j.tile(Nd4j.expandDims(xyGrid,0),1,1,1,3,1).castTo(DataType.FLOAT);
            //((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
            INDArray predxY = ndNn.sigmoid(convrawDxDy).mul(xyScale.getNumber(i)).sub(xyGrid.add(0.5 * XYSCALE.getNumber(i).floatValue() - 1)).mul(strides.getNumber(i));
            INDArray predWh = ndMath.exp(convRawDwDh).mul(ANCHORS.getNumber(i));
            arr.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,4)},Nd4j.concat(-1,predxY,predWh));

        }

        List<INDArray> boundingBoxes = preBbox.stream().map(input -> input.reshape(-1,input.size(-1))).collect(Collectors.toList());
        INDArray ret = Nd4j.concat(0,boundingBoxes.toArray(new INDArray[boundingBoxes.size()]));
        return ret;
    }

    public static INDArray postProcessBoxes(INDArray predBbox,long[] originalImageShape,long inputSize,double scoreThreshold) {
        INDArray validScale = Nd4j.create(new double[]{0.0,Double.POSITIVE_INFINITY});
        INDArray predXyWh = predBbox.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(0,4)});
        INDArray predConf = predBbox.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4)});
        INDArray predProb = predBbox.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(5,predBbox.length())});
        INDArray predCoords = Nd4j.concat(-1,
                predXyWh.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(0,2)).sub(predXyWh.get(new INDArrayIndex[]{
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(2,predXyWh.length())
                })), predXyWh.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(0,2)).add(predXyWh.get(new INDArrayIndex[]{
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(2,predXyWh.length())
                })));

        long originalHeight = originalImageShape[0];
        long originalWidth = originalImageShape[1];
        long resizeRatio = Math.min(inputSize / originalWidth,inputSize / originalHeight);
        long dw = (inputSize - resizeRatio * originalWidth) / 2;
        long dh = (inputSize - resizeRatio * originalHeight) / 2;
        predCoords.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(0,2)},predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(0,2)).sub(dw).divi(resizeRatio));
        predCoords.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.interval(1,2)},predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(1,2)).sub(dh).divi(resizeRatio));
        predCoords = Nd4j.concat(-1,ndMath.max(predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(0,2)),Nd4j.zeros(2)),
                ndMath.min(predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(2,predCoords.length())),Nd4j.createFromArray(originalWidth - 1,originalHeight - 1)));
        INDArray firstClause = predCoords.get(NDArrayIndex.all(),NDArrayIndex.point(0)).gt(predCoords.get(NDArrayIndex.all(),NDArrayIndex.point(2)));
        INDArray secondClause = predCoords.get(NDArrayIndex.all(),NDArrayIndex.point(1)).gt(predCoords.get(NDArrayIndex.all(),NDArrayIndex.point(3)));
        INDArray invalidMask = ndBitwise.or(firstClause,secondClause);
        predCoords.putWhereWithMask(invalidMask,0.0);
        INDArray bboxesScale = ndMath.sqrt(predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(2,4)).sub(predCoords.get(NDArrayIndex.all(),NDArrayIndex.interval(0,2))).prod(-1));
        INDArray scaleMask = ndBitwise.and(bboxesScale.gt(validScale.getNumber(0)),bboxesScale.lt(validScale.getNumber(1)));
        INDArray classes = predProb.argMax(-1);
        INDArray scores = predConf.mul(predProb.get(NDArrayIndex.indices(Nd4j.arange(predCoords.length()).toLongVector()),NDArrayIndex.indices(classes.toLongVector())));
        INDArray scoreMask = scores.gt(scoreThreshold);
        INDArray mask = ndBitwise.and(scaleMask,scoreMask);
        INDArray retCoords = predCoords.get(mask);
        INDArray retScores = scores.get(mask);
        INDArray retClasses = scores.get(mask);
        return Nd4j.concat(-1,retCoords,retScores.get(NDArrayIndex.all(),NDArrayIndex.newAxis()),retClasses.get(NDArrayIndex.all(),NDArrayIndex.newAxis()));

    }


    public static INDArray[] drawBoundingBoxes(INDArray images,INDArray bboxes,INDArray colors) {
        return Nd4j.getExecutioner().exec(new DrawBoundingBoxes(images,bboxes,colors));
    }

    public static INDArray bboxesIou(INDArray boxes1,INDArray boxes2) {
        INDArray boxes1Area = boxes1.get(allButPoint(boxes1,2)).sub(boxes1.get(allButPoint(boxes1,0))).mul(boxes1.get(allButPoint(boxes1,3)).sub(boxes1.get(allButPoint(boxes1,1))));
        INDArray boxes2Area = boxes2.get(allButPoint(boxes1,2)).sub(boxes2.get(allButPoint(boxes1,0))).mul(boxes2.get(allButPoint(boxes2,3)).sub(boxes2.get(allButPoint(boxes2,1))));
        INDArray leftUp = ndMath.max(boxes1.get(allButInterval(boxes1,0,2)),boxes2.get(allButInterval(boxes1,0,2)));
        INDArray rightDown = ndMath.max(boxes1.get(allButInterval(boxes1,2, boxes1.length())),
                boxes2.get(allButInterval(boxes1,2,boxes2.length())));

        INDArray intersection = ndMath.max(rightDown.sub(leftUp),Nd4j.scalar(0.0));
        INDArray interArea = intersection.get(allButPoint(intersection,0)).mul(intersection.get(allButPoint(intersection,1)));
        INDArray unionArea = boxes1Area.add(boxes2Area).sub(interArea);
        INDArray ious = ndMath.max(interArea.castTo(DataType.FLOAT).div(unionArea),ndBase.minMax(DataType.FLOAT.toInt(),0));

        return ious;
    }



    private static INDArrayIndex[] allButInterval(INDArray input,long min,long max) {
        return concat(NDArrayIndex.nTimes(
                        NDArrayIndex.all(),
                        input.rank()),
                new INDArrayIndex[]{NDArrayIndex.interval(min,max)});
    }



    private static INDArrayIndex[] allButPoint(INDArray input,long point) {
        return concat(NDArrayIndex.nTimes(
                        NDArrayIndex.all(),
                        input.rank()),
                new INDArrayIndex[]{NDArrayIndex.point(point)});
    }

    private static INDArrayIndex[] concat(INDArrayIndex[] first,INDArrayIndex[] second) {
        INDArrayIndex[] ret = new INDArrayIndex[first.length + second.length];
        List<INDArrayIndex> retList = new ArrayList<>();
        for(INDArrayIndex firstIndex : first)
            retList.add(firstIndex);
        for(INDArrayIndex secondIndex : second)
            retList.add(secondIndex);
        return retList.toArray(ret);
    }

}

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package com.deeplearning4java.utils;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import android.net.Uri;
import android.os.Environment;
import android.util.Size;

import androidx.annotation.Nullable;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static java.io.File.separator;
import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.core.Core.flip;
import static org.opencv.core.Core.transpose;
import static org.opencv.imgproc.Imgproc.COLOR_YUV420p2BGR;

public class Utils {

    @Nullable
    public static String getPath(Context context, @Nullable Uri uri) {
        if (uri != null && "content".equalsIgnoreCase(uri.getScheme())) {
            String[] projection = { "_data" };
            Cursor cursor = null;

            try {
                cursor = context.getContentResolver().query(uri, projection, null, null, null);
                int column_index = cursor.getColumnIndexOrThrow("_data");
                if (cursor.moveToFirst()) {
                    return cursor.getString(column_index);
                }
                cursor.close();
            } catch (Exception e) {
                // Eat it
            }
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            return uri.getPath();
        }

        return null;
    }

    public static void checkIsDL4JModel(File file, CheckModelListener checkModelListener) {
        unzipFile(file, getCleanDir(Environment.getExternalStorageDirectory().getPath() + separator + "dl4jModel"), checkModelListener);
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public static File getNewFile(String dataPath) throws IOException {
        final File targetFile = new File(dataPath);
        if (targetFile.exists()) {
            targetFile.delete();
        }
        targetFile.createNewFile();
        return targetFile;
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public static File getNewDir(String dataPath) {
        final File targetDirectory = new File(dataPath);
        if (!targetDirectory.exists()) {
            targetDirectory.mkdirs();
        }
        return targetDirectory;
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public static File getCleanDir(String dataPath) {
        final File targetDirectory = new File(dataPath);
        if (targetDirectory.exists()) {
            deleteDir(targetDirectory);
        }
        targetDirectory.mkdirs();
        return targetDirectory;
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    private static void deleteDir(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory()) {
            for (File child : fileOrDirectory.listFiles()) {
                deleteDir(child);
            }
        }
        fileOrDirectory.delete();
    }

    private static void unzipFile(File zipFile, File targetDirectory, CheckModelListener checkModelListener) {
        try (ZipInputStream zis = new ZipInputStream(
                new BufferedInputStream(new FileInputStream(zipFile)))) {
            ZipEntry zipEntry;
            int count;
            byte[] buffer = new byte[8192];
            while ((zipEntry = zis.getNextEntry()) != null) {
                File file = new File(targetDirectory, zipEntry.getName());
                File dir = zipEntry.isDirectory() ? file : file.getParentFile();
                if (!dir.isDirectory() && !dir.mkdirs())
                    throw new FileNotFoundException("Failed to ensure directory: " +
                            dir.getAbsolutePath());
                if (zipEntry.isDirectory())
                    continue;
                try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
                    while ((count = zis.read(buffer)) != -1)
                        fileOutputStream.write(buffer, 0, count);
                }
            }
        } catch (Exception e1) {
            e1.printStackTrace();
            checkModelListener.checkDone(false);
        }
        checkModelListener.checkDone(isDL4JModel(targetDirectory));
    }

    private static boolean isDL4JModel(File dir){
        if (dir.exists()) {
            File[] files = dir.listFiles();
            for (File file : files) {
                if(!file.isDirectory() && file.getName().contains("coefficients.bin")){
                    return true;
                }
            }
        }
        return false;
    }

    public interface CheckModelListener {
        void checkDone(boolean isDL4JModel);
    }

    public static void imageToMat(final Image image, final Mat mat, byte[] data, byte[] rowData) {
        ByteBuffer buffer;
        int rowStride, pixelStride, width = image.getWidth(), height = image.getHeight(), offset = 0;
        Image.Plane[] planes = image.getPlanes();
        if (data == null || data.length != width * height)
            data = new byte[width * height * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        if (rowData == null || rowData.length != planes[0].getRowStride())
            rowData = new byte[planes[0].getRowStride()];
        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int
                    w = (i == 0) ? width : width / 2,
                    h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);
                    // Advance buffer the remainder of the row stride, unless on the last row.
                    // Otherwise, this will throw an IllegalArgumentException because the buffer
                    // doesn't include the last padding.
                    if (h - row != 1)
                        buffer.position(buffer.position() + rowStride - length);
                    offset += length;
                } else {
                    // On the last row only read the width of the image minus the pixel stride
                    // plus one. Otherwise, this will throw a BufferUnderflowException because the
                    // buffer doesn't include the last padding.
                    if (h - row == 1)
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    else
                        buffer.get(rowData, 0, rowStride);
                    for (int col = 0; col < w; col++)
                        data[offset++] = rowData[col * pixelStride];
                }
            }
        }
        mat.put(0, 0, data);
    }

    public static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
        final int MINIMUM_PREVIEW_SIZE = 320;
        final int minSize = Math.max(Math.min(width, height), MINIMUM_PREVIEW_SIZE);
        final Size desiredSize = new Size(width, height);

        // Collect the supported resolutions that are at least as big as the preview Surface
        boolean exactSizeFound = false;
        final List<Size> bigEnough = new ArrayList<>();
        final List<Size> tooSmall = new ArrayList<>();
        for (final Size option : choices) {
            if (option.equals(desiredSize)) {
                // Set the size but don't return yet so that remaining sizes will still be logged.
                exactSizeFound = true;
            }

            if (option.getHeight() >= minSize && option.getWidth() >= minSize) {
                bigEnough.add(option);
            } else {
                tooSmall.add(option);
            }
        }

        if (exactSizeFound) {
            return desiredSize;
        }

        // Pick the smallest of those, assuming we found any
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else {
            return new Size(choices[0].getWidth(), (int) (choices[0].getWidth() * 1.6));
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(final Size lhs, final Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    //helper class to return the largest value in the output array
    public static double arrayMaximum(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double cur : arr)
            max = Math.max(max, cur);
        return max;
    }

    // helper class to find the index (and therefore numerical value) of the largest confidence score
    public static int getIndexOfLargestValue(double[] array) {
        if (array == null || array.length == 0) return 0;
        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }

    public static Mat makeMatFromImage(Image image, int width, int height){
        Mat yuv420Mat = new Mat(height * 3 / 2, width, CvType.CV_8UC1);
        imageToMat(image, yuv420Mat, null, null);
        image.close();
        Mat bgrMat = new Mat(height, width, CvType.CV_8UC3);
        Imgproc.cvtColor(yuv420Mat, bgrMat, COLOR_YUV420p2BGR);
        Mat transMat = new Mat();
        Mat flipMat = new Mat();
        transpose(bgrMat, transMat);
        flip(transMat, flipMat, 1); //transpose+flip(1)=CW
        return flipMat;
    }

    public static Bitmap makeBitmapFromMat(Mat mat){
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        matToBitmap(mat, bitmap);
        return bitmap;
    }
}

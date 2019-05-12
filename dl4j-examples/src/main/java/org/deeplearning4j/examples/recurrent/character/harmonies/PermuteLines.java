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


package org.deeplearning4j.examples.recurrent.character.harmonies;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author Don Smith
 * <p>
 * It's better to sort the text files of training data that are input to Deep Learning. This utility can be used for that purpose.
 */
public class PermuteLines {

    private static void permute(File inFile, File outFile) throws IOException {
        final BufferedReader reader = new BufferedReader(new FileReader(inFile));
        final PrintWriter writer = new PrintWriter(outFile);
        final List<String> lines = new ArrayList<>();
        while (true) {
            String line = reader.readLine();
            if (line == null) {
                break;
            }
            lines.add(line);
        }
        reader.close();
        permute(lines);
        for (String line : lines) {
            writer.println(line);
        }
        writer.close();
    }

    private static void permute(List<String> lines) {
        Random random = new Random();
        int size = lines.size();
        for (int i = 0; i < size; i++) {
            int otherIndex = random.nextInt(size);
            String other = lines.get(otherIndex);
            lines.set(otherIndex, lines.get(i));
            lines.set(i, other);
        }
    }

    public static void main(String[] args) {
        args = new String[]{"d:/tmp/harmonies/abba-harmonies.txt", "d:/tmp/harmonies/abba-harmonies-permuted.txt"};
        File inFile = new File(args[0]);
        File outFile = new File(args[1]);
        if (outFile.exists()) {
            System.err.println(inFile.getAbsolutePath() + " exists");
            System.exit(1);
        }
        try {
            permute(inFile, outFile);
        } catch (Throwable thr) {
            thr.printStackTrace();
        }
    }
}

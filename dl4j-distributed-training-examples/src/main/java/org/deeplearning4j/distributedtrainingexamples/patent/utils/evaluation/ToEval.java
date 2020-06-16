/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.distributedtrainingexamples.patent.utils.evaluation;

import java.io.File;

public class ToEval {
    private final File f;
    private final int count;
    private final long durationSoFar;

    public ToEval(File f, int count, long  durationSoFar){
        this.f = f;
        this.count = count;
        this.durationSoFar = durationSoFar;
    }

    public File getFile(){
        return f;
    }

    public int getCount(){
        return count;
    }

    public long getDurationSoFar(){
        return durationSoFar;
    }

}

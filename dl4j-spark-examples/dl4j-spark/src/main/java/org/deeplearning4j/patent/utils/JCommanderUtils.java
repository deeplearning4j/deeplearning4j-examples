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

package org.deeplearning4j.patent.utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;

public class JCommanderUtils {

    private JCommanderUtils(){ }

    public static void parseArgs(Object obj, String[] args){
        JCommander jcmdr = new JCommander(obj);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();  //User provides invalid input -> print the usage info
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }
    }
}

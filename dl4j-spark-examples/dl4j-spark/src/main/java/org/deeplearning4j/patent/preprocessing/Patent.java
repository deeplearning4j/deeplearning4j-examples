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

package org.deeplearning4j.patent.preprocessing;

/**
 * A simple class used for holding patent text (and classification - i.e., class label) for the patent example
 */
public class Patent {

    protected String id;
    protected String classificationUS;
    protected String allText;

    public void setClassificationUS(String s){
        this.classificationUS = s;
    }

    public void setAllText(String text){
        this.allText = text;
    }

    public String getAllText(){
        return allText;
    }

    public String getClassificationUS(){
        return classificationUS;
    }
}

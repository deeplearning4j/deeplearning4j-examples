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

package org.deeplearning4j.distributedtrainingexamples.patent.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Patent parser for old text format patents
 */
public class LegacyFormatPatentParser {

    public static final Logger log = LoggerFactory.getLogger(LegacyFormatPatentParser.class);

    protected static final String PATENT_START_TAG = "PATN"; // Start patent designation
    protected static final List<String> TAG_PARAGRAPHS = Arrays.asList("PAR", "PAL", "PAC", "PA0", "PA1", "PA2", "PA3", "PA4", "PA5");
    protected static final String TITLE = "TTL";
    protected static final String DESCRIPTION = "DETD"; // detailed description
    protected static final String CLAIM = "DCLM";
    protected static final String CLAIMS = "CLMS";
    protected static final String ABSTRACT = "ABST";
    protected static final String DRAWINGS = "DRWD";
    protected static final String ORIG_CLASSIFICATION = "OCL";

    protected static final String ALL_TAGS = "PATN|PAR|PAL|PAC|PA0|PA1|PA2|PA3|PA4|PA5|TTL|DETD|DCLM|CLMS|BSUM|ABST|DRWD";
    protected static final String ALL_TAGS_STARTSWITH_REGEX = "(PATN|PAR|PAL|PAC|PA0|PA1|PA2|PA3|PA4|PA5|TTL|DETD|DCLM|CLMS|BSUM|ABST|DRWD).*";

    public List<Patent> parsePatentZip(File file) throws IOException {
        log.info("Parsing file [{}]", file.getAbsolutePath());
        List<Patent> patents = new ArrayList<>();

        //All in zip format...
        try {
            ZipFile zf = new ZipFile(file);
            Enumeration<? extends ZipEntry> entries = zf.entries();
            while (entries.hasMoreElements()) {
                ZipEntry ze = entries.nextElement();
                List<String> currentPatent = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(zf.getInputStream(ze)))) {
                    String line;
                    while((line = br.readLine()) != null){
                        if(line.startsWith(PATENT_START_TAG)){
                            if(currentPatent.size() > 0){
                                Patent p = parseSingle(currentPatent);
                                if(p != null && p.getClassificationUS() != null){
                                    patents.add(p);
                                }
                            }
                            currentPatent.clear();
                        }
                        currentPatent.add(line);
                    }
                }
            }
        } catch (IOException e ){
            throw new RuntimeException(e);
        }

        return patents;
    }

    public enum State {
        NONE,
        TITLE,
        ABSTRACT,
        CLAIM,
        DESCRIPTION,
        DRAWINGS
    }

    public Patent parseSingle(List<String> lines){

        State currentState = null;
        List<String> titleList = new ArrayList<>();
        List<String> abstrList = new ArrayList<>();
        List<String> claimList = new ArrayList<>();
        List<String> descrList = new ArrayList<>();
        List<String> origClassification = new ArrayList<>();

        for (String line : lines) {
            if (line.startsWith(TITLE)) {
                currentState = State.TITLE;
            } else if (line.startsWith(ABSTRACT)) {
                currentState = State.ABSTRACT;
            } else if (line.startsWith(CLAIM) || line.startsWith(CLAIMS)) {
                currentState = State.CLAIM;
            } else if (line.startsWith(DESCRIPTION)) {
                currentState = State.DESCRIPTION;
            } else if (line.startsWith(DRAWINGS)) {
                currentState = State.DRAWINGS;
            } else if (line.startsWith(ORIG_CLASSIFICATION)){
                currentState = State.NONE;
                origClassification.add(line);
                continue;
            } else if (!(line.matches("\\s+.*") || lineStartsWithAny(line, TAG_PARAGRAPHS))) {
                //Line does NOT start with whitespace = and it's not a paragraph tag
                //Therefore: assume new section we don't care about
                currentState = State.NONE;
            }

            if (currentState != null) {
                if (line.matches(ALL_TAGS_STARTSWITH_REGEX)) {
                    line = line.replaceFirst(ALL_TAGS, "");
                }
                switch (currentState) {
                    case TITLE:
                        titleList.add(line);
                        break;
                    case ABSTRACT:
                        abstrList.add(line);
                        break;
                    case CLAIM:
                        claimList.add(line);
                        break;
                    case DESCRIPTION:
                        descrList.add(line);
                        break;
                }
            }
        }

        Patent p = new Patent();
        String title = String.join(" ", titleList);
        String abstr = String.join(" ", abstrList);
        String claims = String.join(" ", claimList);
        String descr = String.join(" ", descrList);

        String txt = String.join(" ", title, abstr, claims, descr);
        txt = new TextPreprocess(txt).transform();
        p.setAllText(txt);

        //Original classification: legacy format doesn't seem to have a single "main" classifacion, unlike later formats
        //So we're taking the most common "OCL" - original classification - tag (by examiners)
        if(origClassification.size() == 0){
            return null;
        }
        Map<String,Integer> temp = new HashMap<>();
        String maxClassification = null;
        int maxCount = 0;
        for(String s : origClassification){
            int newCount;
            if(!temp.containsKey(s)){
                newCount = 1;
            } else {
                newCount = temp.get(s) + 1;
            }
            temp.put(s, newCount);

            if(newCount > maxCount){
                maxCount = newCount;
                maxClassification = s;
            }
        }
        String usClassification = maxClassification.substring(5);   //"OCL  " - don't trim leading spaces (important for correct parsing later)
        p.setClassificationUS(usClassification);

        return p;
    }

    private boolean lineStartsWithAny(String line, List<String> start){
        for(String s : start){
            if(line.startsWith(s)){
                return true;
            }
        }
        return false;
    }

}

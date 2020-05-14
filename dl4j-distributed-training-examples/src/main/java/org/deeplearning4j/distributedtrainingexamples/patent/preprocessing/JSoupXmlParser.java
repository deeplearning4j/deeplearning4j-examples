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

package org.deeplearning4j.distributedtrainingexamples.patent.preprocessing;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * JSoup based XML parser for patent example - used for parsing XML-format patents
 */
public class JSoupXmlParser {
    private static final Logger log = LoggerFactory.getLogger(JSoupXmlParser.class);

    public List<Patent> parsePatentZip(File file) {
        log.info("Parsing file [{}]", file.getAbsolutePath());
        List<Patent> result = new ArrayList<>();

        try {
            ZipFile zf = new ZipFile(file);
            Enumeration<? extends ZipEntry> entries = zf.entries();

            while (entries.hasMoreElements()) {
                ZipEntry ze = entries.nextElement();
                try (InputStream bis = new BufferedInputStream(zf.getInputStream(ze)); BufferedReader br = new BufferedReader(new InputStreamReader(bis))) {
                    //Keep reading until we get to end of patent: "</us-patent-grant>"
                    List<String> lines = new ArrayList<>();
                    String line;
                    while ((line = br.readLine()) != null) {
                        lines.add(line);
                        if (line.startsWith("</us-patent-grant>")) {
                            //Parse and add to output
                            String str = String.join("\n", lines);
                            Patent p = parseSingle(str);
                            if(p != null && p.getClassificationUS() != null){
                                result.add(p);
                            }
                            lines.clear();
                        }
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    /**
     * Parse a SINGLE patent
     *
     * @param str Single patent
     */
    public Patent parseSingle(String str) {
        Document document = Jsoup.parse(str, "", Parser.xmlParser());

        Patent p = new Patent();
        String title;
        String abstr;
        String claims;
        String descr;

        Elements patent = document.select("us-patent-grant");
        if (patent.size() > 0) {
            // Assume patent is modern format: USPTO XML format 4.x - Jan 2005 onwards

            //Get main classification
            Elements e = patent.select("classification-national");
            if (e == null || e.size() == 0) {
                log.warn("Skipping patent - no classification-national");
                return null;
            }

            Element e2 = e.first();
            Elements mainClassification = e2.select("main-classification");
            if (mainClassification == null || mainClassification.size() == 0) {
                log.warn("Skipping patent {} in document - no main classification", patent);
                return null;
            }
            String main = e2.select("main-classification").outerHtml().replaceAll("\n", "")
                    .replaceAll("<main-classification>", "").replaceAll("</main-classification>", "")
                    .replaceFirst(" ", ""); //Replace first space - not significant, always present. But SECOND space is important
            p.setClassificationUS(main);

            //Text to include: title, abstract, claims, description
            title = patent.select("invention-title").text();
            abstr = patent.select("abstract").text();
            claims = patent.select("claims").text();
            descr = patent.select("description").text();
        } else {
            patent = document.select("PATDOC");
            if (patent.size() > 0) {
                // Assume patent is older XML format: USPTO XML format 2.5 - Jan 2002 to Dec 2004
                title = patent.select("B540").first().text();
                abstr = patent.select("SDOAB").text();
                claims = patent.select("SDOCL").text();
                descr = String.join(" ", patent.select("DETDESC").text(), patent.select("DRWDESC").text());
            } else {
                return null;
            }
        }

        String txt = String.join(" ", title, abstr, claims, descr);
        txt = new TextPreprocess(txt).transform();
        p.setAllText(txt);
        return p;
    }
}

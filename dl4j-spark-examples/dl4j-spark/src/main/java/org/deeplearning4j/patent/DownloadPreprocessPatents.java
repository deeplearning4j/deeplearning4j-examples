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

package org.deeplearning4j.patent;

import com.beust.jcommander.Parameter;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.patent.preprocessing.JSoupXmlParser;
import org.deeplearning4j.patent.preprocessing.LegacyFormatPatentParser;
import org.deeplearning4j.patent.preprocessing.Patent;
import org.deeplearning4j.patent.preprocessing.PatentLabelGenerator;
import org.deeplearning4j.patent.utils.JCommanderUtils;
import org.deeplearning4j.patent.utils.WordVectorProvider;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.nd4j.linalg.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.util.*;

/**
 * This Spark script downloads and preprocesses the patent files. Is is part 1 of 2 of this example.
 *
 * See PatentExampleReadme.md for more details.
 *
 * NOTE: It assumes you have an Azure storage account - it should be possible to adapt this for other cloud providers.
 * Run using the provided shell scripts.
 *
 * @author Alex Black
 */
public class DownloadPreprocessPatents {
    private static final Logger log = LoggerFactory.getLogger(DownloadPreprocessPatents.class);

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JPatentExamplePreprocessing";

    @Parameter(names = {"--azureStorageAcct"}, description = "Name of the Azure storage account to use for storage", required = true)
    private String azureStorageAcct;

    @Parameter(names = {"--azureContainerZips"}, description = "Name of the container in the specified storage account for the zip files", required = true)
    private String azureContainerZips;

    @Parameter(names = {"--azureContainerPreproc"}, description = "Name of the container in the specified storage account for the serialized training DataSet files")
    private String azureContainerPreproc = "patentPreprocData";

    @Parameter(names = {"--downloadZips"}, description = "Whether the zips should be downloaded", arity = 1)
    private boolean downloadZips = true;

    @Parameter(names = {"--dlFirstYear"}, description = "First year to downloaded")
    private int dlFirstYear = 1976;

    @Parameter(names = {"--dlLastYear"}, description = "Last year to download")
    private int dlLastYear = 2018;

    @Parameter(names = {"--generateTrainingData"}, description = "Whether the training data should be generated", arity = 1)
    private boolean generateTrainingData = true;

    @Parameter(names = {"--firstTestYear"}, description = "First test year. For example, if set to 2017, then all of 2017 and 2018 patent data will be the test set", arity = 1)
    private int firstTestYear = 2018;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size for generated DataSets")
    private int minibatch = 32;

    @Parameter(names = {"--maxSequenceLength"}, description = "Maximum number of words in the sequences for generated DataSets")
    private int maxSequenceLength = 1000;

    @Parameter(names = {"--wordVectorsPath"})
    private String wordVectorsPath = "wasbs://resources@dl4jdata.blob.core.windows.net/wordvectors/GoogleNews-vectors-negative300.bin.gz";


    public static void main(String[] args) throws Exception {
        new DownloadPreprocessPatents().entryPoint(args);
    }

    /**
     * JCommander entry point
     */
    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        //Azure storage account naming rules: https://blogs.msdn.microsoft.com/jmstall/2014/06/12/azure-storage-naming-rules/
        //The default exceptions aren't helpful, we'll validate this here
        if(!azureStorageAcct.matches("^[a-z0-9]+$") || azureStorageAcct.length() < 3 || azureStorageAcct.length() > 24){
            throw new IllegalStateException("Invalid storage account name: must be alphanumeric, lowercase, " +
                    "3 to 24 characters. Got option azureStorageAcct=\"" + azureStorageAcct + "\"");
        }
        if(!azureContainerZips.matches("^[a-z0-9-]+$") || azureContainerZips.length() < 3 || azureContainerZips.length() > 63){
            throw new IllegalStateException("Invalid Azure container name: must be alphanumeric or dash, lowercase, " +
                    "3 to 63 characters. Got option azureContainerZips=\"" + azureContainerZips + "\"");
        }
        if(!azureContainerPreproc.matches("^[a-z0-9-]+$") || azureContainerPreproc.length() < 3 || azureContainerPreproc.length() > 63){
            throw new IllegalStateException("Invalid Azure container name: must be alphanumeric or dash, lowercase, " +
                    "3 to 63 characters. Got option azureContainerPreproc=\"" + azureContainerPreproc + "\"");
        }


        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName(sparkAppName);
        JavaSparkContext sc = new JavaSparkContext();

        String baseDirZips = "wasbs://" + azureContainerZips + "@" + azureStorageAcct + ".blob.core.windows.net/";

        if (downloadZips) {
            log.info("*** Starting download of patent data ***");
            log.info("--- WARNING: This requires approximately 92GB of storage for the patents in zip format! ---"); //Approx 464GB uncompressed - but we process directly from zipped format
            long start = System.currentTimeMillis();
            List<String> downloadUrls = new ArrayList<>();
            String format = "https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/";
            for (int i = dlFirstYear; i <= dlLastYear; i++) {
                String url = format + i + "/";
                downloadUrls.addAll(getZipUrlsFromPage(url));
            }

            //Make sure the container exists; create it if not
            Path p = new Path(URI.create(baseDirZips));
            FileSystem fs = FileSystem.get(URI.create(baseDirZips), new Configuration());
            if (!fs.exists(p)) {
                fs.create(p);
            }

            JavaRDD<String> rdd = sc.parallelize(downloadUrls);
            rdd.foreach(new DownloadToAzureFn(baseDirZips, true));
            long end = System.currentTimeMillis();
            log.info("*** Completed download of patent data in {} sec ***", (end-start)/1000);
        }

        if (generateTrainingData) {
            log.info("*** Starting preprocessing of patent data ***");
            log.info("--- WARNING: This requires approximately 20 GB of storage for the processed data! ---");
            long start = System.currentTimeMillis();
            String dirName = "seqLength" + maxSequenceLength + "_mb" + minibatch;
            String containerRoot = "wasbs://" + azureContainerPreproc + "@" + azureStorageAcct + ".blob.core.windows.net/";
            String baseOutPath = containerRoot + dirName;
            String baseOutputPathTrain = baseOutPath + "/train/";
            String baseOutputPathTest = baseOutPath + "/test/";

            //Seems that we need the container to exist before we can create anything in a subdirectory...
            Configuration config = new Configuration();
            FileSystem fs;
            try {
                fs = FileSystem.get(URI.create(containerRoot), config);
            } catch (Throwable t){
                throw new RuntimeException("Error getting filesystem for container root: " + containerRoot, t);
            }
            if (!fs.exists(new Path(containerRoot))) {
                fs.create(new Path(containerRoot));
            }


            JavaRDD<String> pathsTrain = sc.parallelize(listPaths(1976, firstTestYear - 1, sc, baseDirZips));
            JavaRDD<long[]> rddTrain = pathsTrain.map(new PatentsToIndexFilesFunction(
                    maxSequenceLength,
                    wordVectorsPath,
                    minibatch,
                    PatentLabelGenerator.classLabelToIndex(),
                    baseOutputPathTrain,
                    10
            ));

            log.info("Finished generating training datasets");

            long[] resultTrain = rddTrain.reduce(new ReduceArrayFunction());

            JavaRDD<String> pathsTest = sc.parallelize(listPaths(firstTestYear, 2018, sc, baseDirZips));
            JavaRDD<long[]> rddTest = pathsTest.map(new PatentsToIndexFilesFunction(
                    maxSequenceLength,
                    wordVectorsPath,
                    minibatch,
                    PatentLabelGenerator.classLabelToIndex(),
                    baseOutputPathTest,
                    10
            ));
            log.info("Finished generating testing datasets");

            long[] resultTest = rddTest.reduce(new ReduceArrayFunction());

            log.info("Train - Total datasets: {}", resultTrain[0]);
            log.info("Train - Total examples: {}", resultTrain[1]);
            log.info("Train - Total bytes (DataSets on disk): {}", resultTrain[2]);
            log.info("Train - Total skipped due to length: {}", resultTrain[3]);
            log.info("Test - Total datasets: {}", resultTest[0]);
            log.info("Test - Total examples: {}", resultTest[1]);
            log.info("Test - Total bytes (DataSets on disk): {}", resultTest[2]);
            log.info("Test - Total skipped due to length: {}", resultTest[3]);

            long end = System.currentTimeMillis();
            log.info("*** Completed preprocessing of patent data in {} sec ***", (end-start)/1000);
        }

        log.info("-- Completed All Preprocessing Steps --");
    }

    /**
     * Get a list of all URLs in a page for zip files
     */
    public static List<String> getZipUrlsFromPage(String url) {
        List<String> out = new ArrayList<>();
        try {
            Document doc = Jsoup.connect(url).get();
            Elements links = doc.select("a[href]");

            for (Element e : links) {
                String s = e.attr("href");
                if (s.endsWith(".zip")) {
                    if (s.startsWith("http")) {
                        //Absolute link
                        out.add(s);
                    } else {
                        //Relative link
                        out.add(e.baseUri() + s);
                    }
                }
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return out;
    }

    public static List<String> listPaths(JavaSparkContext sc, String path, boolean recursive) throws IOException {
        if (path.endsWith(".blob.core.windows.net/") || path.endsWith(".blob.core.windows.net")) {
            //Azure library bug: seems that we get an infinite loop if we try to list paths on the
            // root directory, for some versions of the Azure Hadoop library - deadlocks on fileIter.hasNext()
            throw new IllegalStateException("Cannot list paths from root directory due to Azure library bug");
        }

        List<String> paths = new ArrayList<>();
        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(URI.create(path), config);
        RemoteIterator fileIter = hdfs.listFiles(new Path(path), recursive);

        while (fileIter.hasNext()) {
            String filePath = ((LocatedFileStatus) fileIter.next()).getPath().toString();
            paths.add(filePath);
        }

        return paths;
    }

    public List<String> listPaths(int firstYearInclusive, int lastYearInclusive, JavaSparkContext sc, String baseDirZips) throws IOException {
        List<String> paths = new ArrayList<>();
        for (int i = firstYearInclusive; i <= lastYearInclusive; i++) {
            List<String> pathsYear = listPaths(sc, baseDirZips + i, true);
            for (String s : pathsYear) {
                if (s.endsWith(".zip")) {
                    paths.add(s);
                }
            }
        }
        log.info("Number of paths: {}", paths.size());
        return paths;
    }

    public static class DownloadToAzureFn implements VoidFunction<String> {
        private static final Configuration conf = new Configuration();

        private String rootDir;
        private boolean skipExisting;

        public DownloadToAzureFn(String rootDir, boolean skipExisting) {
            this.rootDir = rootDir;
            this.skipExisting = skipExisting;
        }

        @Override
        public void call(String url) throws Exception {
            int idx = url.indexOf("fulltext/");
            String year = url.substring(idx + 9, idx + 9 + 4);
            String filename = FilenameUtils.getName(url);

            URI outUri = new URI(rootDir + year + "/" + filename);

            FileSystem fs = FileSystem.get(outUri, conf);
            URL source = new URL(url);

            Path p = new Path(outUri);
            if (fs.exists(p)) {
                if (skipExisting) {
                    long length = fs.getFileStatus(p).getLen();
                    if (length < 1024 * 1024) {
                        //Assume it must be corrupt somehow if it's < 1MB
                        log.info("Re-downloading file of length {}: {} - {}", length, url, outUri);
                        fs.delete(p, false);
                    } else {
                        log.info("Skipping existing file: {} - {}", url, outUri);
                        return;
                    }
                }
                fs.delete(p, false);
            }

            try (FSDataOutputStream out = fs.create(p); InputStream is = new BufferedInputStream(source.openStream())) {
                IOUtils.copy(is, out);
            } catch (Throwable t) {
                throw new RuntimeException("Error downloading: " + url, t);
            }
            log.info("Downloaded: {} to {}", url, outUri);
        }
    }

    private static class ReduceArrayFunction implements Function2<long[], long[], long[]> {
        @Override
        public long[] call(long[] p1, long[] p2) {
            for (int i = 0; i < p1.length; i++) {
                p1[i] += p2[i];
            }
            return p1;
        }
    }


    public static class PatentsToIndexFilesFunction implements Function<String, long[]> {
        private static Configuration config = new Configuration();
        private final int maxSequenceLength;
        private String wordVectorsPath;
        private int minibatchSize;
        private Map<String, Integer> teir2WordVectorsLabelIdx;
        private String baseOutputPath;

        private final int minTokens;
        private static WordVectors wordVectors;

        public PatentsToIndexFilesFunction(int maxSequenceLength, String wordVectorsPath, int minibatchSize, Map<String, Integer> teir2WordVectorsLabelIdx, String baseOutputPath, int minTokens) {
            this.maxSequenceLength = maxSequenceLength;
            this.wordVectorsPath = wordVectorsPath;
            this.minibatchSize = minibatchSize;
            this.teir2WordVectorsLabelIdx = teir2WordVectorsLabelIdx;
            this.baseOutputPath = baseOutputPath;
            this.minTokens = minTokens;
        }

        @Override
        public long[] call(String s) throws Exception {

            URI u = new URI(s);
            FileSystem fs = FileSystem.get(u, config);
            String name = FilenameUtils.getBaseName(s);
            File temp = Files.createTempFile(name, ".zip").toFile();
            temp.deleteOnExit();

            FileSystem file = null;

            long dataSetCount = 0;
            long exampleCount = 0;
            long totalSize = 0;
            long countSkippedOnSize = 0;
            try {
                try (InputStream in = new BufferedInputStream(fs.open(new Path(u))); OutputStream os = new BufferedOutputStream(new FileOutputStream(temp))) {
                    IOUtils.copy(in, os);
                } catch (Throwable t) {
                    log.warn("Patent failed, skipping: {}", s, t);
                    return new long[]{0,0,0,0};
                }

                String[] split = s.split("/");
                int year = Integer.parseInt(split[split.length - 2]);
                List<Patent> patents;
                if (year <= 2001) {
                    patents = new LegacyFormatPatentParser().parsePatentZip(temp);
                } else {
                    patents = new JSoupXmlParser().parsePatentZip(temp);
                }

                log.info("Finished loading {} patents for path {}", patents.size(), s);

                int[] order = new int[patents.size()];
                for (int i = 0; i < order.length; i++) {
                    order[i] = i;
                }
                MathUtils.shuffleArray(order, new Random());

                List<int[]> toMerge = new ArrayList<>();
                IntArrayList labelToMerge = new IntArrayList();
                IntArrayList tempIntArrayList = new IntArrayList();
                TokenizerFactory tf = new DefaultTokenizerFactory();
                WordVectors wv = WordVectorProvider.getWordVectors(config, wordVectorsPath);
                for (int idx : order) {
                    Patent p = patents.get(idx);

                    String tier2 = null;
                    try {
                        tier2 = PatentLabelGenerator.tier2Label(p.getClassificationUS());
                    } catch (Throwable t){
                        log.warn("Skipping bad patent label: {}", p.getClassificationUS());
                        //Don't continue, in case we need to export on this one
                    }
                    if (tier2 != null && teir2WordVectorsLabelIdx.containsKey(tier2)) {
                        int labelIdx = teir2WordVectorsLabelIdx.get(tier2);
                        String text = p.getAllText();

                        List<String> tokens = tf.create(text).getTokens();
                        if(tokens.size() < minTokens){
                            countSkippedOnSize++;
                            continue;
                        }

                        tempIntArrayList.clear();
                        for(String token : tokens){
                            if(wv.hasWord(token)){
                                tempIntArrayList.add(wv.indexOf(token));
                            }
                            if(tempIntArrayList.size() >= maxSequenceLength){
                                break;
                            }
                        }

                        if(tempIntArrayList.size() < minTokens){
                            countSkippedOnSize++;
                            continue;
                        }

                        toMerge.add(tempIntArrayList.toIntArray());
                        labelToMerge.add(labelIdx);
                    }

                    if (toMerge.size() >= minibatchSize || (order[order.length - 1] == idx && toMerge.size() > 0)) {

                        String filename = "wordIndices_" + UUID.randomUUID().toString() + ".bin";
                        URI uri = new URI(this.baseOutputPath + (!this.baseOutputPath.endsWith("/") && !this.baseOutputPath.endsWith("\\") ? "/" : "") + filename);
                        if (file == null) {
                            file = FileSystem.get(uri, config);
                        }

                        long writtenBytes = 0;
                        try (FSDataOutputStream out = file.create(new Path(uri))) {
                            for( int i=0; i<toMerge.size(); i++ ){
                                int[] idxs = toMerge.get(i);
                                for( int write : idxs){
                                    out.writeInt(write);
                                    writtenBytes += 4;
                                }
                                out.writeInt(-labelToMerge.get(i));
                                exampleCount++;
                                writtenBytes += 4;
                            }
                        } catch (Throwable t) {
                            throw new RuntimeException("Error saving file: path \"" + uri + "\"", t);
                        }

                        totalSize += writtenBytes;
                        dataSetCount++;


                        toMerge.clear();
                        labelToMerge.clear();
                    }
                }
            } catch (Throwable t) {
                log.error("Error parsing: {}", s);
                throw t;
            } finally {
                temp.delete();
            }

            return new long[]{dataSetCount, exampleCount, totalSize, countSkippedOnSize};
        }
    }
}

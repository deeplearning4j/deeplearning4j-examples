package org.deeplearning4j.examples.cifar;

import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Loads CIFAR data over the network and stores the data files
 * in user's home directory.
 */
public class CifarDataLoader {

    private static final Logger log = LoggerFactory.getLogger(CifarDataLoader.class);
    private static final String USER_HOME = System.getProperty("user.home");

    private static final String CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    private static final String CIFAR_10_DIRECTORY = USER_HOME + "/cifar-10-batches-bin/";
    public static final List<String> CIFAR_10_TRAINING_DATA_FILE_NAMES = Arrays.asList(
        CIFAR_10_DIRECTORY + "data_batch_1.bin",
        CIFAR_10_DIRECTORY + "data_batch_2.bin",
        CIFAR_10_DIRECTORY + "data_batch_3.bin",
        CIFAR_10_DIRECTORY + "data_batch_4.bin",
        CIFAR_10_DIRECTORY + "data_batch_5.bin"
    );
    public static final String CIFAR_10_TEST_FILE_NAME = CIFAR_10_DIRECTORY + "test_batch.bin";

    public static final int IMAGE_SIZE = 3072;

    public static byte[] loadCifar10Data(boolean train) throws IOException {
        final Path dataDirectory = Paths.get(CIFAR_10_DIRECTORY);
        boolean found = true;
        if (Files.isDirectory(dataDirectory)) {
            final List<String> dataFiles = Files.list(dataDirectory).map(Path::toAbsolutePath).map(Path::toString).collect(Collectors.toList());
            if(!dataFiles.containsAll(CIFAR_10_TRAINING_DATA_FILE_NAMES) || !dataFiles.contains(CIFAR_10_TEST_FILE_NAME)) {
                found = false;
            }
        }
        if (!found) {
            log.info("Data is not present. Downloading");
            downloadAndUnCompressData(CIFAR_10_URL);
        }

        if(train) {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            for (String trainingDataFile : CIFAR_10_TRAINING_DATA_FILE_NAMES) {
                byteArrayOutputStream.write(Files.readAllBytes(Paths.get(trainingDataFile)));
            }
            return byteArrayOutputStream.toByteArray();
        } else {
            return Files.readAllBytes(Paths.get(CIFAR_10_TEST_FILE_NAME));
        }
    }

    private static void downloadAndUnCompressData(String dataUrlString) throws IOException {
        final URL dataUrl = new URL(dataUrlString);
        log.info("Downloading started");
        try(GzipCompressorInputStream gzipCompressorInputStream = new GzipCompressorInputStream(new BufferedInputStream(dataUrl.openStream()));
            ArchiveInputStream archiveInputStream = new TarArchiveInputStream(gzipCompressorInputStream)) {
            ArchiveEntry entry;
            while ((entry = archiveInputStream.getNextEntry()) != null) {
                if (archiveInputStream.canReadEntryData(entry)) {
                    final Path entryPath = Paths.get(USER_HOME + "/" + entry.getName());
                    if (!Files.exists(entryPath)) {
                        if (entry.isDirectory()) {
                            Files.createDirectory(entryPath);
                        } else {
                            final Path dataFile = Files.createFile(entryPath);
                            try (OutputStream dataOutPutStream = Files.newOutputStream(dataFile)) {
                                IOUtils.copy(archiveInputStream, dataOutPutStream);
                            }
                            log.info("Created " + entry.getName());
                        }
                    }
                } else {
                    log.error("Cannot read Entry " + entry.getName());
                }
            }
        }
        log.info("Downloading ended");
    }

}

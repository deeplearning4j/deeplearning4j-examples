package org.deeplearning4j.examples.utilities;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;

import java.io.*;

/**
 * Common data utility functions.
 * 
 * @author fvaleri
 */
public class DataUtilities {

  /**
   * Download a remote file if it doesn't exist.
   * @param remoteUrl URL of the remote file.
   * @param localPath Where to download the file.
   * @return True if and only if the file has been downloaded.
   * @throws Exception IO error.
   */
  public static boolean downloadFile(String remoteUrl, String localPath) throws IOException {
    boolean downloaded = false;
    if (remoteUrl == null || localPath == null)
      return downloaded;
    File file = new File(localPath);
    if (!file.exists()) {
      file.getParentFile().mkdirs();
      HttpClientBuilder builder = HttpClientBuilder.create();
      CloseableHttpClient client = builder.build();
      try (CloseableHttpResponse response = client.execute(new HttpGet(remoteUrl))) {
        HttpEntity entity = response.getEntity();
        if (entity != null) {
          try (FileOutputStream outstream = new FileOutputStream(file)) {
            entity.writeTo(outstream);
            outstream.flush();
            outstream.close();
          }
        }
      }
      downloaded = true;
    }
    if (!file.exists())
      throw new IOException("File doesn't exist: " + localPath);
    return downloaded;
  }

  /**
   * Extract a "tar.gz" file into a local folder.
   * @param inputPath Input file path.
   * @param outputPath Output directory path.
   * @throws IOException IO error.
   */
  public static void extractTarGz(String inputPath, String outputPath) throws IOException {
    if (inputPath == null || outputPath == null)
      return;
    final int bufferSize = 4096;
    if (!outputPath.endsWith("" + File.separatorChar))
      outputPath = outputPath + File.separatorChar;
    try (TarArchiveInputStream tais = new TarArchiveInputStream(
        new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath))))) {
      TarArchiveEntry entry;
      while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          new File(outputPath + entry.getName()).mkdirs();
        } else {
          int count;
          byte data[] = new byte[bufferSize];
          FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
          BufferedOutputStream dest = new BufferedOutputStream(fos, bufferSize);
          while ((count = tais.read(data, 0, bufferSize)) != -1) {
            dest.write(data, 0, count);
          }
          dest.close();
        }
      }
    }
  }

}

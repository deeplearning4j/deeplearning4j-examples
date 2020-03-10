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

package org.deeplearning4j.examples.utilities;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.http.HttpEntity;
import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.impl.client.HttpClients;

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

      String scheme = "http";
      if(remoteUrl.toLowerCase().startsWith("https")){
        scheme = "https";
      }
      String proxyHost = System.getProperty(scheme + ".proxyHost");
      String proxyPort = System.getProperty(scheme + ".proxyPort");
      String proxyUser = System.getProperty(scheme + ".proxyUser");
      String proxyPass = System.getProperty(scheme + ".proxyPassword");

      CloseableHttpClient client = null;
      if (proxyHost == null || proxyHost.isEmpty() || !NumberUtils.isNumber(proxyPort)) {
        HttpClientBuilder builder = HttpClientBuilder.create();
        client = builder.build();
      } else {
        HttpHost proxy = new HttpHost(proxyHost, Integer.parseInt(proxyPort), scheme);
        CredentialsProvider credsProvider = new BasicCredentialsProvider();
        UsernamePasswordCredentials proxyCredentials = null;
        if (proxyUser != null && proxyPass != null){
          proxyCredentials = new UsernamePasswordCredentials(proxyUser, proxyPass);
          credsProvider.setCredentials(
            new AuthScope(proxy),
            proxyCredentials);
        }
        RequestConfig config = RequestConfig.custom()
          .setProxy(proxy)
          .build();
        client = HttpClients.custom()
          .setDefaultCredentialsProvider(credsProvider)
          .setDefaultRequestConfig(config)
          .build();
      }
      try (CloseableHttpResponse response = client.execute(new HttpGet(remoteUrl))) {
        HttpEntity entity = response.getEntity();
        if (entity != null) {
          try (FileOutputStream outstream = new FileOutputStream(file)) {
            entity.writeTo(outstream);
            outstream.flush();
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

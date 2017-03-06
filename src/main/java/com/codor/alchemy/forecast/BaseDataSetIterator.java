package com.codor.alchemy.forecast;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;

public class BaseDataSetIterator {

	protected volatile RemoteIterator<LocatedFileStatus> hdfsIterator;

	protected volatile FileSystem fs;
	protected volatile String hdfsUrl;

	static final String HDFS_URL = "hdfs://...";
	static final String DATA_DIR = "/user/your_home/workplace";
	private static final String CORE_SITE = "/etc/hadoop-2.7.1/core-site.xml";
	private static final String HDFS_SITE = "/etc/hadoop-2.7.1/hdfs-site.xml";

	public BaseDataSetIterator(String hdfsUrl) {
		this.hdfsUrl = hdfsUrl;
		initialize();
	}

	final void initialize() {

		Configuration configuration = new Configuration();
		configuration.addResource(new Path(CORE_SITE));
		configuration.addResource(new Path(HDFS_SITE));

		configuration.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
		configuration.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());

		try {
			fs = FileSystem.get(configuration);
			hdfsIterator = fs.listFiles(new Path(hdfsUrl), true);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	protected String getRelativeFilename(String path) {
		String index = null;
		StringTokenizer pathTokens = new StringTokenizer(path, "/");
		while (pathTokens.hasMoreTokens()) {
			index = pathTokens.nextToken();
		}
		return index;
	}
}


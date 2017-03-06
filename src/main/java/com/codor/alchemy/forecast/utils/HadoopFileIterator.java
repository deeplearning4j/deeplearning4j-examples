package com.codor.alchemy.forecast.utils;

import java.io.File;

import org.deeplearning4j.text.documentiterator.FileDocumentIterator;

public class HadoopFileIterator extends FileDocumentIterator {

	private static final long serialVersionUID = -3475360183611066187L;

	public HadoopFileIterator(File file) {
		super(file);
	}

}

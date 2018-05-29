package org.deeplearning4j.examples.recurrent.character.harmonies;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ExtractStandardSizeImagesFromMidiImages {
	private static int targetWidth=1000;
	private static int targetHeight=8;
	private final static String tmpDirPath = System.getProperty("java.io.tmpdir");

	private static BufferedImage convert(BufferedImage image) {
		int oldWidth=image.getWidth();
		int oldHeight=image.getHeight();
		BufferedImage newImage = new BufferedImage(targetWidth,targetHeight,image.getType());
		int numberOfRows = Math.min(oldHeight,targetHeight);
		for(int row=0;row<numberOfRows;row++) { 
			for(int col=0;col<targetWidth;col++) {
				newImage.setRGB(col, row, image.getRGB(col, row));
			}
		}
		return newImage;
	}
	private static BufferedImage convert(BufferedImage image, int startRow) {
		BufferedImage newImage = new BufferedImage(targetWidth,targetHeight,image.getType());
		for(int i=0;i<targetHeight;i++) {
			int row=startRow+i;
			for(int col=0;col<targetWidth;col++) {
				newImage.setRGB(col, i, image.getRGB(col, row));
			}
		}
		return newImage;
	}
	private static void convertImages(File file, File destinationDirectory) throws IOException {
		if (file.isDirectory()) {
			for(File child: file.listFiles()) {
				convertImages(child,destinationDirectory);
			}
		} else {
			BufferedImage image=ImageIO.read(file);
			System.out.println(file.getName() + " has dimensions " + image.getWidth() + ", " + image.getHeight());
			if (image.getWidth()< targetWidth) {
				return;
			}
			BufferedImage convertedImage=convert(image);
			File outputFile = new File(destinationDirectory,file.getName());
			ImageIO.write(convertedImage, "png", outputFile);
		}
	}
	
	public static void main(String[] args) {
		try {
			File outputDir=new File(tmpDirPath + "/converted-images");
			if (!outputDir.exists()) {
				outputDir.mkdirs();
			}
			convertImages(new File(tmpDirPath + "/midi-images"), outputDir);
		} catch (Throwable thr) {
			thr.printStackTrace();
		}
	}

}

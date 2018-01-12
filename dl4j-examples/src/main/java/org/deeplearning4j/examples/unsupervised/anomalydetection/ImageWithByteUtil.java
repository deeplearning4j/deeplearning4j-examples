package org.deeplearning4j.examples.unsupervised.anomalydetection;

import org.deeplearning4j.datasets.mnist.MnistManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.File;

/**the util will convert the ubyte into the image,this's ready for abnormal detected
 * @author WANG FENG
 */
public class ImageWithByteUtil {
    private static final Logger log = LoggerFactory.getLogger(ImageWithByteUtil.class);

    protected static final String TEMP_ROOT = System.getProperty("user.home");
    protected static final String MNIST_ROOT= TEMP_ROOT + File.separator + "MNIST" + File.separator;


    public static void bytes2Image( ) throws Exception {
        String images;
        String labels;
        int totalExamples;

        images = MNIST_ROOT + "train-images-idx3-ubyte";
        labels = MNIST_ROOT + "train-labels-idx1-ubyte";
        totalExamples = 60000;
        MnistManager man = new MnistManager(images, labels, true);
        String path = File.separator + "home" + File.separator + "MNIST" + File.separator + "train" + File.separator;

        BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
        for(int i = 0; i < totalExamples; ++ i) {
            byte[] img = man.readImageUnsafe(i);
            for( int k = 0; k < 784; k ++ ){
                bi.setRGB(k % 28, k / 28,  img[k]);
            }
            log.info("the image is writing to the disk. training{}",i);
            String filepath = path +  i + ".jpg";
            ImageIO.write(bi, "jpg",new File(filepath));
        }

        images = MNIST_ROOT + "t10k-images-idx3-ubyte";
        labels = MNIST_ROOT + "t10k-labels-idx1-ubyte";
        totalExamples = 10000;
        man = new MnistManager(images, labels, false);
        path = File.separator + "home" + File.separator + "MNIST" + File.separator + "test" + File.separator;

        for(int i = 0; i < totalExamples; ++ i) {
            byte[] img = man.readImageUnsafe(i);
            for( int k = 0; k < 784; k ++ ){
                bi.setRGB(k % 28, k / 28,  img[k]);
            }
            log.info("the image is writing to the disk. testing{}",i);
            String filepath = path +  i + ".jpg";
            ImageIO.write(bi, "jpg",new File(filepath));
        }

    }
    public static byte[] image2Bytes(File f) throws Exception {
		BufferedImage bi = ImageIO.read(f);
		int imageType = bi.getType();
		int width = bi.getWidth();
		int height = bi.getHeight();

		BufferedImage grayImage = new BufferedImage(width, height,BufferedImage.TYPE_BYTE_GRAY);
		new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(bi, grayImage);
		return (byte[]) grayImage.getData().getDataElements(0, 0, width, height, null);
	}
    public static void main(String[] args) throws Exception {
        bytes2Image();
    }


}

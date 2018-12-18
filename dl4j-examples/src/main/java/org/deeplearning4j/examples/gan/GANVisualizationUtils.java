package org.deeplearning4j.examples.gan;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class GANVisualizationUtils {

    public static JFrame initFrame() {
        JFrame frame = new JFrame();
        frame.setTitle("Viz");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        return frame;
    }

    public static JPanel initPanel(JFrame frame, int numSamples) {
        JPanel panel = new JPanel();

        panel.setLayout(new GridLayout(numSamples / 3, 1, 8, 8));
        frame.add(panel, BorderLayout.CENTER);
        frame.setVisible(true);
        return panel;
    }

    public static void visualize(INDArray[] samples, JFrame frame, JPanel panel) {
        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 784; i++) {
            int pixel = (int)(((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }
}

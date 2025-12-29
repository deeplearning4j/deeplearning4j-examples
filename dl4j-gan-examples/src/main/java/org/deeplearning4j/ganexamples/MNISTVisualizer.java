package org.deeplearning4j.ganexamples;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * @author zdl
 */
public class MNISTVisualizer {
    private double imageScale;
    private List<INDArray> digits;
    private String title;
    private int gridWidth;
    private JFrame frame;

    public MNISTVisualizer(double imageScale, String title) {
        this(imageScale, title, 5);
    }

    public MNISTVisualizer(double imageScale, String title, int gridWidth) {
        this.imageScale = imageScale;
        this.title = title;
        this.gridWidth = gridWidth;
    }

    public void visualize() {
        if (null != frame) {
            frame.dispose();
        }
        frame = new JFrame();
        frame.setTitle(title);
        frame.setSize(800, 600);
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(800, 600));
        panel.setLayout(new GridLayout(0, gridWidth));
        List<JLabel> list = getComponents();
        for (JLabel image : list) {
            panel.add(image);
        }

        frame.add(panel);
        frame.setVisible(true);
        frame.pack();
    }

    public List<JLabel> getComponents() {
        List<JLabel> images = new ArrayList<JLabel>();
        for (INDArray arr : digits) {
            BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < 784; i++) {
                bi.getRaster().setSample(i % 28, i / 28, 0, (int) (255 * arr.getDouble(i)));
            }
            ImageIcon orig = new ImageIcon(bi);
            Image imageScaled = orig.getImage().getScaledInstance((int) (imageScale * 28), (int) (imageScale * 28),
                Image.SCALE_DEFAULT);
            ImageIcon scaled = new ImageIcon(imageScaled);
            images.add(new JLabel(scaled));
        }
        return images;
    }

    public List<INDArray> getDigits() {
        return digits;
    }

    public void setDigits(List<INDArray> digits) {
        this.digits = digits;
    }

}

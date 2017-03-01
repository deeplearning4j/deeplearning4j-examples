package org.deeplearning4j.examples.multigpu.video;

import org.apache.commons.io.FilenameUtils;
import org.jcodec.api.SequenceEncoder;

import java.awt.*;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Random;

/**A support class for generating a synthetic video data set
 * Nothing here is specific to DL4J
 * @author Alex Black
 */
public class VideoGenerator {

    public static final int NUM_SHAPES = 4;  //0=circle, 1=square, 2=arc, 3=line
    public static final int MAX_VELOCITY = 3;
    public static final int SHAPE_SIZE = 25;
    public static final int SHAPE_MIN_DIST_FROM_EDGE = 15;
    public static final int DISTRACTOR_MIN_DIST_FROM_EDGE = 0;
    public static final int LINE_STROKE_WIDTH = 6;  //Width of line (line shape only)
    public static final BasicStroke lineStroke = new BasicStroke(LINE_STROKE_WIDTH);
    public static final int MIN_FRAMES = 10;    //Minimum number of frames the target shape to be present
    public static final float MAX_NOISE_VALUE = 0.5f;

    private static int[] generateVideo(String path, int nFrames, int width, int height, int numShapes, Random r,
                                      boolean backgroundNoise, int numDistractorsPerFrame) throws Exception {

        //First: decide where transitions between one shape and another are
        double[] rns = new double[numShapes];
        double sum = 0;
        for (int i = 0; i < numShapes; i++) {
            rns[i] = r.nextDouble();
            sum += rns[i];
        }
        for (int i = 0; i < numShapes; i++) rns[i] /= sum;

        int[] startFrames = new int[numShapes];
        startFrames[0] = 0;
        for (int i = 1; i < numShapes; i++) {
            startFrames[i] = (int) (startFrames[i - 1] + MIN_FRAMES + rns[i] * (nFrames - numShapes * MIN_FRAMES));
        }

        //Randomly generate shape positions, velocities, colors, and type
        int[] shapeTypes = new int[numShapes];
        int[] initialX = new int[numShapes];
        int[] initialY = new int[numShapes];
        double[] velocityX = new double[numShapes];
        double[] velocityY = new double[numShapes];
        Color[] color = new Color[numShapes];
        for (int i = 0; i < numShapes; i++) {
            shapeTypes[i] = r.nextInt(NUM_SHAPES);
            initialX[i] = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE - 2*SHAPE_MIN_DIST_FROM_EDGE );
            initialY[i] = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE - 2*SHAPE_MIN_DIST_FROM_EDGE );
            velocityX[i] = -1 + 2 * r.nextDouble();
            velocityY[i] = -1 + 2 * r.nextDouble();
            color[i] = new Color(r.nextFloat(), r.nextFloat(), r.nextFloat());
        }

        //Generate a sequence of BufferedImages with the given shapes, and write them to the video
        SequenceEncoder enc = new SequenceEncoder(new File(path));
        int currShape = 0;
        int[] labels = new int[nFrames];
        for (int i = 0; i < nFrames; i++) {
            if (currShape < numShapes - 1 && i >= startFrames[currShape + 1]) currShape++;

            BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2d = bi.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setBackground(Color.BLACK);

            if(backgroundNoise){
                for( int x=0; x<width; x++ ){
                    for( int y=0; y<height; y++ ){
                        bi.setRGB(x,y,new Color(r.nextFloat()*MAX_NOISE_VALUE,r.nextFloat()*MAX_NOISE_VALUE,r.nextFloat()*MAX_NOISE_VALUE).getRGB());
                    }
                }
            }

            g2d.setColor(color[currShape]);

            //Position of shape this frame
            int currX = (int) (initialX[currShape] + (i - startFrames[currShape]) * velocityX[currShape] * MAX_VELOCITY);
            int currY = (int) (initialY[currShape] + (i - startFrames[currShape]) * velocityY[currShape] * MAX_VELOCITY);

            //Render the shape
            switch (shapeTypes[currShape]) {
                case 0:
                    //Circle
                    g2d.fill(new Ellipse2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE));
                    break;
                case 1:
                    //Square
                    g2d.fill(new Rectangle2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE));
                    break;
                case 2:
                    //Arc
                    g2d.fill(new Arc2D.Double(currX,currY,SHAPE_SIZE,SHAPE_SIZE,315,225,Arc2D.PIE));
                    break;
                case 3:
                    //Line
                    g2d.setStroke(lineStroke);
                    g2d.draw(new Line2D.Double(currX,currY,currX+SHAPE_SIZE,currY+SHAPE_SIZE));
                    break;
                default:
                    throw new RuntimeException();
            }

            //Add some distractor shapes, which are present for one frame only
            for( int j=0; j<numDistractorsPerFrame; j++ ){
                int distractorShapeIdx = r.nextInt(NUM_SHAPES);

                int distractorX = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE);
                int distractorY = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE);

                g2d.setColor(new Color(r.nextFloat(), r.nextFloat(), r.nextFloat()));

                switch(distractorShapeIdx){
                    case 0:
                        g2d.fill(new Ellipse2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE));
                        break;
                    case 1:
                        g2d.fill(new Rectangle2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE));
                        break;
                    case 2:
                        g2d.fill(new Arc2D.Double(distractorX,distractorY,SHAPE_SIZE,SHAPE_SIZE,315,225,Arc2D.PIE));
                        break;
                    case 3:
                        g2d.setStroke(lineStroke);
                        g2d.draw(new Line2D.Double(distractorX,distractorY,distractorX+SHAPE_SIZE,distractorY+SHAPE_SIZE));
                        break;
                    default:
                        throw new RuntimeException();
                }
            }

            enc.encodeImage(bi);
            g2d.dispose();
            labels[i] = shapeTypes[currShape];
        }
        enc.finish();   //write .mp4

        return labels;
    }

    public static void generateVideoData(String outputFolder, String filePrefix, int nVideos, int nFrames,
                                         int width, int height, int numShapesPerVideo, boolean backgroundNoise,
                                         int numDistractorsPerFrame, long seed) throws Exception {
        Random r = new Random(seed);

        for (int i = 0; i < nVideos; i++) {
            String videoPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".mp4");
            String labelsPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".txt");
            int[] labels = generateVideo(videoPath, nFrames, width, height, numShapesPerVideo, r, backgroundNoise, numDistractorsPerFrame);

            //Write labels to text file
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < labels.length; j++) {
                sb.append(labels[j]);
                if (j != labels.length - 1) sb.append("\n");
            }
            Files.write(Paths.get(labelsPath), sb.toString().getBytes("utf-8"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }
    }
}

package org.goncalves;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final String LIB_PATH = "C:\\Users\\Ryzen 7\\Downloads\\opencv\\build\\java\\x64";
    private static final String MODEL_CONFIGURATION = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\yolov3.cfg";
    private static final String MODEL_WEIGHTS = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\yolov3.weights";
    private static final String CLASS_NAMES_FILE = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\coco.names";

    public static void main(String[] args) {
        System.setProperty("java.library.path", LIB_PATH);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        JFrame window = setupWindow();
        JLabel label = new JLabel();
        window.getContentPane().add(label); // Adiciona o label ao contentPane da janela
        window.setVisible(true);

        VideoCapture capture = null;
        try {
            capture = new VideoCapture(0);
            configureCapture(capture);
            if (!capture.isOpened()) {
                System.err.println("Erro ao acessar a webcam.");
                return;
            }

            Net net = Dnn.readNetFromDarknet(MODEL_CONFIGURATION, MODEL_WEIGHTS);
            List<String> classNames = loadClassNames(CLASS_NAMES_FILE);

            processVideo(capture, net, classNames, label);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (capture != null && capture.isOpened()) {
                capture.release();
            }
        }
    }

    private static JFrame setupWindow() {
        JFrame window = new JFrame("Detecção de Objetos na Webcam");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setSize(800, 600);
        return window;
    }

    private static void configureCapture(VideoCapture capture) {
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
    }

    private static List<String> loadClassNames(String classNamesFile) {
        List<String> classNames = new ArrayList<>();
        try {
            Files.lines(Paths.get(classNamesFile)).forEach(classNames::add);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private static void processVideo(VideoCapture capture, Net net, List<String> classNames, JLabel label) {
        Mat frame = new Mat();
        while (true) {
            if (!capture.read(frame)) break;

            Mat blob = Dnn.blobFromImage(frame, 1.0 / 255.0, new Size(320, 320), new Scalar(0, 0, 0), true, false);
            net.setInput(blob);

            List<Mat> result = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);
            net.forward(result, outBlobNames);

            drawDetections(frame, result, classNames);

            BufferedImage img = matToBufferedImage(frame);
            ImageIcon image = new ImageIcon(img);
            label.setIcon(image);
            label.repaint();
        }
    }

    private static void drawDetections(Mat frame, List<Mat> result, List<String> classNames) {
        for (Mat level : result) {
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;

                if (confidence > 0.95) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(0, 255, 0), 2);

                    String labelCS = classNames.get((int) classIdPoint.x);
                    String objectType = getObjectType(labelCS);
                    String text = objectType + " (" + String.format("%.2f", confidence * 100) + "%)";
                    Imgproc.putText(frame, text, new Point(left, top - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                }
            }
        }
    }

    private static String getObjectType(String labelCS) {
        switch (labelCS) {
            case "person":
                return "Human";
            case "cat":
                return "Cat";
            case "dog":
                return "Dog";
            case "cell phone":
                return "Cell Phone";
            default:
                return "Unknown";
        }
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        for (Integer i : outLayers) {
            names.add(layersNames.get(i - 1));
        }
        return names;
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] bytes = new byte[bufferSize];
        mat.get(0, 0, bytes);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(bytes, 0, targetPixels, 0, bytes.length);
        return image;
    }
}

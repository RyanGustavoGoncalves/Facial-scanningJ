package org.goncalves;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
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

        // Configuração da janela
        JFrame window = setupWindow();
        JLabel label = new JLabel();
        label.setHorizontalAlignment(JLabel.CENTER);
        label.setVerticalAlignment(JLabel.CENTER);
        window.getContentPane().add(label, BorderLayout.CENTER);
        window.setVisible(true);

        VideoCapture capture = new VideoCapture(0);
        try {
            configureCapture(capture);
            if (!capture.isOpened()) {
                System.err.println("Erro ao acessar a webcam.");
                return;
            }

            // Carrega a rede neural e as classes
            Net net = Dnn.readNetFromDarknet(MODEL_CONFIGURATION, MODEL_WEIGHTS);
            List<String> classNames = loadClassNames(CLASS_NAMES_FILE);

            // Inicia o processamento do vídeo
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
        window.setLayout(new BorderLayout()); // Define o layout para centralizar o vídeo
        return window;
    }

    private static void configureCapture(VideoCapture capture) {
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
    }

    private static List<String> loadClassNames(String classNamesFile) {
        try {
            return Files.readAllLines(Paths.get(classNamesFile));
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    private static void processVideo(VideoCapture capture, Net net, List<String> classNames, JLabel label) {
        Mat frame = new Mat();
        int frameCount = 0;
        long startTime = System.currentTimeMillis();
        double fps = 0.0;

        while (true) {
            if (!capture.read(frame)) break;

            // Redimensiona o frame para melhorar a performance
            Imgproc.resize(frame, frame, new Size(640, 480));

            // Pré-processamento da imagem para a rede neural
            Mat blob = Dnn.blobFromImage(frame, 1.0 / 255.0, new Size(416, 416), new Scalar(0, 0, 0), true, false);
            net.setInput(blob);

            // Executa a detecção
            List<Mat> result = new ArrayList<>();
            List<String> outBlobNames = getOutputNames(net);
            net.forward(result, outBlobNames);

            // Desenha as detecções na imagem
            drawDetections(frame, result, classNames);

            // Atualiza o contador de frames
            frameCount++;
            long currentTime = System.currentTimeMillis();
            long elapsedTime = currentTime - startTime;
            if (elapsedTime >= 1000) {
                fps = frameCount / (elapsedTime / 1000.0);
                frameCount = 0;
                startTime = currentTime;
            }

            // Exibe o FPS na imagem
            Imgproc.putText(frame, String.format("FPS: %.2f", fps), new Point(10, 25), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7,
                    new Scalar(0, 255, 0), 2);

            // Converte o frame para BufferedImage e exibe na janela
            BufferedImage img = matToBufferedImage(frame);
            ImageIcon image = new ImageIcon(img);
            label.setIcon(image);
            label.repaint();
        }
    }

    private static void drawDetections(Mat frame, List<Mat> result, List<String> classNames) {
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();

        float confThreshold = 0.5f; // Limite de confiança para filtrar detecções
        float nmsThreshold = 0.4f;  // Limite para a supressão não máxima

        for (Mat level : result) {
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, row.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                int classId = (int) mm.maxLoc.x;

                if (confidence > confThreshold) {
                    // Obtém os dados da detecção
                    float[] data = new float[(int) (row.total() * row.channels())];
                    row.get(0, 0, data);

                    if (data.length >= 4) {
                        // Calcula as coordenadas da caixa delimitadora
                        float centerX = data[0] * frame.cols();
                        float centerY = data[1] * frame.rows();
                        float width = data[2] * frame.cols();
                        float height = data[3] * frame.rows();
                        float left = centerX - width / 2;
                        float top = centerY - height / 2;

                        classIds.add(classId);
                        confidences.add(confidence);
                        boxes.add(new Rect2d(left, top, width, height));
                    }
                }
            }
        }

        // Verifica se há detecções antes de prosseguir
        if (!confidences.isEmpty()) {
            // Aplica a supressão não máxima para eliminar caixas sobrepostas
            MatOfFloat confidencesMat = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
            Rect2d[] boxesArray = boxes.toArray(new Rect2d[0]);
            MatOfRect2d boxesMat = new MatOfRect2d(boxesArray);
            MatOfInt indices = new MatOfInt();

            Dnn.NMSBoxes(boxesMat, confidencesMat, confThreshold, nmsThreshold, indices);

            int[] indicesArray = indices.toArray();
            for (int idx : indicesArray) {
                Rect2d box = boxes.get(idx);
                int classId = classIds.get(idx);
                float confidence = confidences.get(idx);

                // Desenha o retângulo da detecção
                Imgproc.rectangle(frame, new Point(box.x, box.y), new Point(box.x + box.width, box.y + box.height),
                        new Scalar(0, 255, 0), 2);

                String labelCS = classNames.get(classId);
                String objectType = getObjectType(labelCS);
                String text = objectType + " (" + String.format("%.2f", confidence * 100) + "%)";

                // Ajusta o tamanho da fonte e a espessura para melhor visualização
                Imgproc.putText(frame, text, new Point(box.x, box.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                        new Scalar(0, 255, 0), 1);
            }
        }
    }

    private static String getObjectType(String labelCS) {
        switch (labelCS) {
            case "person":
                return "Humano";
            case "cat":
                return "Gato";
            case "dog":
                return "Cachorro";
            case "cell phone":
                return "Celular";
            default:
                return labelCS;
        }
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        MatOfInt outLayers = new MatOfInt(net.getUnconnectedOutLayers());
        List<String> layersNames = net.getLayerNames();
        int[] outLayersArray = outLayers.toArray();
        for (int i : outLayersArray) {
            names.add(layersNames.get(i - 1));
        }
        return names;
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }
        int type = BufferedImage.TYPE_INT_RGB;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        if (mat.channels() == 3) {
            // Converte os dados de BGR para RGB
            int[] data = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            int r, g, blue;
            for (int i = 0; i < data.length; i++) {
                blue = b[i * 3] & 0xFF;
                g = b[i * 3 + 1] & 0xFF;
                r = b[i * 3 + 2] & 0xFF;
                data[i] = (r << 16) | (g << 8) | blue;
            }
        } else if (mat.channels() == 1) {
            System.arraycopy(b, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData(), 0, b.length);
        }
        return image;
    }
}

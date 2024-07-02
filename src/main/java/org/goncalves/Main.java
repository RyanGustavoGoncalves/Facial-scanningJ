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
    public static void main(String[] args) {
        // Caminho para a biblioteca nativa do OpenCV
        String libPath = "C:\\Users\\Ryzen 7\\Downloads\\opencv\\build\\java\\x64";
        System.setProperty("java.library.path", libPath);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Inicializa a captura de vídeo da webcam
        VideoCapture capture = new VideoCapture(0); // 0 representa a primeira webcam conectada
        // Reduz a resolução do frame capturado
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);  // Largura reduzida
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480); // Altura reduzida

        if (!capture.isOpened()) {
            System.out.println("Erro ao acessar a webcam.");
            return;
        }

        // Configura a janela para exibir o vídeo
        JFrame window = new JFrame("Detecção de Objetos na Webcam");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel label = new JLabel();
        window.setContentPane(label);
        window.setSize(800, 600);
        window.setVisible(true);

        // Caminhos para os arquivos de configuração do YOLOv3
        String modelConfiguration = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\yolov3.cfg";
        String modelWeights = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\yolov3.weights";
        String classNamesFile = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\coco.names";

        // Leitura dos arquivos de configuração e nomes das classes (fora do loop principal)
        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        List<String> classNames = new ArrayList<>();
        try {
            Files.lines(Paths.get(classNamesFile)).forEach(classNames::add);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Mat frame = new Mat();
        BufferedImage img;
        while (window.isVisible()) {
            if (capture.read(frame)) {
                // Converte o frame para um blob com tamanho reduzido
                Mat blob = Dnn.blobFromImage(frame, 1.0 / 255.0, new Size(320, 320), new Scalar(0, 0, 0), true, false);


                // Configura o blob como input da rede
                net.setInput(blob);

                // Realiza a detecção de objetos
                List<Mat> result = new ArrayList<>();
                List<String> outBlobNames = getOutputNames(net);
                net.forward(result, outBlobNames);

                // Processa os resultados
                for (int i = 0; i < result.size(); ++i) {
                    Mat level = result.get(i);
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

                            // Desenha retângulos ao redor dos objetos detectados
                            Imgproc.rectangle(frame, new Point(left, top), new Point(left + width, top + height), new Scalar(0, 255, 0), 2);

                            // Identifica se é um humano ou outro objeto
                            String labelCS = classNames.get((int) classIdPoint.x);
                            String objectType = labelCS.equals("person") ? "Humano" : "Objeto";
                            Imgproc.putText(frame, objectType, new Point(left, top - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                        }
                    }
                }

                // Converte o frame para BufferedImage para exibição na janela Swing
                img = matToBufferedImage(frame);
                ImageIcon image = new ImageIcon(img);
                label.setIcon(image);
                label.repaint();
            }
        }

        capture.release();
    }

    // Método auxiliar para obter os nomes das camadas de saída do YOLO
    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();
        for (Integer i : outLayers) {
            names.add(layersNames.get(i - 1));
        }
        return names;
    }

    // Método auxiliar para converter Mat para BufferedImage
    public static BufferedImage matToBufferedImage(Mat mat) {
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

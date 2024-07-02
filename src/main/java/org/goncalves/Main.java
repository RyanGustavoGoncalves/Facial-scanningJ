package org.goncalves;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class Main {
    public static void main(String[] args) {
        // Caminho para a biblioteca nativa do OpenCV
        String libPath = "C:\\Users\\Ryzen 7\\Downloads\\opencv\\build\\java\\x64"; // Substitua pelo seu caminho

        // Configura o java.library.path
        System.setProperty("java.library.path", libPath);

        // Carrega a biblioteca nativa do OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Inicializa a captura de vídeo da webcam
        VideoCapture capture = new VideoCapture(0); // 0 representa a primeira webcam conectada

        // Verifica se a captura foi iniciada corretamente
        if (!capture.isOpened()) {
            System.out.println("Erro ao acessar a webcam.");
            return;
        }

        // Cria uma janela para exibir o vídeo
        JFrame window = new JFrame("Detecção Facial na Webcam");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel label = new JLabel();
        window.setContentPane(label);
        window.setSize(800, 600);
        window.setVisible(true);

        // Carrega o classificador Haar Cascade para detecção de rostos
        String xmlFile = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\haarcascade_frontalface_alt.xml"; // Substitua pelo seu caminho
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        // Loop principal para captura e processamento de frames
        Mat frame = new Mat();
        BufferedImage img;
        while (window.isVisible()) {
            // Captura um frame da webcam
            if (capture.read(frame)) {
                // Realiza a detecção de rostos
                MatOfRect faceDetections = new MatOfRect();
                classifier.detectMultiScale(frame, faceDetections);

                // Desenha retângulos ao redor dos rostos detectados
                for (Rect rect : faceDetections.toArray()) {
                    Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0));
                }

                // Converte o frame para BufferedImage para exibição na janela Swing
                img = matToBufferedImage(frame);

                // Atualiza a imagem na janela Swing
                ImageIcon image = new ImageIcon(img);
                label.setIcon(image);
                label.repaint();
            }
        }

        // Libera os recursos da captura de vídeo
        capture.release();
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

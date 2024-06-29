package org.goncalves;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class Main {
    public static void main(String[] args) {
        // Set the library path (update this to your path)
        String libPath = "C:\\Users\\Ryzen 7\\Downloads\\opencv\\build\\java\\x64";
        System.setProperty("java.library.path", libPath);

        // Reload the library path
        System.setProperty("java.library.path", System.getProperty("java.library.path") + ";" + libPath);

        // Load the OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the Haar Cascade XML file (update this to your path)
        String xmlFile = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\haarcascade_frontalface_alt.xml";
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        // Load the image file (update this to your path)
        String imagePath = "C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\3x4.jpeg";
        Mat image = Imgcodecs.imread(imagePath);

        // Check if the image was loaded correctly
        if (image.empty()) {
            System.out.println("Couldn't open image file: " + imagePath);
            return;
        }

        // Perform face detection
        MatOfRect faceDetections = new MatOfRect();
        classifier.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        // Draw rectangles around each face
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }

        // Save the output image
        Imgcodecs.imwrite("C:\\Users\\Ryzen 7\\Desktop\\Facial-scanningJ\\src\\main\\data\\output.jpg", image);
    }
}

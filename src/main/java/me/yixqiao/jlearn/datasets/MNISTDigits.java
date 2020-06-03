package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

import java.io.*;
import java.util.ArrayList;

/**
 * Class for the MNIST digits dataset.
 */
public final class MNISTDigits {
    /**
     * Load the data.
     *
     * @return the dataset
     */
    public static DatasetTT load() {
        return load("datasets/mnist/data/");
    }

    /**
     * Load the data from a directory.
     * <p>
     * The directory must contain two files: test.dat and train.dat
     * </p>
     *
     * @param directoryPath path to the data
     * @return the dataset
     */
    public static DatasetTT load(String directoryPath) {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();

        // Load training

        Matrix trainX = new Matrix(60000, 28 * 28);
        Matrix trainY = new Matrix(60000, 10);
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(directoryPath + "train.dat")));
            for (int i = 0; i < 60000; i++) {
                trainY.mat[i][dis.readByte()] = 1;

                for (int j = 0; j < 28 * 28; j++) {
                    trainX.mat[i][j] = dis.readByte() + 128;
                }


            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        trainX.multiplyIP(1.0 / 255);

        // Load testing
        Matrix testX = new Matrix(10000, 28 * 28);
        Matrix testY = new Matrix(10000, 10);
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(directoryPath + "test.dat")));
            for (int i = 0; i < 10000; i++) {
                testY.mat[i][dis.readByte()] = 1;

                for (int j = 0; j < 28 * 28; j++) {
                    testX.mat[i][j] = dis.readByte() + 128;
                }


            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        testX.multiplyIP(1.0 / 255);

        return new DatasetTT(trainX, trainY, testX, testY);
    }

    public void writeDataset() {
        // Flattens all images
        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_train.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/train.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 60000; imgCount++) {
                line = br.readLine();
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);

                output.mat[0][Integer.parseInt(values[0])] = 1;
                dos.writeByte((byte) Integer.parseInt(values[0]));

                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);
                    dos.writeByte((byte) (input.mat[0][i] - 128));
                }

                input.multiplyIP(1.0 / 255);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_test.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/test.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 10000; imgCount++) {
                line = br.readLine();
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);

                output.mat[0][Integer.parseInt(values[0])] = 1;
                dos.writeByte((byte) Integer.parseInt(values[0]));

                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);
                    dos.writeByte((byte) (input.mat[0][i] - 128));
                }

                input.multiplyIP(1.0 / 255);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Finished writing to file.");

        System.exit(0);
    }
}

package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class MNISTDigits {
    public static DatasetTT load() {
        return load("datasets/mnist/data/");
    }

    public static DatasetTT load(String directoryPath) {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();

        // Load training

        Matrix trainX = new Matrix(60000, 28 * 28);
        Matrix trainY = new Matrix(60000, 10);
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(directoryPath + "train.dat")));
            for (int i = 0; i < 60000; i++) {
                trainY.mat[0][dis.readByte()] = 1;

                for (int j = 0; j < 28 * 28; j++) {
                    trainX.mat[i][j] = dis.readByte() + 128;
                }


            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        trainX.multiplyIP(1.0 / 255);

        // Load testing
        Matrix testX = new Matrix(60000, 28 * 28);
        Matrix testY = new Matrix(60000, 10);
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(directoryPath + "test.dat")));
            for (int i = 0; i < 60000; i++) {
                testY.mat[0][dis.readByte()] = 1;

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
}

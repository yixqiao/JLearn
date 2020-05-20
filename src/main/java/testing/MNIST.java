package testing;

import activations.ReLU;
import activations.Softmax;
import core.Matrix;
import layers.Dense;
import layers.InputLayer;
import losses.CrossEntropy;
import metrics.Accuracy;
import models.Model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class MNIST {
    ArrayList<Matrix> inputsAL = new ArrayList<>();
    ArrayList<Matrix> outputsAL = new ArrayList<>();
    Model model;
    private Matrix inputs;
    private Matrix outputs;

    public static void main(String[] args) {
        MNIST mnist = new MNIST();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    private void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(28*28))
                .addLayer(new Dense(256, new ReLU()))
                .addLayer(new Dense(64, new ReLU()))
                .addLayer(new Dense(10, new Softmax()));

        model.buildModel(new CrossEntropy());
    }

    private void train() {
        printPredictions();

        model.fit(inputs, outputs, 0.005, 32, 30, 1, new Accuracy());

        printPredictions();
    }

    private void printPredictions() {
        for (int i = 0; i < inputsAL.size(); i += 10000) {
            Matrix output = model.predict(inputsAL.get(i));
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[0][j]));
                if (j != output.cols - 1) System.out.print("\t");
            }

            System.out.print("\t-\t");
            for (int j = 0; j < outputs.cols; j++) {
                System.out.print(String.format("%.3f", outputs.mat[i][j]));
                if (j != outputs.cols - 1) System.out.print("\t");
            }

            System.out.println();
        }
    }

    private void writeDataset() {
        // Flattens all images
        try (BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_train.csv"))) {
            String line;
            br.readLine(); // Discard first line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);
                output.mat[0][Integer.parseInt(values[0])] = 1;
                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);

                }
                input.multiplyIP(1.0 / 255);
                inputsAL.add(input);
                outputsAL.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        inputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        outputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                inputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                outputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }
    }

    private void initInputs() {
        writeDataset();
    }
}

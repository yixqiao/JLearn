package testing;

import activations.LeakyReLU;
import activations.ReLU;
import activations.Softmax;
import core.Matrix;
import layers.Dense;
import layers.InputLayer;
import models.Model;

import java.util.ArrayList;

public class XOR {
    Model model;
    private ArrayList<Matrix> inputs;
    private ArrayList<Matrix> outputs;

    public static void main(String[] args) {
        XOR xor = new XOR();
        xor.initInputs();
        xor.buildModel();
        xor.train();
    }

    private void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(2))
                .addLayer(new Dense(16, new LeakyReLU(0.02)))
                .addLayer(new Dense(1, new LeakyReLU(0.02)));
        model.buildModel(0.001);
    }

    private void train() {
        for (int i = 0; i < 100000; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                model.trainOnBatch(inputs.get(j), outputs.get(j));
            }
            if (i % 10000 == 0) {
                System.out.println("\n" + i);
                printPredictions();
            }
        }
    }

    private void printPredictions() {
        for (int i = 0; i < inputs.size(); i++) {
            Matrix input = inputs.get(i);
            for (int j = 0; j < input.cols; j++) {
                System.out.print(input.mat[0][j]);
                if (j != input.cols - 1) System.out.print(",");
            }
            System.out.print(" :  ");
            Matrix output = model.predict(inputs.get(i));
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[0][j]));
                if (j != output.cols - 1) System.out.print(",");
            }
            System.out.println();
        }
    }

    private void initInputs() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();

        inputs.add(new Matrix(new double[][]{{0.0, 0.0}}));
        outputs.add(new Matrix(new double[][]{{0.0, 1.0}}));

        inputs.add(new Matrix(new double[][]{{0.0, 1.0}}));
        outputs.add(new Matrix(new double[][]{{1.0, 0.0}}));

        inputs.add(new Matrix(new double[][]{{1.0, 1.0}}));
        outputs.add(new Matrix(new double[][]{{0.0, 1.0}}));

        inputs.add(new Matrix(new double[][]{{1.0, 0.0}}));
        outputs.add(new Matrix(new double[][]{{1.0, 0.0}}));
    }
}
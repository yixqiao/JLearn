package testing;

import activations.ReLU;
import activations.Sigmoid;
import core.Matrix;
import models.Model;

import java.util.ArrayList;

public class XOR {
    private ArrayList<Matrix> inputs;
    private ArrayList<Matrix> outputs;
    Model model;

    public static void main(String[] args) {
        XOR xor = new XOR();
        xor.initInputs();
        xor.buildModel();
        xor.train();
    }

    private void buildModel() {
        model = new Model();
        model.addLayer(2).addLayer(16).addLayer(1);
        model.buildModel(new ReLU(), 0.01);
    }

    private void train() {
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                model.fitSingle(inputs.get(j), outputs.get(j));
            }
            if (i % 100 == 0) {
                System.out.println("\n" + i);
                printPredictions();
            }
        }
    }

    private void printPredictions() {
        for (int i = 0; i < inputs.size(); i++) {
            for (int j = 0; j < inputs.get(i).cols; j++) {
                System.out.print(inputs.get(i).mat[0][j]);
                if (j != inputs.get(i).cols - 1) System.out.print(",");
            }
            System.out.println(" : " + model.predict(inputs.get(i)).mat[0][0]);
        }
    }

    private void initInputs() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();

        inputs.add(new Matrix(new double[][]{{0.0, 0.0}}));
        outputs.add(new Matrix(new double[][]{{0.0}}));

        inputs.add(new Matrix(new double[][]{{0.0, 1.0}}));
        outputs.add(new Matrix(new double[][]{{1.0}}));

        inputs.add(new Matrix(new double[][]{{1.0, 1.0}}));
        outputs.add(new Matrix(new double[][]{{0.0}}));

        inputs.add(new Matrix(new double[][]{{1.0, 0.0}}));
        outputs.add(new Matrix(new double[][]{{1.0}}));
    }
}

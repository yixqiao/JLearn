package testing;

import activations.ReLU;
import activations.Sigmoid;
import core.Matrix;
import models.Model;

import java.util.ArrayList;

public class XOR {
    private static ArrayList<Matrix> inputs;
    private static ArrayList<Matrix> outputs;

    public static void main(String[] args) {
        initInputs();
        Model model = new Model();
        model.addLayer(2).addLayer(16).addLayer(1).buildModel(new ReLU(), 0.1);

        System.out.println(model.predict(inputs.get(0)).mat[0][0]);
        System.out.println(model.predict(inputs.get(2)).mat[0][0]);

        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                model.fitSingle(inputs.get(j), outputs.get(j));
            }
            if (i % 100 == 0) {
                System.out.println(i);
            }
        }
        for (int i = 0; i < inputs.size(); i++) {
            for (int j = 0; j < inputs.get(i).cols; j++) {
                System.out.print(inputs.get(i).mat[0][j]);
                if (j != inputs.get(i).cols - 1) System.out.print(",");
            }
            System.out.println(" : " + model.predict(inputs.get(i)).mat[0][0]);
        }
    }

    private static void initInputs() {
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

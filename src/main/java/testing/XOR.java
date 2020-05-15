package testing;

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
        model.addLayer(2).addLayer(16).addLayer(1).buildModel(new Sigmoid(), 0.1);

        System.out.println(model.predict(inputs.get(0)).mat[0][0]);
        System.out.println(model.predict(inputs.get(2)).mat[0][0]);

        for (int i = 0; i < 100000; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                model.fitSingle(inputs.get(j), outputs.get(j));
            }
            if (i % 10000 == 0) {
                System.out.println(i);
            }
            // System.out.println();
        }
        for (int j = 0; j < inputs.size(); j++) {
            System.out.println(model.predict(inputs.get(j)).mat[0][0]);
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

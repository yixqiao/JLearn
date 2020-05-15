package testing;

import core.Matrix;
import core.Model;

import java.util.ArrayList;

public class XOR {
    private static ArrayList<Matrix> inputs;
    private static ArrayList<Matrix> outputs;

    public static void main(String[] args) {
        initInputs();
        Model model = new Model();
        model.addLayer(2);
        model.addLayer(4);
        model.addLayer(1);
        model.buildModel(0.1);

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
        Matrix in = new Matrix(1, 2);
        Matrix out = new Matrix(1, 1);
        {
            in.mat[0][0] = 0;
            in.mat[0][1] = 0;
            out.mat[0][0] = 0;
            inputs.add(in.clone());
            outputs.add(out.clone());
        }
        {
            in.mat[0][0] = 1;
            in.mat[0][1] = 0;
            out.mat[0][0] = 1;
            inputs.add(in.clone());
            outputs.add(out.clone());
        }
        {
            in.mat[0][0] = 1;
            in.mat[0][1] = 1;
            out.mat[0][0] = 0;
            inputs.add(in.clone());
            outputs.add(out.clone());
        }
        {
            in.mat[0][0] = 0;
            in.mat[0][1] = 1;
            out.mat[0][0] = 1;
            inputs.add(in.clone());
            outputs.add(out.clone());
        }
    }
}

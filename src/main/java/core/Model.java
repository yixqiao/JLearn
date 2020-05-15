package core;

import java.util.ArrayList;

public class Model {
    private static ArrayList<Matrix> network = new ArrayList<>();
    private static ArrayList<Matrix> activations = new ArrayList<>();
    private static ArrayList<Matrix> activationsNoS = new ArrayList<>();
    private static double learningRate = 1;
    private static Matrix input = new Matrix(1, 2, true);
    private static ArrayList<Matrix> errors;
    private static ArrayList<Matrix> inputs;
    private static ArrayList<Matrix> outputs;

    public static void main(String[] args) {
        initNetwork();
        initInputs();
        System.out.println(forwardPropagate(inputs.get(0)).mat[0][0]);
        System.out.println(forwardPropagate(inputs.get(2)).mat[0][0]);
        for (int i = 0; i < 100000; i++) {
            for (int j = 0; j < inputs.size(); j++) {
                singlePass(inputs.get(j), outputs.get(j));
            }
            if (i % 10000 == 0) {
                System.out.println();
                System.out.println(i + ": " + activations.get(2).mat[0][0] + "," + errors.get(0).mat[0][0]);
//                System.out.println(forwardPropagate(inputs.get(0)).mat[0][0]);
//                System.out.println(forwardPropagate(inputs.get(2)).mat[0][0]);
            }
            // System.out.println();
        }
        for (int j = 0; j < inputs.size(); j++) {
            System.out.println(forwardPropagate(inputs.get(j)).mat[0][0]);
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

    private static void singlePass(Matrix input, Matrix expected) {
        forwardPropagate(input);//.printMatrix();
        //System.out.println("Output: ");
        //activations.get(2).printMatrix();
        ArrayList<Matrix> errors = backPropagate(expected);
        Model.errors = errors;
        // System.out.println(errors);
        // errors.get(0).printMatrix();
        // errors.get(1).printMatrix();
        // errors.get(2).printMatrix();
        update(errors);
    }

    private static void initNetwork() {
        network.add(new Matrix(2, 36, Math.sqrt(2.0 / 2)));
        network.add(new Matrix(36, 1, Math.sqrt(2.0 / 36)));
    }

    private static Matrix forwardPropagate(Matrix input) {
        activations = new ArrayList<>();
        activationsNoS = new ArrayList<>();
        Matrix activationsLocal = input.clone();
        activationsNoS.add(activationsLocal);
        activationsLocal.sigmoid();
        Model.activations.add(activationsLocal.clone());
        for (Matrix layer : network) {
            Matrix newActivations = activationsLocal.dot(layer);
            activationsNoS.add(newActivations.clone());
            newActivations.sigmoid();
            activationsLocal = newActivations.clone();
            Model.activations.add(activationsLocal.clone());
        }
        return activations.get(activations.size() - 1);
    }

    private static ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        Matrix first = new Matrix(1, 1);
        // first.mat[0][0] = (expected.mat[0][0] - activations.get(2).mat[0][0]) * (expected.mat[0][0] - activations.get(2).mat[0][0]);
        first.mat[0][0] = (expected.mat[0][0] - activations.get(2).mat[0][0]);
        first.mat[0][0] *= transferDerivative(activations.get(2).mat[0][0]);
        errors.add(first);

        Matrix second = new Matrix(1, 36);
        for (int curN = 0; curN < 36; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < 1; prevN++) {
                error += network.get(1).mat[curN][prevN] * errors.get(0).mat[0][prevN];
            }
            second.mat[0][curN] = error * transferDerivative(activations.get(1).mat[0][curN]);
        }
        errors.add(second);

        Matrix third = new Matrix(1, 2);
        for (int curN = 0; curN < 2; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < 36; prevN++) {
                error += network.get(0).mat[curN][prevN] * errors.get(1).mat[0][prevN];
            }
            third.mat[0][curN] = error * transferDerivative(activations.get(0).mat[0][curN]);
        }
        errors.add(third);

        return errors;
    }

    private static void update(ArrayList<Matrix> errors) {
        for (int layer = 0; layer < 2; layer++) {
            int eLayer = 1 - layer;
            for (int curN = 0; curN < activations.get(layer).cols; curN++) {
                for (int nextN = 0; nextN < activations.get(layer + 1).cols; nextN++) {
                    // System.out.println(layer + "," + curN + "," + nextN);
                    // network.get(layer).mat[curN][nextN] = network.get(layer).mat[curN][nextN];
                    network.get(layer).mat[curN][nextN] += learningRate * errors.get(eLayer).mat[0][nextN]
                            * (activations.get(layer).mat[0][curN]);
                }
            }
        }
    }

    private static double transferDerivative(double x) {
        // return (x < 0) ? 0 : 1;
        return x * (1 - x);
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}

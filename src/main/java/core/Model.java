package core;

import java.util.ArrayList;

public class Model {
    private static ArrayList<Matrix> network = new ArrayList<>();
    private static ArrayList<Matrix> activations = new ArrayList<>();
    private static double learningRate = 0.01;
    private static Matrix input = new Matrix(1, 2, true);
    private static ArrayList<Matrix> errors;

    public static void main(String[] args) {
        initNetwork();
        for (int i = 0; i < 10000; i++) {
            singlePass();
            if (i % 500 == 0) {
                System.out.println(i + ": " + activations.get(2).mat[0][0] + "," + errors.get(0).mat[0][0]);
            }

            // System.out.println();
        }
    }

    private static void singlePass() {
        //Matrix input = new Matrix(1, 2, true);
        activations = new ArrayList<>();
        activations.add(input);
        forwardPropagate(input);//.printMatrix();
        //System.out.println("Output: ");
        //activations.get(2).printMatrix();
        Matrix expected = new Matrix(1, 1, true);
        expected.mat[0][0] = 0.8;
        ArrayList<Matrix> errors = backPropagate(expected);
        Model.errors = errors;
        // System.out.println(errors);
        // errors.get(0).printMatrix();
        // errors.get(1).printMatrix();
        // errors.get(2).printMatrix();
        update(errors);
    }

    private static void initNetwork() {
        network.add(new Matrix(2, 4, true));
        network.add(new Matrix(4, 1, true));
    }

    private static Matrix forwardPropagate(Matrix input) {
        Matrix activationsLocal = input.clone();
        for (Matrix layer : network) {
            Matrix newActivations = activationsLocal.dot(layer);
            newActivations.sigmoid();
            activationsLocal = newActivations.clone();
            Model.activations.add(activationsLocal.clone());
        }
        return activationsLocal;
    }

    private static ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        Matrix first = new Matrix(1, 1);
        //first.mat[0][0] = (expected.mat[0][0] - activations.get(2).mat[0][0]) * (expected.mat[0][0] - activations.get(2).mat[0][0]);
        first.mat[0][0] = expected.mat[0][0] - activations.get(2).mat[0][0];
        first.mat[0][0] = transferDerivative(first.mat[0][0]);
        errors.add(first);

        Matrix second = new Matrix(1, 4);
        for (int curN = 0; curN < 4; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < 1; prevN++) {
                error += network.get(1).mat[curN][prevN] * errors.get(0).mat[0][prevN];
            }
            second.mat[0][curN] = transferDerivative(error);
        }
        errors.add(second);

        Matrix third = new Matrix(1, 2);
        for (int curN = 0; curN < 2; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < 4; prevN++) {
                error += network.get(0).mat[curN][prevN] * errors.get(1).mat[0][prevN];
            }
            third.mat[0][curN] = transferDerivative(error);
        }
        errors.add(third);

        return errors;
    }

    private static void update(ArrayList<Matrix> errors) {
        for (int layer = 1; layer >= 0; layer--) {
            int eLayer = 2 - layer;
            for (int neuron = 0; neuron < errors.get(eLayer).cols; neuron++) {
                for (int weight = 0; weight < activations.get(layer + 1).cols; weight++) {
                    // System.out.println(layer + "," + neuron + "," + weight);
                    // network.get(layer).mat[neuron][weight] = network.get(layer).mat[neuron][weight];
                    network.get(layer).mat[neuron][weight] += learningRate * errors.get(eLayer).mat[0][neuron] * activations.get(layer + 1).mat[0][weight];
                }
            }
        }
    }

    private static double transferDerivative(double x) {
        // return (x < 0) ? 0 : 1;
        return x * (1 - x);
    }
}

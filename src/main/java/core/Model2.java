package core;

import java.util.ArrayList;

public class Model2 {
    private static ArrayList<Matrix> network = new ArrayList<>();
    private static ArrayList<Matrix> activations = new ArrayList<>();
    private static double learningRate = 0.1;
    private static Matrix input = new Matrix(1, 2, true);
    private static ArrayList<Matrix> errors;

    public static void main(String[] args) {
        initNetwork();
        for (int i = 0; i < 10000; i++) {
            singlePass();
            if (i % 500 == 0) {
                System.out.println();
                System.out.println(i + ": " + activations.get(1).mat[0][0] + "," + errors.get(0).mat[0][0]);
                errors.get(1).printMatrix();
                System.out.println("Activations");
                activations.get(0).printMatrix();
                System.out.println("Network");
                network.get(0).printMatrix();
            }

            // System.out.println();
        }
    }

    private static void singlePass() {
        Matrix input = new Matrix(1, 2, true);
        input.mat[0][0] = 0;
        input.mat[0][1] = 1;
        // input.sigmoid();
        forwardPropagate(input);//.printMatrix();
        //System.out.println("Output: ");
        //activations.get(2).printMatrix();
        Matrix expected = new Matrix(1, 1, true);
        expected.mat[0][0] = 0.6;
        ArrayList<Matrix> errors = backPropagate(expected);
        Model2.errors = errors;
        // System.out.println(errors);
        // errors.get(0).printMatrix();
        // errors.get(1).printMatrix();
        // errors.get(2).printMatrix();
        update(errors);
    }

    private static void initNetwork() {
        network.add(new Matrix(2, 1, true));
    }

    private static Matrix forwardPropagate(Matrix input) {
        // input.sigmoid();
        activations = new ArrayList<>();
        Matrix activationsLocal = input.clone();
        activationsLocal.sigmoid();
        activations.add(activationsLocal);
        for (Matrix layer : network) {
            Matrix newActivations = activationsLocal.dot(layer);
            newActivations.sigmoid();
            activationsLocal = newActivations.clone();
            Model2.activations.add(activationsLocal.clone());
        }
        return activationsLocal;
    }

    private static ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        Matrix first = new Matrix(1, 1);
        // first.mat[0][0] = (expected.mat[0][0] - activations.get(1).mat[0][0]) * (expected.mat[0][0] - activations.get(1).mat[0][0]);
        first.mat[0][0] = 2 * (expected.mat[0][0] - activations.get(1).mat[0][0]);
        // System.out.println(first.mat[0][0]);
        first.mat[0][0] *= transferDerivative(activations.get(1).mat[0][0]);
        // System.out.println(" "+first.mat[0][0]);
        errors.add(first);

        Matrix second = new Matrix(1, 2);
        for (int curN = 0; curN < 2; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < 1; prevN++) {
                error += network.get(0).mat[curN][prevN] * errors.get(0).mat[0][prevN];
            }
            second.mat[0][curN] = error * transferDerivative(activations.get(0).mat[0][curN]);
        }
        errors.add(second);

        return errors;
    }

    private static void update(ArrayList<Matrix> errors) {
        for (int layer = 0; layer >= 0; layer--) {
            int eLayer = 1 - layer;
            for (int neuron = 0; neuron < errors.get(eLayer).cols; neuron++) {
                for (int weight = 0; weight < activations.get(layer + 1).cols; weight++) {
                    // System.out.println(layer + "," + neuron + "," + weight);
                    // network.get(layer).mat[neuron][weight] = network.get(layer).mat[neuron][weight];
                    network.get(layer).mat[neuron][weight] += learningRate * errors.get(eLayer).mat[0][neuron] * activations.get(layer + 1).mat[0][weight];
                }
            }
            //network.get(layer).mat[1][0] += learningRate * errors.get(eLayer).mat[0][1];
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

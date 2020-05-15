package core;

import java.util.ArrayList;

public class Model {
    private ArrayList<Integer> layerSizes;
    private ArrayList<Matrix> network;
    private ArrayList<Matrix> activations;
    private int layerCount;
    private double learningRate;

    public Model() {
        layerSizes = new ArrayList<>();
        network = new ArrayList<>();
    }

    public void addLayer(int layerSize) {
        layerSizes.add(layerSize);
    }

    public void buildModel(double learningRate) {
        for (int i = 0; i < layerSizes.size() - 1; i++) {
            network.add(new Matrix(layerSizes.get(i), layerSizes.get(i + 1), Math.sqrt(2.0 / layerSizes.get(i))));
        }
        this.learningRate = learningRate;
        layerCount = layerSizes.size();
    }

    public void fitSingle(Matrix input, Matrix expected) {
        forwardPropagate(input);
        ArrayList<Matrix> errors = backPropagate(expected);
        update(errors);
    }

    public Matrix predict(Matrix input) {
        return forwardPropagate(input);
    }

    private Matrix forwardPropagate(Matrix input) {
        activations = new ArrayList<>();
        Matrix activationsLocal = input.clone();
        activationsLocal.sigmoid();
        activations.add(activationsLocal.clone());
        for (Matrix layer : network) {
            Matrix newActivations = activationsLocal.dot(layer);
            newActivations.sigmoid();
            activationsLocal = newActivations.clone();
            activations.add(activationsLocal.clone());
        }
        return activations.get(activations.size() - 1);
    }

    private ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();

        Matrix first = new Matrix(1, 1);
        first.mat[0][0] = (expected.mat[0][0] - activations.get(layerCount - 1).mat[0][0]);
        first.mat[0][0] *= transferDerivative(activations.get(layerCount - 1).mat[0][0]);
        errors.add(first);

        for (int layer = layerCount - 2; layer >= 0; layer--) {
            Matrix curError = new Matrix(1, layerSizes.get(layer));
            for (int curN = 0; curN < layerSizes.get(layer); curN++) {
                double error = 0;
                for (int prevN = 0; prevN < layerSizes.get(layer + 1); prevN++) {
                    error += network.get(layer).mat[curN][prevN]
                            * errors.get(layerCount - 2 - layer).mat[0][prevN];
                }
                curError.mat[0][curN] = error * transferDerivative(activations.get(layer).mat[0][curN]);
            }
            errors.add(curError);
        }

        return errors;
    }

    private void update(ArrayList<Matrix> errors) {
        for (int layer = 0; layer < layerCount - 1; layer++) {
            int eLayer = layerCount - 2 - layer;
            for (int curN = 0; curN < activations.get(layer).cols; curN++) {
                for (int nextN = 0; nextN < activations.get(layer + 1).cols; nextN++) {
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

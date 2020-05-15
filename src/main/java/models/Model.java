package models;

import activations.Activation;
import activations.Sigmoid;
import core.Matrix;

import java.util.ArrayList;

public class Model {
    private ArrayList<Integer> layerSizes;
    private ArrayList<Matrix> network;
    private ArrayList<Matrix> neurons;
    private Activation activation = new Sigmoid();
    private int layerCount;
    private double learningRate;

    public Model() {
        layerSizes = new ArrayList<>();
        network = new ArrayList<>();
    }

    public Model addLayer(int layerSize) {
        layerSizes.add(layerSize);
        return this;
    }

    public void buildModel(Activation activation, double learningRate) {
        this.activation = activation;

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
        neurons = new ArrayList<>();
        Matrix activationsLocal = input.clone();
        activationsLocal.applyEach(activation.getActivation());
        neurons.add(activationsLocal.clone());
        for (int lNum = 0; lNum < network.size(); lNum++) {
            Matrix layer = network.get(lNum);
            Matrix newActivations = activationsLocal.dot(layer);
            newActivations.applyEach(activation.getActivation());
            activationsLocal = newActivations.clone();
            neurons.add(activationsLocal.clone());
        }
        return neurons.get(neurons.size() - 1);
    }

    private ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();

        Matrix first = new Matrix(1, 1);
        first.mat[0][0] = (expected.mat[0][0] - neurons.get(layerCount - 1).mat[0][0]);
        first.mat[0][0] *= activation.getTransferDerivative().applyAsDouble(neurons.get(layerCount - 1).mat[0][0]);
        errors.add(first);

        for (int layer = layerCount - 2; layer >= 0; layer--) {
            Matrix curError = new Matrix(1, layerSizes.get(layer));
            for (int curN = 0; curN < layerSizes.get(layer); curN++) {
                double error = 0;
                for (int prevN = 0; prevN < layerSizes.get(layer + 1); prevN++) {
                    error += network.get(layer).mat[curN][prevN]
                            * errors.get(layerCount - 2 - layer).mat[0][prevN];
                }
                curError.mat[0][curN] = error * activation.getTransferDerivative().applyAsDouble(neurons.get(layer).mat[0][curN]);
            }
            errors.add(curError);
        }

        return errors;
    }

    private void update(ArrayList<Matrix> errors) {
        for (int layer = 0; layer < layerCount - 1; layer++) {
            int eLayer = layerCount - 2 - layer;
            for (int curN = 0; curN < neurons.get(layer).cols; curN++) {
                for (int nextN = 0; nextN < neurons.get(layer + 1).cols; nextN++) {
                    network.get(layer).mat[curN][nextN] += learningRate * errors.get(eLayer).mat[0][nextN]
                            * (neurons.get(layer).mat[0][curN]);
                }
            }
        }
    }
}

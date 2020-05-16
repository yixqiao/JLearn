package models;

import activations.Activation;
import activations.Sigmoid;
import core.Matrix;

import java.util.ArrayList;

public class Model {
    private ArrayList<Integer> layerSizes;
    private ArrayList<Matrix> weights;
    private ArrayList<Matrix> biases;
    private ArrayList<Matrix> neurons;
    private Activation activation = new Sigmoid();
    private int layerCount;
    private double learningRate;

    public Model() {
        layerSizes = new ArrayList<>();
        weights = new ArrayList<>();
        biases = new ArrayList<>();
    }

    public Model addLayer(int layerSize) {
        layerSizes.add(layerSize);
        return this;
    }

    public void buildModel(Activation activation, double learningRate) {
        this.activation = activation;

        layerCount = layerSizes.size();

        for (int i = 0; i < layerCount - 1; i++) {
            weights.add(new Matrix(layerSizes.get(i), layerSizes.get(i + 1), Math.sqrt(2.0 / layerSizes.get(i))));
            biases.add(new Matrix(1, layerSizes.get(i + 1)));
        }
        this.learningRate = learningRate;

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
        // activationsLocal.applyEach(activation.getActivation());
        neurons.add(activationsLocal.clone());
        for (int layerNum = 0; layerNum < layerCount - 1; layerNum++) {
            Matrix layer = weights.get(layerNum);
            Matrix newActivations = activationsLocal.dot(layer);
            newActivations.addIP(biases.get(layerNum));
            newActivations.applyEach(activation.getActivation());
            activationsLocal = newActivations.clone();
            neurons.add(activationsLocal.clone());
        }
        return neurons.get(neurons.size() - 1);
    }

    private ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();

//        Matrix lastLayer = new Matrix(1, layerSizes.get(1));
//        lastLayer.mat[0][0] = (expected.mat[0][0] - neurons.get(layerCount - 1).mat[0][0]);
//        lastLayer.mat[0][0] *= activation.getTransferDerivative().applyAsDouble(neurons.get(layerCount - 1).mat[0][0]);
//        errors.add(lastLayer);

        for (int layer = layerCount - 1; layer >= 0; layer--) {
            Matrix curError = new Matrix(1, layerSizes.get(layer));
            if (layer == layerCount - 1) {
                for (int curN = 0; curN < layerSizes.get(layer); curN++) {
                    curError.mat[0][curN] = (expected.mat[0][curN] - neurons.get(layer).mat[0][curN]);
                    curError.mat[0][curN] *= activation.getTransferDerivative().applyAsDouble(neurons.get(layer).mat[0][curN]);
                }
            } else {
                for (int curN = 0; curN < layerSizes.get(layer); curN++) {
                    double error = 0;
                    for (int prevN = 0; prevN < layerSizes.get(layer + 1); prevN++) {
                        error += weights.get(layer).mat[curN][prevN]
                                * errors.get(layerCount - 2 - layer).mat[0][prevN];
                    }
                    curError.mat[0][curN] = error * activation.getTransferDerivative().applyAsDouble(neurons.get(layer).mat[0][curN]);
                }
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
                    weights.get(layer).mat[curN][nextN] += learningRate * errors.get(eLayer).mat[0][nextN]
                            * (neurons.get(layer).mat[0][curN]);
                }
            }
            biases.get(layer).addIP(errors.get(eLayer).multiply(learningRate));
        }
    }
}

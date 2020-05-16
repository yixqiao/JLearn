package models;

import activations.ElementwiseActivation;
import activations.Sigmoid;
import core.Matrix;

import java.util.ArrayList;
import java.util.Collections;

public class Model {
    private ArrayList<Integer> layerSizes;
    private ArrayList<Matrix> weights;
    private ArrayList<Matrix> biases;
    private ArrayList<Matrix> neurons;
    private ElementwiseActivation activation = new Sigmoid();
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

    public void buildModel(ElementwiseActivation activation, double learningRate) {
        this.activation = activation;

        layerCount = layerSizes.size();

        for (int i = 0; i < layerCount - 1; i++) {
            weights.add(new Matrix(layerSizes.get(i), layerSizes.get(i + 1), Math.sqrt(2.0 / layerSizes.get(i))));
            biases.add(new Matrix(1, layerSizes.get(i + 1)));
        }
        this.learningRate = learningRate;

    }

    public void fit(Matrix input, Matrix expected, int batchSize, int epochs) {
        int totalSamples = input.rows;
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) indices.add(i);

        ArrayList<Matrix> errors = null;

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(indices);
            for (int batchNum = 0; batchNum < totalSamples / batchSize; batchNum++) {
                Matrix batchInput = new Matrix(batchSize, input.cols);
                Matrix batchExpected = new Matrix(batchSize, expected.cols);
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < input.cols; j++) {
                        batchInput.mat[i][j] = input.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                    for (int j = 0; j < expected.cols; j++) {
                        batchExpected.mat[i][j] = expected.mat[indices.get(batchNum * batchSize + i)][j];
                    }
                }
                forwardPropagate(batchInput);
                errors = backPropagate(batchExpected);
                update(errors);
            }

            if ((epoch + 1) % (epochs / 10) == 0) {
                System.out.println(String.format("E: %d, L: %.5f", epoch + 1, getLoss(errors)));
            }
        }
    }

    public void fitSingle(Matrix input, Matrix expected) {
        forwardPropagate(input);
        ArrayList<Matrix> errors = backPropagate(expected);
        update(errors);
    }

    public Matrix predict(Matrix input) {
        return forwardPropagate(input);
    }

    // return np.exp(x) / np.sum(np.exp(x), axis=0)

    public Matrix forwardPropagate(Matrix input) {
        neurons = new ArrayList<>();
        Matrix activationsLocal = input.clone();
        // activationsLocal.applyEach(activation.getActivation());
        neurons.add(activationsLocal.clone());
        for (int layerNum = 0; layerNum < layerCount - 1; layerNum++) {
            Matrix layer = weights.get(layerNum);
            Matrix newActivations = activationsLocal.dot(layer);
            for (int row = 0; row < newActivations.rows; row++) {
                for (int col = 0; col < newActivations.cols; col++) {
                    newActivations.mat[row][col] += biases.get(layerNum).mat[0][col];
                }
            }
            if (layerNum == layerCount - 2) {
                // TODO: Stabilize (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
                Matrix exps = newActivations.applyEach(Math::exp);
                for (int row = 0; row < newActivations.rows; row++) {
                    double sum = 0;
                    for (int col = 0; col < newActivations.cols; col++)
                        sum += exps.mat[row][col];
                    for (int col = 0; col < newActivations.cols; col++) {
                        newActivations.mat[row][col] = exps.mat[row][col] / sum;
                    }
                }
            } else
                newActivations.applyEachIP(activation.getActivation());
            activationsLocal = newActivations.clone();
            neurons.add(activationsLocal.clone());
        }
        return neurons.get(neurons.size() - 1);
    }

    private ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();

        for (int layer = layerCount - 1; layer >= 0; layer--) {
            Matrix curError = new Matrix(1, layerSizes.get(layer));
            if (layer == layerCount - 1) {

                for (int curN = 0; curN < layerSizes.get(layer); curN++) {
                    for (int inputNum = 0; inputNum < expected.rows; inputNum++) {
                        curError.mat[0][curN] += (expected.mat[inputNum][curN] - neurons.get(layer).mat[inputNum][curN]);
                    }
                    curError.mat[0][curN] /= expected.rows;
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

    public double getLoss(ArrayList<Matrix> errors) {
        double loss = 0;
        for (int i = 0; i < errors.get(0).cols; i++) {
            loss += Math.abs(errors.get(0).mat[0][i]);
        }
        loss /= errors.get(0).cols;
        return loss;
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

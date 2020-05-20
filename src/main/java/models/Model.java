package models;

import activations.Activation;
import activations.ElementwiseActivation;
import activations.Sigmoid;
import activations.Softmax;
import core.Matrix;
import layers.Layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Model {
    private ArrayList<Layer> layers;
    private int layerCount;
    private double learningRate;

    public Model() {
        layers = new ArrayList<>();
    }

    public Model addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public void buildModel(double learningRate) {
        this.learningRate = learningRate;
        layerCount = layers.size();
        // TODO check that first layer is an input layer
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize());
        }
    }

    public void fit(Matrix input, Matrix expected, int batchSize, int epochs) {
        int totalSamples = input.rows;
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) indices.add(i);

        ArrayList<Matrix> errors;

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

            if ((epoch + 1) % (epochs / 20) == 0) {
                System.out.println(String.format("E: %d, L: %.5f", epoch + 1, getLoss(forwardPropagate(input), expected)));
            }
        }
    }

    public void trainOnBatch(Matrix input, Matrix expected) {
        forwardPropagate(input);
        ArrayList<Matrix> errors = backPropagate(expected);
        update(errors);
    }

    public Matrix predict(Matrix input) {
        return forwardPropagate(input);
    }

    // return np.exp(x) / np.sum(np.exp(x), axis=0)

    public Matrix forwardPropagate(Matrix input) {
        Matrix activations = input.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    private ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        for (int layer = layerCount - 1; layer >= 0; layer--) {
            if (layer == layerCount - 1) {
                errors.add(layers.get(layer).getErrorsExpected(expected));
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            } else {
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            }
        }
        return errors;
    }

    public double getLoss(Matrix input, Matrix expected) {
        ArrayList<Matrix> inputAL = new ArrayList<>();
        inputAL.add(input);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getLoss(inputAL, expectedAL);
    }

    public double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        double loss = 0;
        int total = 0;
        for (int inputNum = 0; inputNum < output.size(); inputNum++) {
            for (int row = 0; row < output.get(inputNum).rows; row++) {
                for (int col = 0; col < output.get(inputNum).cols; col++) {
                    if (expected.get(inputNum).mat[row][col] == 1)
                        loss += -Math.log(output.get(inputNum).mat[row][col]);
                    else
                        loss += -Math.log(1 - output.get(inputNum).mat[row][col]);
                    total++;
                }
            }
        }
        loss /= total;
        return loss;
    }

    private void update(ArrayList<Matrix> errors) {
        for (int layer = 0; layer < layerCount; layer++) {
//            System.out.println(errors.size() + ", " + layerCount);
            int eLayer = layerCount - 1 - layer;
            layers.get(layer).update(errors.get(eLayer), learningRate);
        }
    }

//    private void update(ArrayList<Matrix> errors) {
//        for (int layer = 0; layer < layerCount - 2; layer++) {
//            int eLayer = layerCount - 2 - layer;
//            for (int curN = 0; curN < neurons.get(layer).cols; curN++) {
//                for (int nextN = 0; nextN < neurons.get(layer + 1).cols; nextN++) {
//                    weights.get(layer).mat[curN][nextN] += learningRate * errors.get(eLayer).mat[0][nextN]
//                            * (neurons.get(layer).mat[0][curN]);
//                }
//            }
//            biases.get(layer).addIP(errors.get(eLayer).multiply(learningRate));
//        }
//    }
}

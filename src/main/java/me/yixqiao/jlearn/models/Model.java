package me.yixqiao.jlearn.models;

import me.yixqiao.jlearn.exceptions.NeuralNetworkException;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Layer;
import me.yixqiao.jlearn.losses.Loss;
import me.yixqiao.jlearn.metrics.Metric;

import java.util.ArrayList;
import java.util.Collections;

public class Model {
    private ArrayList<Layer> layers;
    private int layerCount;
    private Loss loss;

    /**
     * Create a new model.
     */
    public Model() {
        layers = new ArrayList<>();
    }

    /**
     * Add a layer to the model.
     *
     * @param layer the layer instance to add
     * @return the model itself to allow for daisy chaining
     */
    public Model addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Build the model. Run this after adding layers and before training.
     *
     * @param loss an instance of the loss function to use in the model
     */
    public void buildModel(Loss loss) {
        layerCount = layers.size();
        if (!(layers.get(0) instanceof InputLayer)) {
            throw new NeuralNetworkException("First layer is not an input layer");
        }
        for (int layer = 0; layer < layerCount - 1; layer++) {
            layers.get(layer + 1).initLayer(layers.get(layer).getOutSize());
        }
        this.loss = loss;
    }

    /**
     * Train the model on data.
     *
     * @param input input data to train on
     * @param expected expected outputs
     * @param learningRate learning rate of training
     * @param batchSize size of each minibatch
     * @param epochs number of epochs to train for
     * @param metric metric to display
     */
    public void fit(Matrix input, Matrix expected, double learningRate, int batchSize, int epochs, Metric metric) {
        fit(input, expected, learningRate, batchSize, epochs, 1, metric);
    }

    /**
     * Train the model on data.
     *
     * @param input input data to train on
     * @param expected expected outputs
     * @param learningRate learning rate of training
     * @param batchSize size of each minibatch
     * @param epochs number of epochs to train for
     * @param logInterval log every n epochs
     * @param metric metric to display
     */
    public void fit(Matrix input, Matrix expected, double learningRate, int batchSize, int epochs, int logInterval, Metric metric) {
        int totalSamples = input.rows;
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) indices.add(i);

        ArrayList<Matrix> errors;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double lossA = 0, metricA = 0;

            long epochStart = System.nanoTime();

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
                Matrix batchOutput = forwardPropagate(batchInput);
                if ((epoch + 1) % logInterval == 0) {
                    lossA += getLoss(loss, batchOutput, batchExpected);
                    metricA += getMetric(metric, batchOutput, batchExpected);
                }

                errors = backPropagate(batchExpected);
                update(errors, learningRate);
            }

            if ((epoch + 1) % logInterval == 0) {
                // Matrix output = forwardPropagate(input);
                lossA /= (double) (totalSamples / batchSize);
                metricA /= (double) (totalSamples / batchSize);
                double timeElapsed = (double) (System.nanoTime() - epochStart) / 1e9;
                System.out.println(String.format("E: %d, T: %.2fs, L: %.5f, A: %.1f%%",
                        epoch + 1, timeElapsed,
                        lossA, metricA * 100));
            }
        }
    }

    /**
     * Train on a single batch of input and output.
     *
     * @param input input data to train on
     * @param expected expected outputs
     * @param learningRate learning rate of training
     */
    public void trainOnBatch(Matrix input, Matrix expected, double learningRate) {
        forwardPropagate(input);
        ArrayList<Matrix> errors = backPropagate(expected);
        update(errors, learningRate);
    }

    /**
     * Predict on a batch of input.
     *
     * @param input input data to feed forward
     * @return prediction of model for input
     */
    public Matrix predict(Matrix input) {
        return forwardPropagate(input);
    }


    /**
     * Forward propagate a batch of input.
     *
     * @param input input data to feed forward
     * @return result of model for input
     */
    public Matrix forwardPropagate(Matrix input) {
        Matrix activations = input.clone();
        for (int layerNum = 0; layerNum < layerCount; layerNum++) {
            activations = layers.get(layerNum).forwardPropagate(activations);
        }
        return activations;
    }

    /**
     * Backpropagate model after forward propagating input.
     *
     * @param expected expected output for model
     * @return errors for each layer from backpropagation
     */
    public ArrayList<Matrix> backPropagate(Matrix expected) {
        ArrayList<Matrix> errors = new ArrayList<>();
        for (int layer = layerCount - 1; layer > 0; layer--) {
            if (layer == layerCount - 1) {
                errors.add(layers.get(layer).getErrorsExpected(expected));
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            } else {
                errors.add(layers.get(layer).getErrors(errors.get(errors.size() - 1)));
            }
        }
        return errors;
    }

    public double getLoss(Loss loss, Matrix output, Matrix expected) {
        return loss.getLoss(output, expected);
    }

    public double getLoss(Loss loss, ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        return loss.getLoss(output, expected);
    }

    public double getMetric(Metric metric, Matrix output, Matrix expected) {
        return metric.getMetric(output, expected);
    }

    public double getMetric(Metric metric, ArrayList<Matrix> input, ArrayList<Matrix> expected) {
        return metric.getMetric(input, expected);
    }

    private void update(ArrayList<Matrix> errors, double learningRate) {
        for (int layer = 1; layer < layerCount; layer++) {
            int eLayer = layerCount - layer - 1;
            layers.get(layer).update(errors.get(eLayer), learningRate);
        }
    }
}

package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.core.Matrix;

public class Dense extends Layer {
    private int inSize, outSize;
    private Matrix weights, biases;
    private Activation activation;
    private Matrix inputNeurons;
    private Matrix outputNeurons;

//    public Dense(int prevSize, int nextSize, Activation activation) {
//        this.prevSize = prevSize;
//        this.nextSize = nextSize;
//        this.activation = activation;
//        weights = new Matrix(prevSize, nextSize, Math.sqrt(2.0 / prevSize));
//        biases = new Matrix(1, nextSize);
//    }

    public Dense(int outSize, Activation activation) {
        this.outSize = outSize;
        this.activation = activation;
    }

    @Override
    public void initLayer(int inSize) {
        this.inSize = inSize;
        weights = new Matrix(inSize, outSize, Math.sqrt(2.0 / inSize));
        biases = new Matrix(1, outSize);
    }

    @Override
    public int getOutSize() {
        return outSize;
    }

    @Override
    public Matrix forwardPropagate(Matrix input) {
        inputNeurons = input.clone();
        Matrix output = input.dot(weights);
        for (int row = 0; row < output.rows; row++) {
            for (int col = 0; col < output.cols; col++) {
                output.mat[row][col] += biases.mat[0][col];
            }
        }
        activation.getActivation().accept(output);
        outputNeurons = output.clone();
        return output;
    }

    @Override
    public Matrix getErrors(Matrix prevErrors) {
        Matrix derivative = activation.getTransferDerivative().apply(inputNeurons);
        Matrix weightT = weights.getTranspose();
        Matrix errors = prevErrors.dot(weightT);
        for (int prevN = 0; prevN < inSize; prevN++) {
            errors.mat[0][prevN] *= derivative.mat[0][prevN];
        }
        return errors;
    }

    @Override
    public Matrix getErrorsExpected(Matrix expected) {
        Matrix curError = new Matrix(1, outSize);
        Matrix derivative = activation.getTransferDerivative().apply(outputNeurons);
        for (int outN = 0; outN < outSize; outN++) {
            for (int inputNum = 0; inputNum < expected.rows; inputNum++) {
                curError.mat[0][outN] += (expected.mat[inputNum][outN] - outputNeurons.mat[inputNum][outN]);
            }
            curError.mat[0][outN] /= expected.rows;
            curError.mat[0][outN] *= derivative.mat[0][outN];
        }
        return curError;
    }

    @Override
    public void update(Matrix errors, double learningRate) {
        for (int prevN = 0; prevN < inSize; prevN++) {
            for (int nextN = 0; nextN < outSize; nextN++) {
                weights.mat[prevN][nextN] += learningRate * errors.mat[0][nextN] * (inputNeurons.mat[0][prevN]);
            }
        }
        biases.addIP(errors.multiply(learningRate));
    }
}

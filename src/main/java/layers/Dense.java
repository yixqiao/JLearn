package layers;

import activations.Activation;
import core.Matrix;

import java.util.ArrayList;

public class Dense extends Layer {
    private int size, nextSize;
    private Matrix weights, biases;
    private Activation activation;

    public Dense(int size, int nextSize, Activation activation) {
        this.size = size;
        this.nextSize = nextSize;
        this.activation = activation;
        weights = new Matrix(size, nextSize, Math.sqrt(2.0 / size));
        biases = new Matrix(1, nextSize);
    }

    @Override
    public Matrix forwardPropagate(Matrix input) {
        Matrix output = input.dot(weights);
        for (int row = 0; row < output.rows; row++) {
            for (int col = 0; col < output.cols; col++) {
                output.mat[row][col] += biases.mat[0][col];
            }
        }
        activation.getActivation().accept(output);
        return output;
    }

    @Override
    public Matrix getErrors(Matrix neurons, Matrix expected, Matrix prevErrors) {
        Matrix curError = new Matrix(1, size);
        Matrix derivative = activation.getTransferDerivative().apply(neurons);
        for (int curN = 0; curN < size; curN++) {
            double error = 0;
            for (int prevN = 0; prevN < nextSize; prevN++) {
                error += weights.mat[curN][prevN]
                        * prevErrors.mat[0][prevN];
            }
            curError.mat[0][curN] = error * derivative.mat[0][curN];
        }
        return curError;
    }

    @Override
    public Matrix getErrors(Matrix neurons, Matrix expected) {
        Matrix curError = new Matrix(1, size);
        Matrix derivative = activation.getTransferDerivative().apply(neurons);
        for (int curN = 0; curN < size; curN++) {
            for (int inputNum = 0; inputNum < expected.rows; inputNum++) {
                curError.mat[0][curN] += (expected.mat[inputNum][curN] - neurons.mat[inputNum][curN]);
            }
            curError.mat[0][curN] /= expected.rows;
            curError.mat[0][curN] *= derivative.mat[0][curN];
        }
        return curError;
    }
}

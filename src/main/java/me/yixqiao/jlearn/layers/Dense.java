package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.activations.Sigmoid;
import me.yixqiao.jlearn.activations.Softmax;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.settings.Settings;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Basic fully-connected layer.
 */
public class Dense extends Layer {
    private int inSize, outSize;
    private Matrix weights, biases;
    private Activation activation, prevActivation;
    private Matrix inputNeurons;
    private Matrix outputNeurons;

    /**
     * Create a new dense layer.
     *
     * @param outSize    size of output
     * @param activation activation to use
     */

    public Dense(int outSize, Activation activation) {
        this.outSize = outSize;
        this.activation = activation;
    }

    @Override
    public void initLayer(int inSize, Activation prevActivation) {
        this.inSize = inSize;
        weights = new Matrix(inSize, outSize, new Matrix.Init.Gaussian(Math.sqrt(2.0 / inSize)));
        biases = new Matrix(1, outSize);
        this.prevActivation = prevActivation;
    }

    @Override
    public Activation getActivation() {
        return activation;
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
        Matrix derivative = prevActivation.getTransferDerivative().apply(inputNeurons);
        Matrix weightT = weights.getTranspose();
        Matrix errors = prevErrors.dot(weightT);
        for (int inputNum = 0; inputNum < prevErrors.rows; inputNum++) {
            for (int prevN = 0; prevN < inSize; prevN++) {
                errors.mat[inputNum][prevN] *= derivative.mat[inputNum][prevN];
            }
        }
        return errors;
    }

    @Override
    public Matrix getErrorsExpected(Matrix expected) {
        Matrix curError = new Matrix(expected.rows, outSize);
        Matrix derivative = activation.getTransferDerivative().apply(outputNeurons);
        for (int inputNum = 0; inputNum < expected.rows; inputNum++) {
            for (int outN = 0; outN < outSize; outN++) {
                curError.mat[inputNum][outN] += (expected.mat[inputNum][outN] - outputNeurons.mat[inputNum][outN]);
                curError.mat[inputNum][outN] *= derivative.mat[0][outN];
            }
        }
        return curError;
    }

    @Override
    public void update(Matrix errors, double learningRate) {
        Matrix bChanges = new Matrix(1, errors.cols);

        if (inSize * outSize >= Settings.THREADING_MIN_OPS) {
            // System.out.println(inSize * outSize + ", " + errors.rows);

            class CalcInput implements Runnable {
                private final int inputNum;

                public CalcInput(int inputNum) {
                    this.inputNum = inputNum;
                }

                public void run() {
                    for (int prevN = 0; prevN < inSize; prevN++) {
                        for (int nextN = 0; nextN < outSize; nextN++) {
                            weights.mat[prevN][nextN] += (learningRate / errors.rows) * errors.mat[inputNum][nextN]
                                    * (inputNeurons.mat[inputNum][prevN]);
                        }
                    }
                }
            }

            ExecutorService pool = Executors.newFixedThreadPool(Settings.THREAD_COUNT);

            for (int inputNum = 0; inputNum < errors.rows; inputNum++) {
                Runnable rn = new CalcInput(inputNum);
                pool.execute(rn);
            }

            pool.shutdown();

            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        } else {
            for (int inputNum = 0; inputNum < errors.rows; inputNum++) {
                for (int prevN = 0; prevN < inSize; prevN++) {
                    for (int nextN = 0; nextN < outSize; nextN++) {
                        weights.mat[prevN][nextN] += (learningRate / errors.rows) * errors.mat[inputNum][nextN]
                                * (inputNeurons.mat[inputNum][prevN]);
                    }
                }
            }
        }

        for (int inputNum = 0; inputNum < errors.rows; inputNum++) {
            for (int nextN = 0; nextN < outSize; nextN++) {
                bChanges.mat[0][nextN] += errors.mat[inputNum][nextN];
            }
        }
        bChanges.applyEachIP(x -> x / errors.rows);
        biases.addIP(bChanges.multiply(learningRate));
    }

    @Override
    public String toString() {
        return String.format("Dense: in: %d, out: %d, activation: %s", inSize, outSize, activation.toString());
    }
}

package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.initializers.He;
import me.yixqiao.jlearn.initializers.Initializer;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.settings.JLSettings;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Basic fully-connected layer.
 */
public class Dense extends Layer {
    /**
     * Weights.
     */
    public Matrix weights;
    /**
     * Biases.
     */
    public Matrix biases;
    private int inSize, outSize;
    private Activation activation, prevActivation;
    private Matrix inputNeurons;
    private Matrix outputNeurons;
    private Initializer init = new He();

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

    /**
     * Create a new dense layer.
     *
     * @param outSize    size of output
     * @param activation activation to use
     * @param init       initialization method
     */
    public Dense(int outSize, Activation activation, Initializer init) {
        this(outSize, activation);
        this.init = init;
    }

    @Override
    public void initLayer(int inSize, Activation prevActivation) {
        this.inSize = inSize;
        weights = new Matrix(inSize, outSize, init.getInit(inSize, outSize));
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
    public Matrix forwardPropagate(Matrix x) {
        inputNeurons = x.cloneMatrix();

        Matrix output = x.dot(weights);
        for (int row = 0; row < output.rows; row++) {
            for (int col = 0; col < output.cols; col++) {
                output.mat[row][col] += biases.mat[0][col];
            }
        }
        activation.getActivation().accept(output);
        outputNeurons = output.cloneMatrix();

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
    public Matrix getErrorsExpected(Matrix y) {
        Matrix curError = new Matrix(y.rows, outSize);
        Matrix derivative = activation.getTransferDerivative().apply(outputNeurons);
        for (int inputNum = 0; inputNum < y.rows; inputNum++) {
            for (int outN = 0; outN < outSize; outN++) {
                curError.mat[inputNum][outN] += (y.mat[inputNum][outN] - outputNeurons.mat[inputNum][outN]);
                curError.mat[inputNum][outN] *= derivative.mat[0][outN];
            }
        }
        return curError;
    }

    @Override
    public void update(Matrix errors, double learningRate) {
        Matrix bChanges = new Matrix(1, errors.cols);

        if (inSize * outSize >= JLSettings.THREADING_MIN_OPS) {
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

            ExecutorService pool = Executors.newFixedThreadPool(JLSettings.THREAD_COUNT);

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

    @Override
    public Layer cloneSettings() {
        Dense clone = new Dense(outSize, activation);
        clone.initLayer(inSize, prevActivation);
        return clone;
    }

    @Override
    public Layer cloneLayer() {
        Dense clone = new Dense(outSize, activation);
        clone.initLayer(inSize, prevActivation);
        clone.weights = weights.cloneMatrix();
        clone.biases = biases.cloneMatrix();
        return clone;
    }
}

package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.activations.Linear;
import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Layer for input.
 * <p>
 * All functions are no-ops.
 * </p>
 */
public class InputLayer extends Layer {
    private final int size;

    /**
     * Create an input layer.
     *
     * @param size size of the input
     */
    public InputLayer(int size) {
        this.size = size;
    }

    /**
     * No op.
     *
     * @param prevSize size of the previous layer (can be anything)
     */
    @Override
    public void initLayer(int prevSize, Activation prevActivation) {

    }

    /**
     * Return a linear activation.
     *
     * @return a new linear activation
     */
    @Override
    public Activation getActivation() {
        return new Linear();
    }


    @Override
    public int getOutSize() {
        return size;
    }

    /**
     * No op forward propagation.
     *
     * @param input input matrix
     * @return the same input matrix
     */
    @Override
    public Matrix forwardPropagate(Matrix input) {
        return input;
    }

    /**
     * No op.
     *
     * @param prevErrors errors from previous layer (layer after output)
     * @return null
     */
    @Override
    public Matrix getErrors(Matrix prevErrors) {
        return null;
    }

    /**
     * No op.
     *
     * @param expected expected outputs
     * @return null
     */
    @Override
    public Matrix getErrorsExpected(Matrix expected) {
        return null;
    }

    /**
     * No op. Nothing is updated.
     *
     * @param errors       calculated errors
     * @param learningRate learning rate of changes
     */
    @Override
    public void update(Matrix errors, double learningRate) {

    }
}

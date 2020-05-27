package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Abstract layer class.
 */
public abstract class Layer {
    /**
     * Create the layer.
     *
     * @param prevSize size of the previous layer
     */
    public abstract void initLayer(int prevSize);

    /**
     * Get the output size of the layer.
     *
     * @return the output size
     */
    public abstract int getOutSize();

    /**
     * Forward propagate a batch of input.
     *
     * @param input input matrix
     * @return output matrix
     */
    public abstract Matrix forwardPropagate(Matrix input);

    /**
     * Get backpropagated errors
     *
     * @param prevErrors errors from previous layer (layer after output)
     * @return matrix of errors
     */
    public abstract Matrix getErrors(Matrix prevErrors);

    /**
     * Get errors from output layer.
     *
     * @param expected expected outputs
     * @return matrix of errors
     */
    public abstract Matrix getErrorsExpected(Matrix expected);

    /**
     * Update the layer after calculating errors.
     *
     * @param errors calculated errors
     * @param learningRate learning rate of changes
     */
    public abstract void update(Matrix errors, double learningRate);
}

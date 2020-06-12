package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.optimizers.Optimizer;

import java.io.Serializable;

/**
 * Abstract layer class.
 */
public abstract class Layer implements Serializable {
    /**
     * Create the layer.
     *
     * @param prevSize       size of the previous layer
     * @param prevActivation activation function or previous layer
     */
    public abstract void initLayer(int prevSize, Activation prevActivation);

    public abstract void setOptimizers(Optimizer wOptimizer, Optimizer bOptimizer);

    /**
     * Get the activation of the layer.
     *
     * @return the activation object
     */
    public abstract Activation getActivation();

    /**
     * Get the output size of the layer.
     *
     * @return the output size
     */
    public abstract int getOutSize();

    /**
     * Forward propagate a batch of input.
     *
     * @param x input matrix
     * @return output matrix
     */
    public abstract Matrix forwardPropagate(Matrix x);

    /**
     * Get backpropagated errors.
     *
     * @param prevErrors errors from previous layer (layer after output)
     * @return matrix of errors
     */
    public abstract Matrix getErrors(Matrix prevErrors);

    /**
     * Get errors from output layer.
     *
     * @param y expected outputs
     * @return matrix of errors
     */
    public abstract Matrix getErrorsExpected(Matrix y);

    /**
     * Update the layer after calculating errors.
     *
     * @param errors       calculated errors
     */
    public abstract void update(Matrix errors);

    /**
     * Get a string representation.
     *
     * @return the string
     */
    public abstract String toString();

    /**
     * Clone the settings.
     * <p>
     * This will return a layer with the same size and activation, but with randomly initialized weights and biases.
     * </p>
     *
     * @return the clone
     */
    public abstract Layer cloneSettings();

    /**
     * Clone the layer, including weights and biases.
     *
     * @return the clone
     */
    public abstract Layer cloneLayer();
}

package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.matrix.Matrix;

import java.io.Serializable;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Abstract activation function.
 */
public abstract class Activation implements Serializable {

    /**
     * Get a function that will apply the activation.
     *
     * @return the function
     */
    public abstract Consumer<Matrix> getActivation();

    /**
     * Get a function that will transfer derivative onto an already-activated matrix.
     *
     * @return the function
     */
    public abstract Function<Matrix, Matrix> getTransferDerivative();

    public abstract String toString();
}

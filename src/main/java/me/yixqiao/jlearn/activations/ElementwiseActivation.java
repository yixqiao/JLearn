package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * Activation that is applied across all numbers the same.
 */
public abstract class ElementwiseActivation extends Activation {
    /**
     * Returns a function that applies on each element.
     *
     * @return the function
     */
    public Consumer<Matrix> getActivation() {
        return x -> x.applyEachIP(getEActivation(), false);
    }

    /**
     * Return a function that will transfer derivative on each element.
     *
     * @return the function
     */
    public Function<Matrix, Matrix> getTransferDerivative() {
        return x -> x.applyEach(getETransferDerivative(), false);
    }

    /**
     * Get the function that will be applied to each element.
     *
     * @return the function
     */
    public abstract ToDoubleFunction<Double> getEActivation();

    /**
     * Return the function to transfer derivative that will be applied to each element.
     *
     * @return the function
     */
    public abstract ToDoubleFunction<Double> getETransferDerivative();
}

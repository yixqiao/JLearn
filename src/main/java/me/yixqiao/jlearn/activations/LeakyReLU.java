package me.yixqiao.jlearn.activations;

import java.util.function.ToDoubleFunction;

/**
 * Leaky Rectified Linear Unit activation.
 */

public class LeakyReLU extends ElementwiseActivation {
    private final double alpha;

    /**
     * Initialize the activation.
     *
     * @param alpha slope when x is below 0
     */
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> (x <= 0 ? x * alpha : x);
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> (x <= 0 ? alpha : 1);
    }
}

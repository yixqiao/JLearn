package me.yixqiao.jlearn.activations;

import java.util.function.ToDoubleFunction;

/**
 * Rectified Linear Unit activation.
 */
public class ReLU extends ElementwiseActivation {
    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> Math.max(0, x);
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> (x <= 0 ? 0 : 1);
    }

    @Override
    public String toString() {
        return "ReLU";
    }
}

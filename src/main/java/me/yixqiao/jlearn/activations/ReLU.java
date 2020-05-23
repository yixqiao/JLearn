package me.yixqiao.jlearn.activations;

import java.util.function.ToDoubleFunction;

public class ReLU extends ElementwiseActivation {
    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> Math.max(0, x);
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> (x <= 0 ? 0 : 1);
    }
}

package me.yixqiao.jlearn.activations;

import java.util.function.ToDoubleFunction;

public class LeakyReLU extends ElementwiseActivation {
    double alpha;

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

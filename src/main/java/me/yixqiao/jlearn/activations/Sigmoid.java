package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public class Sigmoid extends ElementwiseActivation {
    @Override
    public Function<Matrix, Matrix> getTransferDerivative() {
        return x -> x.applyEach(getETransferDerivative());
    }

    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> x * (1 - x);
    }
}

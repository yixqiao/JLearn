package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.core.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public abstract class ElementwiseActivation extends Activation {
    public Consumer<Matrix> getActivation() {
        return x -> x.applyEachIP(getEActivation());
    }

    public Function<Matrix, Matrix> getTransferDerivative() {
        return x -> x.applyEach(getETransferDerivative());
    }

    public abstract ToDoubleFunction<Double> getEActivation();

    public abstract ToDoubleFunction<Double> getETransferDerivative();
}
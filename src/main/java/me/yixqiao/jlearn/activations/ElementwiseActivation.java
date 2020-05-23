package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public abstract class ElementwiseActivation extends Activation {
    public Consumer<Matrix> getActivation() {
        return x -> x.applyEachIP(getEActivation(), false);
    }

    public Function<Matrix, Matrix> getTransferDerivative() {
        return x -> x.applyEach(getETransferDerivative(), false);
    }

    public abstract ToDoubleFunction<Double> getEActivation();

    public abstract ToDoubleFunction<Double> getETransferDerivative();
}

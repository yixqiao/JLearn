package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.core.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;

public abstract class Activation {
    public abstract Consumer<Matrix> getActivation();

    public abstract Function<Matrix, Matrix> getTransferDerivative();
}

package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.matrix.Matrix;

public abstract class Layer {
    public abstract void initLayer(int prevSize, Activation prevActivation);

    public abstract Activation getActivation();

    public abstract int getOutSize();

    public abstract Matrix forwardPropagate(Matrix input);

    public abstract Matrix getErrors(Matrix prevErrors);

    public abstract Matrix getErrorsExpected(Matrix expected);

    public abstract void update(Matrix errors, double learningRate);
}

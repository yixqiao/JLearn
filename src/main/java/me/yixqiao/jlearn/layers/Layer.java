package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.core.Matrix;

public abstract class Layer {
    public abstract void initLayer(int prevSize);

    public abstract int getOutSize();

    public abstract Matrix forwardPropagate(Matrix input);

    public abstract Matrix getErrors(Matrix prevErrors);

    public abstract Matrix getErrorsExpected(Matrix expected);

    public abstract void update(Matrix errors, double learningRate);
}

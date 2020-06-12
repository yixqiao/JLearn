package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

public abstract class Optimizer {
    public abstract Matrix apply(Matrix g);

    public abstract void multiplyLR(double d);

    public abstract Optimizer cloneOptimizer();
}

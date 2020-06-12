package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

public class SGD extends Optimizer {
    double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Optimizer cloneOptimizer() {
        return new SGD(learningRate);
    }

    @Override
    public void multiplyLR(double d) {
        this.learningRate *= d;
    }

    @Override
    public Matrix apply(Matrix g) {
        g.multiplyIP(learningRate);
        return g;
    }
}
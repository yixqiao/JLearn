package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Gradient descent optimizer.
 */
public class SGD extends Optimizer {
    double learningRate;

    /**
     * Create a new instance.
     *
     * @param learningRate learning rate
     */
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Optimizer cloneSettings() {
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
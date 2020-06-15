package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Gradient descent with momentum.
 */
public class Momentum extends Optimizer {
    double learningRate;
    double momentum;
    boolean started = false;
    Matrix velocity;

    /**
     * Create a new instance.
     *
     * @param learningRate learning rate
     * @param momentum     momentum (0-1, higher value means more momentum)
     */
    public Momentum(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public Optimizer cloneSettings() {
        return new Momentum(learningRate, momentum);
    }

    @Override
    public void multiplyLR(double d) {
        this.learningRate *= d;
    }

    private void init(Matrix g) {
        velocity = new Matrix(g.rows, g.cols);
        started = true;
    }

    @Override
    public Matrix apply(Matrix g) {
        if (!started)
            init(g);
        velocity.multiplyIP(momentum);
        velocity.addIP(g.multiply(learningRate));
        return velocity;
    }
}

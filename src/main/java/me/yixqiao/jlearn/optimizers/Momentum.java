package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

public class Momentum extends Optimizer {
    double learningRate;
    double momentum;
    boolean started = false;
    Matrix velocity;

    public Momentum(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
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
//
// velocity = momentum * velocity - learning_rate * g
//         w = w * velocity
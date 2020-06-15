package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Adam optimizer.
 */
public class Adam extends Optimizer {
    double learningRate;
    double beta1 = 0.9, beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;
    Matrix m, v;

    /**
     * Initialize the optimizer.
     *
     * @param learningRate learning rate
     */
    public Adam(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Initialize the optimizer.
     *
     * @param learningRate learning rate
     * @param beta1 beta 1 (default is 0.9)
     * @param beta2 beta 2 (default is 0.999)
     */
    public Adam(double learningRate, double beta1, double beta2) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public Optimizer cloneSettings() {
        return new Adam(learningRate, beta1, beta2);
    }

    @Override
    public void multiplyLR(double d) {
        this.learningRate *= d;
    }

    private void init(Matrix g) {
        m = new Matrix(g.rows, g.cols);
        v = new Matrix(g.rows, g.cols);
    }

    @Override
    public Matrix apply(Matrix g) {
        if (t == 0)
            init(g);
        t++;

        m.multiplyIP(beta1);
        m.addIP(g.multiply(1 - beta1));

        v.multiplyIP(beta2);
        v.addIP(g.applyEach(x -> x * x).multiply(1 - beta2));

        Matrix mc = m.multiply(1.0 / (1 - Math.pow(beta1, t)));
        Matrix vc = v.multiply(1.0 / (1 - Math.pow(beta2, t)));

        Matrix out = mc.multiply(learningRate);
        vc.applyEachIP(Math::sqrt);
        vc.applyEachIP(x -> x + epsilon);
        out.divideIP(vc);

        return out;
    }
}

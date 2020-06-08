package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Gaussian distribution initialization.
 */
public class GaussianInit extends Initializer {
    private final double deviation;

    /**
     * Create a new instance.
     *
     * @param deviation standard deviation
     */
    public GaussianInit(double deviation) {
        this.deviation = deviation;
    }

    @Override
    public Matrix.Init getInit(int inSize, int outSize) {
        return new Matrix.Init.Gaussian(0, deviation);
    }
}

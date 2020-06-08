package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

public class GaussianInit extends Initializer {
    private double deviation;

    public GaussianInit(double deviation) {
        this.deviation = deviation;
    }

    @Override
    public Matrix.Init getInit(int inSize, int outSize) {
        return new Matrix.Init.Gaussian(0, deviation);
    }
}

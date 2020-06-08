package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

public class Xavier extends Initializer {
    @Override
    public Matrix.Init getInit(int inSize, int outSize) {
        return new Matrix.Init.Gaussian(0, Math.sqrt(1.0 / inSize));
    }
}

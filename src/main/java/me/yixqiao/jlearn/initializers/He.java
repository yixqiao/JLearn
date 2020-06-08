package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

public class He extends Initializer {
    @Override
    public Matrix.Init getInit(int inSize, int outSize) {
        return new Matrix.Init.Gaussian(0, Math.sqrt(2.0 / inSize));
    }
}

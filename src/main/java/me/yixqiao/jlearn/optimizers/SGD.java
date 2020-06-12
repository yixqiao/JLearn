package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

public class SGD extends Optimizer {
    @Override
    public Matrix apply(Matrix g) {
        return g;
    }
}

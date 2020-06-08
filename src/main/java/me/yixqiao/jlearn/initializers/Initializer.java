package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

public abstract class Initializer {
    public abstract Matrix.Init getInit(int inSize, int outSize);
}

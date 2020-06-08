package me.yixqiao.jlearn.initializations;

import me.yixqiao.jlearn.matrix.Matrix;

public abstract class Initialization {
    public abstract Matrix.Init getInit(int inSize, int outSize);
}

package me.yixqiao.jlearn.initializers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Abstract initializer class.
 */
public abstract class Initializer {
    /**
     * Get the initialization method.
     *
     * @param inSize input size of layer
     * @param outSize output size of layer
     * @return an matrix initialization instance
     */
    public abstract Matrix.Init getInit(int inSize, int outSize);
}

package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Abstract class for optimizers.
 */
public abstract class Optimizer {
    /**
     * Apply the optimizer.
     *
     * @param g errors calculated from backpropagation.
     * @return the gradients from the optimizer
     */
    public abstract Matrix apply(Matrix g);

    /**
     * Multiply the learning rate.
     * <p>
     * Used to scale the learning rate in different batch sizes.
     * </p>
     *
     * @param d amount
     */
    public abstract void multiplyLR(double d);

    /**
     * Return a clone of this optimizer's settings.
     *
     * @return the clone
     */
    public abstract Optimizer cloneSettings();
}

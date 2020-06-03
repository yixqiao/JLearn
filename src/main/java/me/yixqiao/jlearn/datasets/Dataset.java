package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Basic input output dataset.
 */
public class Dataset {
    /**
     * Input data.
     */
    public final Matrix x;
    /**
     * Output data.
     */
    public final Matrix y;

    /**
     * Create a new dataset.
     *
     * @param x input
     * @param y output
     */
    public Dataset(Matrix x, Matrix y) {
        this.x = x;
        this.y = y;
    }
}

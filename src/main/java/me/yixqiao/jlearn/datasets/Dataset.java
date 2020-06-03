package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

public class Dataset {
    public final Matrix x;
    public final Matrix y;

    public Dataset(Matrix x, Matrix y) {
        this.x = x;
        this.y = y;
    }
}

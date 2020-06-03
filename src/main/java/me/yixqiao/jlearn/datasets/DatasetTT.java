package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

public class DatasetTT {
    public final Dataset train;
    public final Dataset test;

    public DatasetTT(Dataset train, Dataset test) {
        this.train = train;
        this.test = test;
    }

    public DatasetTT(Matrix trainX, Matrix trainY, Matrix testX, Matrix testY) {
        this.train = new Dataset(trainX, trainY);
        this.test = new Dataset(testX, testY);
    }
}

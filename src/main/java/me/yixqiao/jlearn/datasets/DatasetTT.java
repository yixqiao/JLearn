package me.yixqiao.jlearn.datasets;

import me.yixqiao.jlearn.matrix.Matrix;

/**
 * Dataset with a train and test.
 */
public class DatasetTT {
    /**
     * Train data.
     */
    public final Dataset train;
    /**
     * Test data.
     */
    public final Dataset test;

    /**
     * Create a new dataset.
     *
     * @param train train data
     * @param test test data
     */
    public DatasetTT(Dataset train, Dataset test) {
        this.train = train;
        this.test = test;
    }

    /**
     * Create a new dataset.
     *
     * @param trainX train input
     * @param trainY train output
     * @param testX test input
     * @param testY test output
     */
    public DatasetTT(Matrix trainX, Matrix trainY, Matrix testX, Matrix testY) {
        this.train = new Dataset(trainX, trainY);
        this.test = new Dataset(testX, testY);
    }
}

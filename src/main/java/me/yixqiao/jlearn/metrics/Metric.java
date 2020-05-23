package me.yixqiao.jlearn.metrics;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

public abstract class Metric {
    public abstract double getMetric (ArrayList<Matrix> output, ArrayList<Matrix> expected);

    public double getMetric(Matrix output, Matrix expected) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(output);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getMetric(outputAL, expectedAL);
    }
}

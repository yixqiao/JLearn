package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.metrics.Metric;

import java.util.ArrayList;

public abstract class Loss extends Metric {
    public double getMetric(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        return getLoss(output, expected);
    }

    public String getFormatString() {
        return "L: %d";
    }

    public abstract double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected);

    public double getLoss(Matrix output, Matrix expected) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(output);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getLoss(outputAL, expectedAL);
    }
}

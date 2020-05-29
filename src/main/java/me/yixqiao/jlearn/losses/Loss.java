package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.metrics.Metric;

import java.util.ArrayList;

/**
 * Class for loss function.
 */
public abstract class Loss extends Metric {
    @Override
    public double getMetric(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        return getLoss(output, expected);
    }

    @Override
    public String getFormatString() {
        return "L: %.4f";
    }

    /**
     * Get the loss.
     *
     * @param output output of network
     * @param expected correct output
     * @return the loss
     */
    public abstract double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected);

    /**
     * Get the loss.
     *
     * @param output output of network
     * @param expected correct output
     * @return the loss
     */
    public double getLoss(Matrix output, Matrix expected) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(output);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getLoss(outputAL, expectedAL);
    }
}

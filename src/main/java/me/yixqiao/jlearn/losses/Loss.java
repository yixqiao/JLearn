package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.metrics.Metric;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Class for loss function.
 */
public abstract class Loss extends Metric implements Serializable {
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
     * @param out output of network
     * @param y   correct output
     * @return the loss
     */
    public abstract double getLoss(ArrayList<Matrix> out, ArrayList<Matrix> y);

    /**
     * Get the loss.
     *
     * @param out output of network
     * @param y   correct output
     * @return the loss
     */
    public double getLoss(Matrix out, Matrix y) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(out);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(y);
        return getLoss(outputAL, expectedAL);
    }
}

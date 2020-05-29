package me.yixqiao.jlearn.metrics;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

/**
 * Class for a metric.
 */
public abstract class Metric {
    /**
     * Get the value of the metric.
     *
     * @param output output of network
     * @param expected expected output
     * @return the metric value
     */
    public abstract double getMetric (ArrayList<Matrix> output, ArrayList<Matrix> expected);

    /**
     * Get the string to format the metric.
     * <p>
     *     Return a string that can be used with printf in conjunction with the metric.
     *     <br>
     *     Example: <code>"M: %.2f"</code>
     * </p>
     *
     * @return the string
     */
    public abstract String getFormatString();

    /**
     * Get the value of the metric.
     *
     * @param output output of network
     * @param expected expected output
     * @return the metric value
     */
    public double getMetric(Matrix output, Matrix expected) {
        ArrayList<Matrix> outputAL = new ArrayList<>();
        outputAL.add(output);
        ArrayList<Matrix> expectedAL = new ArrayList<>();
        expectedAL.add(expected);
        return getMetric(outputAL, expectedAL);
    }
}

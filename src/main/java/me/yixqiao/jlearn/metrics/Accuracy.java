package me.yixqiao.jlearn.metrics;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

public class Accuracy extends Metric {
    private boolean formatPercent;

    public Accuracy() {
        this.formatPercent = true;
    }

    public Accuracy(boolean formatPercent) {
        this.formatPercent = formatPercent;
    }

    @Override
    public double getMetric(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        int correct = 0, total = 0;
        for (int batchNum = 0; batchNum < output.size(); batchNum++) {
            for (int row = 0; row < output.get(batchNum).rows; row++) {
                int maxIO = -1, maxIE = -1;
                double maxVO = -Double.MAX_VALUE, maxVE = -Double.MAX_VALUE;
                for (int col = 0; col < output.get(batchNum).cols; col++) {
                    if (output.get(batchNum).mat[row][col] > maxVO) {
                        maxVO = output.get(batchNum).mat[row][col];
                        maxIO = col;
                    }
                    if (expected.get(batchNum).mat[row][col] > maxVE) {
                        maxVE = output.get(batchNum).mat[row][col];
                        maxIE = col;
                    }
                }
                if (maxIO == maxIE)
                    correct++;
                total++;
            }
        }
        return formatPercent ? ((double) (correct) / total * 100) : ((double) (correct) / total);
    }

    @Override
    public String getFormatString() {
        return formatPercent ? "A: %.2f%%" : "A: %.3f";
    }
}

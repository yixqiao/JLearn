package metrics;

import core.Matrix;

import java.util.ArrayList;

public class Accuracy extends Metric {

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
        return (double) (correct) / total;
    }
}

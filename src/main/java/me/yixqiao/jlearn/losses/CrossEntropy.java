package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

/**
 * Cross entropy loss.
 * <p>
 * For now, only for use with the softmax activation.
 * </p>
 */
public class CrossEntropy extends Loss {
    @Override
    public double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        double loss = 0;
        int total = 0;
        for (int batchNum = 0; batchNum < output.size(); batchNum++) {
            for (int row = 0; row < output.get(batchNum).rows; row++) {
                for (int col = 0; col < output.get(batchNum).cols; col++) {
                    if (expected.get(batchNum).mat[row][col] == 1)
                        loss += -Math.log(output.get(batchNum).mat[row][col]);
                    else
                        loss += -Math.log(1 - output.get(batchNum).mat[row][col]);
                    total++;
                }
            }
        }
        loss /= total;
        return loss;
    }
}

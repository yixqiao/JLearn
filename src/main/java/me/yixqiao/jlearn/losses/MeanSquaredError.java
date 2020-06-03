package me.yixqiao.jlearn.losses;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.ArrayList;

/**
 * Mean squared error loss.
 */
public class MeanSquaredError extends Loss {
    @Override
    public double getLoss(ArrayList<Matrix> out, ArrayList<Matrix> y) {
        // return np.sum((yHat - y)**2) / y.size
        double loss = 0;
        for (int i = 0; i < out.size(); i++) {
            Matrix o = out.get(i), e = y.get(i);
            Matrix diff = o.subtract(e);
            diff.applyEachIP(x -> x * x);
            double sum = diff.getSum();
            loss += sum / (o.rows * o.cols);
        }
        return loss;
    }
}

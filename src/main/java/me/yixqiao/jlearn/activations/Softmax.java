package me.yixqiao.jlearn.activations;

import me.yixqiao.jlearn.matrix.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;

public class Softmax extends Activation {
    private final double epsilon = Double.MIN_VALUE;

    @Override
    public Consumer<Matrix> getActivation() {
        // Stabilized (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
        return x -> {
            double max = x.getMaxValue();
            Matrix shiftx = x.applyEach(xd -> (xd - max));
            // shiftx.printMatrix();
            Matrix exps = shiftx.applyEach(Math::exp);
            // exps.printMatrix();
            for (int row = 0; row < x.rows; row++) {
                double sum = epsilon;
                for (int col = 0; col < x.cols; col++)
                    sum += exps.mat[row][col];
                for (int col = 0; col < x.cols; col++) {
                    x.mat[row][col] = exps.mat[row][col] / sum;
                }
            }
        };
    }

    @Override
    public Function<Matrix, Matrix> getTransferDerivative() {
        return x -> x.applyEach(xd -> 1, false);
        // return x -> x.applyEach(xd -> xd * (1 - xd));
    }
}

package losses;

import core.Matrix;

import java.util.ArrayList;

public class MeanSquaredError extends Loss {
    @Override
    public double getLoss(ArrayList<Matrix> output, ArrayList<Matrix> expected) {
        // return np.sum((yHat - y)**2) / y.size
        double loss = 0;
        for (int i = 0; i < output.size(); i++) {
            Matrix o = output.get(i), e = expected.get(i);
            Matrix diff = o.subtract(e);
            diff.applyEachIP(x -> x * x);
            double sum = diff.sum();
            loss += sum / (o.rows * o.cols);
        }
        return loss;
    }
}

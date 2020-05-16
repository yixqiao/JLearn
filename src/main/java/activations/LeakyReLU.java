package activations;

import java.util.function.ToDoubleFunction;

public class LeakyReLU extends Activation {
    double alpha;

    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public ToDoubleFunction<Double> getActivation() {
        return x -> (x <= 0 ? x * alpha : x);
    }

    @Override
    public ToDoubleFunction<Double> getTransferDerivative() {
        return x -> (x <= 0 ? alpha : 1);
    }
}

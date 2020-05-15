package activations;

import java.util.function.ToDoubleFunction;

public class Sigmoid extends Activation {
    @Override
    public ToDoubleFunction<Double> getActivation() {
        return x -> 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public ToDoubleFunction<Double> getTransferDerivative() {
        return x -> x * (1 - x);
    }
}

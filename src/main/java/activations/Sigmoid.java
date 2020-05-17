package activations;

import java.util.function.ToDoubleFunction;

public class Sigmoid extends ElementwiseActivation {
    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> x * (1 - x);
    }
}

package activations;

import java.util.function.ToDoubleFunction;

public class Linear extends ElementwiseActivation {
    @Override
    public ToDoubleFunction<Double> getActivation() {
        return x -> x;
    }

    @Override
    public ToDoubleFunction<Double> getTransferDerivative() {
        return x -> 1;
    }
}
package activations;

import java.util.function.ToDoubleFunction;

public class Linear extends ElementwiseActivation {
    @Override
    public ToDoubleFunction<Double> getEActivation() {
        return x -> x;
    }

    @Override
    public ToDoubleFunction<Double> getETransferDerivative() {
        return x -> 1;
    }
}
package activations;

import java.util.function.ToDoubleFunction;

public class ReLU extends Activation {
    @Override
    public ToDoubleFunction<Double> getActivation() {
        return x -> Math.max(0, x);
    }

    @Override
    public ToDoubleFunction<Double> getTransferDerivative() {
        return x -> (x < 0 ? 0 : 1);
    }
}

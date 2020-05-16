package activations;

import java.util.function.ToDoubleFunction;

public class Softmax extends ElementwiseActivation {

    @Override
    public ToDoubleFunction<Double> getActivation() {
        return null;
    }

    @Override
    public ToDoubleFunction<Double> getTransferDerivative() {
        return null;
    }
}

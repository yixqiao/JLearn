package activations;

import java.util.function.ToDoubleFunction;

public abstract class Activation {
    public abstract ToDoubleFunction<Double> getActivation();
    public abstract ToDoubleFunction<Double> getTransferDerivative();
}

package activations;

import core.Matrix;

import java.util.function.ToDoubleFunction;

public abstract class ElementwiseActivation{
    public abstract ToDoubleFunction<Double> getActivation();
    public abstract ToDoubleFunction<Double> getTransferDerivative();
}

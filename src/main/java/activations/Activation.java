package activations;

import core.Matrix;

import java.util.function.ToDoubleFunction;

public abstract class Activation {
    public abstract ToDoubleFunction<Matrix> getActivation();
    public abstract ToDoubleFunction<Matrix> getTransferDerivative();
}

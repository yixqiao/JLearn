package activations;

import core.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

public abstract class Activation {
    public abstract Consumer<Matrix> getActivation();

    public abstract Function<Matrix, Matrix> getTransferDerivative();
}

package activations;

import core.Matrix;

import java.util.function.Consumer;
import java.util.function.Function;

public class Softmax extends Activation {

    @Override
    public Consumer<Matrix> getActivation() {
        return null;
    }

    @Override
    public Consumer<Matrix> getTransferDerivative() {
        return null;
    }
}

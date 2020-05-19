package layers;

import core.Matrix;

import java.util.ArrayList;

public abstract class Layer {
    public abstract Matrix forwardPropagate(Matrix input);

    public abstract Matrix getErrors(Matrix prevErrors);

    public abstract Matrix getErrorsExpected(Matrix expected);

    public abstract void update(Matrix errors, double learningRate);
}

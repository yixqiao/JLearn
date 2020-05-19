package layers;

import core.Matrix;

import java.util.ArrayList;

public abstract class Layer {
    public abstract Matrix forwardPropagate(Matrix input);
    public abstract Matrix getErrors(Matrix neurons, Matrix expected, Matrix prevErrors);
    public abstract Matrix getErrors(Matrix neurons, Matrix expected);
}

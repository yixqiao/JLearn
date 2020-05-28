package me.yixqiao.jlearn.layers;

import me.yixqiao.jlearn.activations.Activation;
import me.yixqiao.jlearn.activations.Linear;
import me.yixqiao.jlearn.matrix.Matrix;

public class InputLayer extends Layer {
    private int size;

    public InputLayer(int size){
        this.size = size;
    }

    @Override
    public void initLayer(int prevSize, Activation prevActivation) {

    }

    @Override
    public Activation getActivation() {
        return new Linear();
    }


    @Override
    public int getOutSize() {
        return size;
    }

    @Override
    public Matrix forwardPropagate(Matrix input) {
        return input;
    }

    @Override
    public Matrix getErrors(Matrix prevErrors) {
        return null;
    }

    @Override
    public Matrix getErrorsExpected(Matrix expected) {
        return null;
    }

    @Override
    public void update(Matrix errors, double learningRate) {

    }
}

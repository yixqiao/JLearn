package me.yixqiao.jlearn.exceptions;

/**
 * Exception in neural network.
 */
public class NeuralNetworkException extends RuntimeException {
    public NeuralNetworkException() {
        super();
    }

    public NeuralNetworkException(String errorMessage) {
        super(errorMessage);
    }
}

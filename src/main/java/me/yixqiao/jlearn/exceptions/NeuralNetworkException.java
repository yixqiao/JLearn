package me.yixqiao.jlearn.exceptions;

public class NeuralNetworkException extends RuntimeException {
    public NeuralNetworkException() {
        super();
    }

    public NeuralNetworkException(String errorMessage) {
        super(errorMessage);
    }
}

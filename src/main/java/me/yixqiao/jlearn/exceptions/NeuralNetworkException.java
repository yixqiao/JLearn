package me.yixqiao.jlearn.exceptions;

/**
 * Exception in neural network.
 */
public class NeuralNetworkException extends RuntimeException {
    /**
     * Create an exception.
     */
    public NeuralNetworkException() {
        super();
    }

    /**
     * Create an exception.
     *
     * @param errorMessage custom error message
     */
    public NeuralNetworkException(String errorMessage) {
        super(errorMessage);
    }
}
